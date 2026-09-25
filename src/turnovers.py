"""Every open-play turnover with the tracking window around it, seen from the team that LOST the ball.

A turnover is a player_possession ending in `possession_loss` or an unsuccessful pass. The moment
and place of the loss are taken from the rival's FIRST controlled possession afterwards (for a
misplaced pass the event's own x/y is where the passer stood, not where the ball was won). Only
gains within MAX_GAP_S of the loss are kept (the rest are stoppages: ball out, fouls).

Coordinates are rotated so the team that lost the ball always attacks towards +x: a loss at
x=+30 happened near the rival's goal, the losing team's own goal is at x=-52.5.
"""
import glob
import json

import numpy as np
import pandas as pd

FPS = 10
MAX_GAP_S = 3.0
BEFORE_S, AFTER_S = 1.0, 5.0


def _match_turnovers(data_dir, mid):
    ev = pd.read_csv(f"{data_dir}/{mid}/{mid}_dynamic_events.csv", low_memory=False, usecols=[
        "event_type", "end_type", "pass_outcome", "team_id", "player_id", "frame_start", "frame_end",
        "x_start", "y_start", "attacking_side", "period", "phase_index",
    ])
    pp = ev[ev["event_type"] == "player_possession"].sort_values("frame_start").reset_index(drop=True)
    is_loss = (pp["end_type"] == "possession_loss") | ((pp["end_type"] == "pass") & (pp["pass_outcome"] == "unsuccessful"))

    rows = []
    for i in np.flatnonzero(is_loss.to_numpy()):
        lost = pp.iloc[i]
        nxt = pp.iloc[i + 1:]
        gain = nxt[(nxt["team_id"] != lost["team_id"])].head(1)
        between = nxt[nxt["frame_start"] < (gain["frame_start"].iloc[0] if len(gain) else np.inf)]
        if gain.empty or len(between) or gain["period"].iloc[0] != lost["period"]:
            continue  # no rival gain, the losing team touched it first, or across half-time
        g = gain.iloc[0]
        if (g["frame_start"] - lost["frame_end"]) / FPS > MAX_GAP_S:
            continue
        rows.append({
            "match_id": mid,
            "loss_type": "possession_loss" if lost["end_type"] == "possession_loss" else "misplaced_pass",
            "losing_team_id": lost["team_id"],
            "gaining_team_id": g["team_id"],
            "gain_frame": int(g["frame_start"]),
            # rival's own attack-normalised coords -> rotate into the LOSING team's frame
            "loss_x": -g["x_start"],
            "loss_y": -g["y_start"],
            "losing_attacking_side": lost["attacking_side"],
            "period": lost["period"],
        })
    return pd.DataFrame(rows)


def build_turnover_tables(data_dir):
    match_dirs = sorted(glob.glob(f"{data_dir}/*"))
    match_ids = [d.replace("\\", "/").split("/")[-1] for d in match_dirs]

    to_frames, traj_frames = [], []
    for mid in match_ids:
        to = _match_turnovers(data_dir, mid)
        if to.empty:
            continue
        to["turnover_id"] = [f"{mid}_{k}" for k in range(len(to))]
        to_frames.append(to)

        with open(f"{data_dir}/{mid}/{mid}_match.json", encoding="utf-8") as f:
            m = json.load(f)
        player_team = {p["id"]: p["team_id"] for p in m["players"]}
        goalkeepers = {p["id"] for p in m["players"] if p["player_role"]["name"] == "Goalkeeper"}

        wanted = {}
        for _, r in to.iterrows():
            for fr in range(r["gain_frame"] - int(BEFORE_S * FPS), r["gain_frame"] + int(AFTER_S * FPS) + 1):
                wanted.setdefault(fr, []).append((r["turnover_id"], r["losing_team_id"], r["gain_frame"],
                                                  1 if r["losing_attacking_side"] == "left_to_right" else -1))
        rows = []
        with open(f"{data_dir}/{mid}/{mid}_tracking_extrapolated.jsonl", encoding="utf-8") as fh:
            for line in fh:
                fr = int(line.split('"frame":', 1)[1].split(",", 1)[0])
                if fr not in wanted:
                    continue
                d = json.loads(line)
                for tid, losing_team, gain_frame, sgn in wanted[fr]:
                    t = (fr - gain_frame) / FPS
                    for p in d["player_data"]:
                        team = player_team.get(p["player_id"])
                        role = "losing" if team == losing_team else "gaining"
                        rows.append((tid, fr, t, p["player_id"], role, p["player_id"] in goalkeepers, p["x"] * sgn, p["y"] * sgn))
                    if d["ball_data"]["x"] is not None:
                        rows.append((tid, fr, t, None, "ball", False, d["ball_data"]["x"] * sgn, d["ball_data"]["y"] * sgn))
        traj_frames.append(pd.DataFrame(rows, columns=["turnover_id", "frame", "t", "player_id", "role", "is_goalkeeper", "x", "y"]))

    return pd.concat(to_frames, ignore_index=True), pd.concat(traj_frames, ignore_index=True)


def load_or_build_turnovers(data_dir, cache_dir):
    """Cached tables; derived SkillCorner data, so the cache stays gitignored."""
    from pathlib import Path

    cache = Path(cache_dir)
    to_path, tr_path = cache / "turnovers.parquet", cache / "turnover_trajectories.parquet"
    if to_path.exists() and tr_path.exists():
        return pd.read_parquet(to_path), pd.read_parquet(tr_path)
    turnovers, trajectories = build_turnover_tables(data_dir)
    cache.mkdir(parents=True, exist_ok=True)
    turnovers.to_parquet(to_path)
    trajectories.to_parquet(tr_path)
    return turnovers, trajectories
