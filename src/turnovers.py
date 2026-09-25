"""Every open-play turnover with the tracking window around it, seen from the team that LOST the ball.

A turnover starts from a player_possession ending in `possession_loss` or an unsuccessful pass, and
comes in two kinds:

* rival_controlled=True -- the rival's next event is a controlled possession within MAX_GAP_S.
  Moment/place = that first rival possession (for a misplaced pass the event's own x/y is where
  the passer stood, not where the ball was won).
* rival_controlled=False -- a teammate is the next to control the ball, within MAX_RECOVER_S, but a
  rival got within CONTEST_M of the ball in between (the ball really was contested: an immediate
  recovery, often the most successful counter-presses). Moment/place = the first frame of that
  contest, from tracking. If no rival ever got that close, it was an imprecise pass collected by a
  teammate -- never a loss -- and it is dropped.

Everything else (rival control after MAX_GAP_S, i.e. mostly ball out of play) is a stoppage, not an
open-play loss. Coordinates are rotated so the team that lost the ball always attacks towards +x: a
loss at x=+30 happened near the rival's goal; the losing team's own goal is at x=-52.5.
"""
import glob
import json

import numpy as np
import pandas as pd

FPS = 10
MAX_GAP_S = 3.0
MAX_RECOVER_S = 5.0
CONTEST_M = 1.5
BEFORE_S, AFTER_S = 1.0, 5.0


def _match_candidates(data_dir, mid):
    ev = pd.read_csv(f"{data_dir}/{mid}/{mid}_dynamic_events.csv", low_memory=False, usecols=[
        "event_type", "end_type", "pass_outcome", "team_id", "frame_start", "frame_end",
        "x_start", "y_start", "attacking_side", "period",
    ])
    pp = ev[ev["event_type"] == "player_possession"].sort_values("frame_start").reset_index(drop=True)
    is_loss = (pp["end_type"] == "possession_loss") | ((pp["end_type"] == "pass") & (pp["pass_outcome"] == "unsuccessful"))

    controlled, contested = [], []
    for i in np.flatnonzero(is_loss.to_numpy()):
        lost = pp.iloc[i]
        if i + 1 >= len(pp):
            continue
        nxt = pp.iloc[i + 1]
        if nxt["period"] != lost["period"]:
            continue
        base = {
            "match_id": mid,
            "loss_type": "possession_loss" if lost["end_type"] == "possession_loss" else "misplaced_pass",
            "losing_team_id": lost["team_id"],
            "losing_attacking_side": lost["attacking_side"],
            "period": lost["period"],
        }
        gap = (nxt["frame_start"] - lost["frame_end"]) / FPS
        if nxt["team_id"] != lost["team_id"]:
            if gap <= MAX_GAP_S:
                controlled.append(base | {
                    "rival_controlled": True,
                    "gaining_team_id": nxt["team_id"],
                    "gain_frame": int(nxt["frame_start"]),
                    # rival's own attack-normalised coords -> rotate into the LOSING team's frame
                    "loss_x": -nxt["x_start"],
                    "loss_y": -nxt["y_start"],
                })
        elif gap <= MAX_RECOVER_S:
            contested.append(base | {
                "rival_controlled": False,
                "search_from": int(lost["frame_end"]),
                "search_to": int(nxt["frame_start"]),
            })
    return pd.DataFrame(controlled), pd.DataFrame(contested)


def build_turnover_tables(data_dir):
    match_dirs = sorted(glob.glob(f"{data_dir}/*"))
    match_ids = [d.replace("\\", "/").split("/")[-1] for d in match_dirs]
    before, after = int(BEFORE_S * FPS), int(AFTER_S * FPS)

    to_frames, traj_frames = [], []
    for mid in match_ids:
        controlled, contested = _match_candidates(data_dir, mid)
        with open(f"{data_dir}/{mid}/{mid}_match.json", encoding="utf-8") as f:
            m = json.load(f)
        player_team = {p["id"]: p["team_id"] for p in m["players"]}
        goalkeepers = {p["id"] for p in m["players"] if p["player_role"]["name"] == "Goalkeeper"}

        # one pass over the tracking file keeps every frame any candidate could need
        needed = set()
        for _, r in controlled.iterrows():
            needed.update(range(r["gain_frame"] - before, r["gain_frame"] + after + 1))
        for _, r in contested.iterrows():
            needed.update(range(r["search_from"] - before, r["search_to"] + after + 1))
        frames = {}
        with open(f"{data_dir}/{mid}/{mid}_tracking_extrapolated.jsonl", encoding="utf-8") as fh:
            for line in fh:
                fr = int(line.split('"frame":', 1)[1].split(",", 1)[0])
                if fr in needed:
                    frames[fr] = json.loads(line)

        # contested candidates: first frame a rival is within CONTEST_M of the ball
        kept_contested = []
        for _, r in contested.iterrows():
            for fr in range(r["search_from"], r["search_to"] + 1):
                d = frames.get(fr)
                if d is None or d["ball_data"]["x"] is None:
                    continue
                bx, by = d["ball_data"]["x"], d["ball_data"]["y"]
                rivals = [p for p in d["player_data"] if player_team.get(p["player_id"]) not in (None, r["losing_team_id"])]
                if rivals and min(np.hypot(p["x"] - bx, p["y"] - by) for p in rivals) <= CONTEST_M:
                    sgn = 1 if r["losing_attacking_side"] == "left_to_right" else -1
                    gaining = next(t for t in (m["home_team"]["id"], m["away_team"]["id"]) if t != r["losing_team_id"])
                    kept_contested.append({k: r[k] for k in ["match_id", "loss_type", "losing_team_id", "losing_attacking_side", "period", "rival_controlled"]}
                                          | {"gaining_team_id": gaining, "gain_frame": fr, "loss_x": bx * sgn, "loss_y": by * sgn})
                    break
        to = pd.concat([controlled, pd.DataFrame(kept_contested)], ignore_index=True)
        if to.empty:
            continue
        to = to.sort_values("gain_frame").reset_index(drop=True)
        to["turnover_id"] = [f"{mid}_{k}" for k in range(len(to))]
        to_frames.append(to)

        rows = []
        for _, r in to.iterrows():
            sgn = 1 if r["losing_attacking_side"] == "left_to_right" else -1
            for fr in range(r["gain_frame"] - before, r["gain_frame"] + after + 1):
                d = frames.get(fr)
                if d is None:
                    continue
                t = (fr - r["gain_frame"]) / FPS
                for p in d["player_data"]:
                    role = "losing" if player_team.get(p["player_id"]) == r["losing_team_id"] else "gaining"
                    rows.append((r["turnover_id"], fr, t, p["player_id"], role, p["player_id"] in goalkeepers, p["x"] * sgn, p["y"] * sgn))
                if d["ball_data"]["x"] is not None:
                    rows.append((r["turnover_id"], fr, t, None, "ball", False, d["ball_data"]["x"] * sgn, d["ball_data"]["y"] * sgn))
        traj_frames.append(pd.DataFrame(rows, columns=["turnover_id", "frame", "t", "player_id", "role", "is_goalkeeper", "x", "y"]))

    return pd.concat(to_frames, ignore_index=True), pd.concat(traj_frames, ignore_index=True)


def compute_turnover_outcomes(data_dir, turnovers, window_s=15):
    """What the rival did before the losing team touched the ball again.

    regain_s: seconds until the losing team's next controlled possession. shot/goal: the rival shot
    (scored) within `window_s` of the loss and before that regain. Goal flags come from shot rows'
    `lead_to_goal`, validated in notebook 07 against the real scorelines (64 of 66 goals; the two
    missing are not open-play shots by the scoring team).
    """
    out = []
    for mid, grp in turnovers.groupby("match_id"):
        ev = pd.read_csv(f"{data_dir}/{mid}/{mid}_dynamic_events.csv", low_memory=False,
                         usecols=["event_type", "end_type", "team_id", "frame_start", "frame_end", "lead_to_goal"])
        p = ev[ev["event_type"] == "player_possession"].sort_values("frame_start")
        for _, r in grp.iterrows():
            after = p[p["frame_start"] >= r["gain_frame"]]
            back = after[after["team_id"] == r["losing_team_id"]].head(1)
            regain = back["frame_start"].iloc[0] if len(back) else np.inf
            rival = after[(after["team_id"] == r["gaining_team_id"]) & (after["frame_start"] < regain)]
            shots = rival[(rival["end_type"] == "shot") & (rival["frame_end"] <= r["gain_frame"] + window_s * FPS)]
            out.append({"turnover_id": r["turnover_id"], "regain_s": (regain - r["gain_frame"]) / FPS,
                        "shot_15s": len(shots) > 0, "goal_15s": bool((shots["lead_to_goal"] == True).any())})
    res = pd.DataFrame(out)
    res["regained_5s"] = res["regain_s"] <= 5
    return res


def player_reactions(trajectories, horizon_s=2.0, press_move=3.0, press_near=5.0, retreat_back=3.0):
    """Each outfield player of the losing team: press / retreat / hold over the first `horizon_s`.

    press: the player HIMSELF moved >= press_move metres towards where the ball went AND ended within
    press_near metres of it. (Measuring 'distance to the ball shrank' instead is wrong: in 2 s the ball
    travels ~9 m vs ~4 m for players, so a ball cleared towards a centre-back labels him a presser --
    shown and validated against SkillCorner engagement events in notebook 07.)
    retreat: not pressing and moved >= retreat_back metres towards his own goal (-x).
    """
    a = trajectories[np.isclose(trajectories["t"], 0.0)]
    b = trajectories[np.isclose(trajectories["t"], horizon_s)]
    ball0 = a[a["role"] == "ball"][["turnover_id", "x", "y"]].rename(columns={"x": "bx0", "y": "by0"})
    ballH = b[b["role"] == "ball"][["turnover_id", "x", "y"]].rename(columns={"x": "bxH", "y": "byH"})
    p0 = a[(a["role"] == "losing") & ~a["is_goalkeeper"]][["turnover_id", "player_id", "x", "y"]]
    pH = b[(b["role"] == "losing") & ~b["is_goalkeeper"]][["turnover_id", "player_id", "x", "y"]]
    pl = p0.merge(pH, on=["turnover_id", "player_id"], suffixes=("0", "H")).merge(ball0, on="turnover_id").merge(ballH, on="turnover_id")
    pl["d0"] = np.hypot(pl["x0"] - pl["bx0"], pl["y0"] - pl["by0"])
    pl["dH"] = np.hypot(pl["xH"] - pl["bxH"], pl["yH"] - pl["byH"])
    ux, uy = pl["bxH"] - pl["x0"], pl["byH"] - pl["y0"]
    pl["approach"] = ((pl["xH"] - pl["x0"]) * ux + (pl["yH"] - pl["y0"]) * uy) / np.hypot(ux, uy).replace(0, np.nan)
    pl["dx"] = pl["xH"] - pl["x0"]
    press = (pl["approach"] >= press_move) & (pl["dH"] <= press_near)
    retreat = ~press & (pl["dx"] <= -retreat_back)
    pl["action"] = np.select([press, retreat], ["press", "retreat"], "hold")
    return pl


def players_near_ball(trajectories, radius=10.0, t=0.0):
    """Outfield players of each team within `radius` metres of the ball at time t after the loss."""
    a = trajectories[np.isclose(trajectories["t"], t)]
    ball = a[a["role"] == "ball"][["turnover_id", "x", "y"]].rename(columns={"x": "bx", "y": "by"})
    p = a[(a["role"] != "ball") & ~a["is_goalkeeper"]].merge(ball, on="turnover_id")
    p = p[np.hypot(p["x"] - p["bx"], p["y"] - p["by"]) <= radius]
    cnt = p.groupby(["turnover_id", "role"]).size().unstack(fill_value=0)
    cnt = cnt.rename(columns={"losing": "own_near", "gaining": "rival_near"})
    return cnt.reindex(columns=["own_near", "rival_near"], fill_value=0).reindex(ball["turnover_id"], fill_value=0).reset_index()


def load_player_roles(data_dir):
    """One row per (match, player): team and tactical role from match.json."""
    rows = []
    for d in sorted(glob.glob(f"{data_dir}/*")):
        mid = d.replace("\\", "/").split("/")[-1]
        with open(f"{d}/{mid}_match.json", encoding="utf-8") as f:
            m = json.load(f)
        for p in m["players"]:
            rows.append({"match_id": mid, "player_id": p["id"], "team_id": p["team_id"],
                         "role_name": p["player_role"]["name"], "position_group": p["player_role"]["position_group"]})
    return pd.DataFrame(rows)


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
