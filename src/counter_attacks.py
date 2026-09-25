"""Extracts counter-attack phases (transition/quick_break) with full player trajectories.

A counter-attack here is any phases_of_play row where the ATTACKING team's own phase type
(`team_in_possession_phase_type`) is `transition` or `quick_break` -- the direct opposite
perspective of the `defending_transition`/`defending_quick_break` block types used elsewhere in
this project. 296 such phases exist across the 20 tracked matches (verified in conversation before
writing this module). Each phase's raw tracking window (frame_start..frame_end) is pulled once per
match (a single pass over that match's tracking file, not one pass per phase) and every player is
labelled attacker/defender using match.json's player-to-team roster.
"""
import glob
import json

import pandas as pd

COUNTER_PHASE_TYPES = ["transition", "quick_break"]


def load_player_team_map(data_dir, match_id):
    with open(f"{data_dir}/{match_id}/{match_id}_match.json", encoding="utf-8") as f:
        m = json.load(f)
    return {p["id"]: p["team_id"] for p in m["players"]}, m["home_team"]["id"], m["away_team"]["id"]


def build_counterattack_tables(data_dir):
    """Returns (counterattacks, trajectories).

    counterattacks: one row per phase (~296) with match_id, phase_id, frame window, attacking
        team, outcome (lead_to_shot/lead_to_goal), duration.
    trajectories: long-format (match_id, phase_id, frame, player_id, team_id, role, x, y),
        role in {"attacker", "defender"}, plus one ball row per frame (player_id=None,
        role="ball").
    """
    match_dirs = sorted(glob.glob(f"{data_dir}/*"))
    match_ids = [d.replace("\\", "/").split("/")[-1] for d in match_dirs]

    ca_frames = []
    traj_frames = []

    for mid in match_ids:
        ph = pd.read_csv(f"{data_dir}/{mid}/{mid}_phases_of_play.csv")
        ca = ph[ph["team_in_possession_phase_type"].isin(COUNTER_PHASE_TYPES)].copy()
        if ca.empty:
            continue
        ca["match_id"] = mid
        ca = ca.rename(columns={"index": "phase_id"})
        ca_frames.append(ca[[
            "match_id", "phase_id", "frame_start", "frame_end", "duration",
            "team_in_possession_id", "team_in_possession_phase_type", "attacking_side",
            "team_possession_lead_to_shot", "team_possession_lead_to_goal",
            "team_possession_loss_in_phase",
        ]])

        player_team, home_id, away_id = load_player_team_map(data_dir, mid)

        # frame -> phase_id lookup for this match's counter-attack windows only
        frame_to_phase = {}
        for _, row in ca.iterrows():
            for fr in range(int(row["frame_start"]), int(row["frame_end"]) + 1):
                frame_to_phase[fr] = (row["phase_id"], row["team_in_possession_id"])

        if not frame_to_phase:
            continue

        rows = []
        with open(f"{data_dir}/{mid}/{mid}_tracking_extrapolated.jsonl", encoding="utf-8") as f:
            for line in f:
                # cheap pre-filter on the raw text before json.loads, since most frames aren't in any window
                frame_num_str = line.split('"frame":', 1)[1].split(",", 1)[0].strip()
                if int(frame_num_str) not in frame_to_phase:
                    continue
                d = json.loads(line)
                phase_id, attacking_team_id = frame_to_phase[d["frame"]]
                for p in d["player_data"]:
                    team_id = player_team.get(p["player_id"])
                    role = "attacker" if team_id == attacking_team_id else "defender"
                    rows.append((mid, phase_id, d["frame"], p["player_id"], team_id, role, p["x"], p["y"]))
                if d["ball_data"]["x"] is not None:
                    rows.append((mid, phase_id, d["frame"], None, None, "ball", d["ball_data"]["x"], d["ball_data"]["y"]))

        if rows:
            traj_frames.append(pd.DataFrame(rows, columns=[
                "match_id", "phase_id", "frame", "player_id", "team_id", "role", "x", "y",
            ]))

    counterattacks = pd.concat(ca_frames, ignore_index=True)
    trajectories = pd.concat(traj_frames, ignore_index=True)
    return counterattacks, trajectories
