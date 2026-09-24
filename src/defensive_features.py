"""Team-level defensive/pressing feature table built from the 20-match tracking sample.

Shared by notebooks 02 and 03 so the join logic (phase_index <-> phases_of_play,
defending-team resolution per event type) lives in one place. See
notebooks/02_low_vs_high_concede_patterns.ipynb for how each field was derived and validated.
"""
import glob
import json

import numpy as np
import pandas as pd

DYNAMIC_EVENT_COLS = [
    "match_id", "event_type", "event_subtype", "team_id",
    "xloss_player_possession_start", "xloss_player_possession_end",
    "organised_defense", "defensive_structure", "n_defensive_lines",
    "last_defensive_line_height_start", "overall_pressure_start",
]


def load_match_meta(data_dir):
    match_dirs = sorted(glob.glob(f"{data_dir}/*"))
    match_ids = [d.replace("\\", "/").split("/")[-1] for d in match_dirs]

    match_meta, team_names = {}, {}
    for mid in match_ids:
        with open(f"{data_dir}/{mid}/{mid}_match.json", encoding="utf-8") as f:
            m = json.load(f)
        home_id, away_id = m["home_team"]["id"], m["away_team"]["id"]
        match_meta[mid] = {
            "home_id": home_id, "away_id": away_id,
            "home_name": m["home_team"]["short_name"], "away_name": m["away_team"]["short_name"],
            "home_score": m["home_team_score"], "away_score": m["away_team_score"],
        }
        team_names[home_id] = m["home_team"]["short_name"]
        team_names[away_id] = m["away_team"]["short_name"]
    return match_ids, match_meta, team_names


def build_team_summary(match_meta):
    records = []
    for mid, mm in match_meta.items():
        records.append({"team": mm["home_name"], "match_id": mid, "conceded": mm["away_score"], "scored": mm["home_score"]})
        records.append({"team": mm["away_name"], "match_id": mid, "conceded": mm["home_score"], "scored": mm["away_score"]})
    team_games = pd.DataFrame(records)
    league_avg_conceded = team_games["conceded"].mean()

    team_summary = team_games.groupby("team").agg(
        n_games=("match_id", "size"),
        goals_conceded=("conceded", "sum"),
        goals_scored=("scored", "sum"),
    )
    team_summary["conceded_per_game"] = team_summary["goals_conceded"] / team_summary["n_games"]
    team_summary["scored_per_game"] = team_summary["goals_scored"] / team_summary["n_games"]
    team_summary["group"] = np.where(team_summary["conceded_per_game"] < league_avg_conceded, "low_concede", "high_concede")
    return team_summary, league_avg_conceded


def build_team_defensive_features(data_dir):
    """Returns one row per team: goals, block shape, pressing subtype mix, defensive shape."""
    match_ids, match_meta, team_names = load_match_meta(data_dir)
    team_summary, _ = build_team_summary(match_meta)

    phase_frames, engagement_frames, possession_frames = [], [], []
    for mid in match_ids:
        mm = match_meta[mid]
        other_team = {mm["home_id"]: mm["away_id"], mm["away_id"]: mm["home_id"]}

        ph = pd.read_csv(f"{data_dir}/{mid}/{mid}_phases_of_play.csv")
        ph["defending_team"] = ph["team_in_possession_id"].map(other_team).map(team_names)
        ph["match_id"] = mid
        phase_frames.append(ph)

        dyn = pd.read_csv(f"{data_dir}/{mid}/{mid}_dynamic_events.csv", usecols=lambda c: c in DYNAMIC_EVENT_COLS, low_memory=False)
        dyn["match_id"] = mid

        obe = dyn[dyn["event_type"] == "on_ball_engagement"].copy()
        obe["defending_team"] = obe["team_id"].map(team_names)
        engagement_frames.append(obe)

        pp = dyn[dyn["event_type"] == "player_possession"].copy()
        pp["defending_team"] = pp["team_id"].map(other_team).map(team_names)
        possession_frames.append(pp)

    phases = pd.concat(phase_frames, ignore_index=True)
    engagements = pd.concat(engagement_frames, ignore_index=True)
    possessions = pd.concat(possession_frames, ignore_index=True)

    phases["def_area_end"] = phases["team_out_of_possession_width_end"] * phases["team_out_of_possession_length_end"]
    block_share = (
        phases.groupby("defending_team")["team_out_of_possession_phase_type"]
        .value_counts(normalize=True).unstack(fill_value=0)
        .add_prefix("block_share_")
    )

    engagements["xloss_delta"] = engagements["xloss_player_possession_end"] - engagements["xloss_player_possession_start"]
    n_opponent_touches = possessions.groupby("defending_team").size().rename("n_opponent_touches")
    press = engagements.groupby("defending_team").agg(
        n_engagements=("event_subtype", "size"),
        mean_xloss_delta_caused=("xloss_delta", "mean"),
        share_pressing=("event_subtype", lambda s: (s == "pressing").mean()),
        share_counter_press=("event_subtype", lambda s: (s == "counter_press").mean()),
        share_recovery_press=("event_subtype", lambda s: (s == "recovery_press").mean()),
    ).join(n_opponent_touches)
    press["engagements_per_100_opp_touches"] = 100 * press["n_engagements"] / press["n_opponent_touches"]

    possessions["organised_defense"] = possessions["organised_defense"].astype(float)
    shape = possessions.groupby("defending_team").agg(
        organised_defense_rate=("organised_defense", "mean"),
        mean_n_defensive_lines=("n_defensive_lines", "mean"),
        mean_last_line_height=("last_defensive_line_height_start", "mean"),
    )
    shape["share_high_pressure_applied"] = (
        possessions.groupby("defending_team")["overall_pressure_start"]
        .apply(lambda s: s.isin(["high_pressure", "very_high_pressure"]).mean())
    )

    return team_summary.join(block_share).join(press).join(shape)


PHASE_MODEL_DYNAMIC_COLS = [
    "match_id", "event_type", "event_subtype", "team_id", "phase_index",
    "last_defensive_line_height_start", "game_state",
]


def build_phase_level_table(data_dir):
    """One row per defensive phase (n~8,834), with within-phase event flags for modeling.

    Unlike build_team_defensive_features (one row per team, n=13), this keeps every phase as
    its own observation -- the unit needed for a properly powered controlled model. See
    notebooks/04_controlled_model.ipynb for how each column is used and validated.
    """
    match_ids, match_meta, team_names = load_match_meta(data_dir)

    phase_frames = []
    eng_agg_frames, run_agg_frames, lh_frames, gs_frames = [], [], [], []
    for mid in match_ids:
        mm = match_meta[mid]
        other = {mm["home_id"]: mm["away_id"], mm["away_id"]: mm["home_id"]}

        ph = pd.read_csv(f"{data_dir}/{mid}/{mid}_phases_of_play.csv")
        ph["defending_team"] = ph["team_in_possession_id"].map(other).map(team_names)
        ph["match_id"] = mid
        phase_frames.append(ph)

        dyn = pd.read_csv(f"{data_dir}/{mid}/{mid}_dynamic_events.csv", usecols=lambda c: c in PHASE_MODEL_DYNAMIC_COLS, low_memory=False)
        dyn["match_id"] = mid

        obe = dyn[dyn["event_type"] == "on_ball_engagement"]
        eng_agg_frames.append(obe.groupby(["match_id", "phase_index"]).agg(
            n_engagements_in_phase=("event_subtype", "size"),
            has_pressing=("event_subtype", lambda s: (s == "pressing").any()),
            has_counter_or_recovery_press=("event_subtype", lambda s: s.isin(["counter_press", "recovery_press"]).any()),
        ).reset_index())

        obr = dyn[dyn["event_type"] == "off_ball_run"]
        run_agg_frames.append(obr.groupby(["match_id", "phase_index"]).agg(
            n_runs_in_phase=("event_subtype", "size"),
            has_behind_run=("event_subtype", lambda s: (s == "behind").any()),
        ).reset_index())

        pp = dyn[dyn["event_type"] == "player_possession"]
        lh_frames.append(pp.groupby(["match_id", "phase_index"]).agg(
            mean_last_line_height=("last_defensive_line_height_start", "mean"),
        ).reset_index())

        gs_frames.append(
            dyn.dropna(subset=["game_state"]).groupby(["match_id", "phase_index"])["game_state"].first().reset_index()
        )

    phases = pd.concat(phase_frames, ignore_index=True)
    eng_agg = pd.concat(eng_agg_frames, ignore_index=True)
    run_agg = pd.concat(run_agg_frames, ignore_index=True)
    lh_agg = pd.concat(lh_frames, ignore_index=True)
    gs_agg = pd.concat(gs_frames, ignore_index=True)

    model_df = phases.merge(eng_agg, left_on=["match_id", "index"], right_on=["match_id", "phase_index"], how="left")
    model_df = model_df.merge(run_agg, left_on=["match_id", "index"], right_on=["match_id", "phase_index"], how="left", suffixes=("", "_run"))
    model_df = model_df.merge(lh_agg, left_on=["match_id", "index"], right_on=["match_id", "phase_index"], how="left", suffixes=("", "_lh"))
    model_df = model_df.merge(gs_agg, left_on=["match_id", "index"], right_on=["match_id", "phase_index"], how="left", suffixes=("", "_gs"))

    # NaN here is structural ("no such event happened in this phase"), not missing data -- verified in Notebook 4.
    for c in ["has_pressing", "has_counter_or_recovery_press", "has_behind_run"]:
        model_df[c] = model_df[c].fillna(False)

    model_df["def_area_end"] = model_df["team_out_of_possession_width_end"] * model_df["team_out_of_possession_length_end"]
    return model_df
