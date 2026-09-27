"""Pass risk lens: price the risk of every pass (and every passing option) in expected goals conceded.

    risk   = P(pass fails) * expected cost of losing the ball where that pass would die
    reward = P(pass completes) * xthreat of the target

* P(pass fails) = 1 - SkillCorner `xpass_completion` of the option.
* Cost of losing the ball at (x, y) = P(rival shot within 15 s | loss there) * goals-per-shot after a
  turnover. Fitted on the open-play turnovers of `turnovers.py` with a logit in x and |y| (the simplest
  model that validated best on held-out matches; see notebook 09).
* Where a pass would die is not observable for options that weren't played, so the real failure
  geometries of misplaced passes (fraction of the path travelled, metres off the line) are replayed on
  each option's path and the cost is averaged over them.

Coordinates are SkillCorner dynamic-events coordinates, already attack-normalised for the team in
possession (the attacked goal is at x=+52.5) -- the same frame `turnovers.py` uses.
"""
import glob

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from turnovers import compute_turnover_outcomes

HORIZON_FRAMES = 150  # 15 s at 10 fps


# ---------------------------------------------------------------- cost surface
class CostSurface:
    """P(rival shot within 15 s | ball lost at x, y), and its value in expected goals."""

    def __init__(self, params, goals_per_shot):
        self.params = params
        self.goals_per_shot = goals_per_shot

    @classmethod
    def fit(cls, turnovers_with_outcomes):
        d = turnovers_with_outcomes.assign(shot=lambda t: t["shot_15s"].astype(int), abs_y=lambda t: t["loss_y"].abs())
        m = smf.logit("shot ~ loss_x + abs_y", data=d).fit(disp=0)
        gps = d["goal_15s"].sum() / max(d["shot_15s"].sum(), 1)
        return cls(m.params, gps)

    def p_shot(self, x, y):
        b = self.params
        return 1 / (1 + np.exp(-(b["Intercept"] + b["loss_x"] * np.asarray(x) + b["abs_y"] * np.abs(np.asarray(y)))))

    def goals(self, x, y):
        return self.p_shot(x, y) * self.goals_per_shot


def turnover_training_table(data_dir, turnovers):
    """Turnovers joined with their outcomes (the cost surface's training data)."""
    return turnovers.merge(compute_turnover_outcomes(data_dir, turnovers), on="turnover_id")


# ---------------------------------------------------------------- failure geometry
def failed_pass_geometry(data_dir, turnovers, min_length=3.0):
    """For real misplaced passes: where along the intended path the ball was lost.

    frac_along: 0 = at the passer, 1 = at the intended receiver (>1 = beyond him).
    lateral_m: signed metres off the passer->receiver line.
    """
    mp = turnovers[turnovers["loss_type"] == "misplaced_pass"]
    rows = []
    for mid, grp in mp.groupby("match_id"):
        ev = pd.read_csv(f"{data_dir}/{mid}/{mid}_dynamic_events.csv", low_memory=False, usecols=[
            "event_id", "event_type", "end_type", "pass_outcome", "team_id", "frame_end", "x_end", "y_end",
            "associated_player_possession_event_id", "targeted"])
        pp = ev[(ev["event_type"] == "player_possession") & (ev["end_type"] == "pass") & (ev["pass_outcome"] == "unsuccessful")]
        opt = ev[(ev["event_type"] == "passing_option") & (ev["targeted"] == True)]
        for _, r in grp.iterrows():
            lost = pp[(pp["team_id"] == r["losing_team_id"]) & (pp["frame_end"] <= r["gain_frame"])].sort_values("frame_end").tail(1)
            if lost.empty:
                continue
            tgt = opt[opt["associated_player_possession_event_id"] == lost["event_id"].iloc[0]]
            if tgt.empty:
                continue
            rows.append({"match_id": mid, "turnover_id": r["turnover_id"],
                         "px": lost["x_end"].iloc[0], "py": lost["y_end"].iloc[0],
                         "tx": tgt["x_end"].iloc[0], "ty": tgt["y_end"].iloc[0], "lx": r["loss_x"], "ly": r["loss_y"]})
    g = pd.DataFrame(rows)
    v = g[["tx", "ty"]].to_numpy() - g[["px", "py"]].to_numpy()
    w = g[["lx", "ly"]].to_numpy() - g[["px", "py"]].to_numpy()
    L = np.linalg.norm(v, axis=1)
    g["pass_len"] = L
    g["frac_along"] = (w * v).sum(axis=1) / np.where(L > 0, L**2, np.nan)
    g["lateral_m"] = (v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]) / np.where(L > 0, L, np.nan)
    return g[g["pass_len"] > min_length].reset_index(drop=True)


def expected_loss_cost(px, py, tx, ty, surface, geometry, n_samples=400, seed=0):
    """Average cost (expected goals) of losing the ball over real failure geometries replayed on each path."""
    s = geometry.sample(n=min(n_samples, len(geometry)), random_state=seed)
    fr, lat = s["frac_along"].to_numpy(), s["lateral_m"].to_numpy()
    P = np.stack([np.asarray(px, float), np.asarray(py, float)], axis=1)
    V = np.stack([np.asarray(tx, float), np.asarray(ty, float)], axis=1) - P
    L = np.linalg.norm(V, axis=1, keepdims=True)
    U = np.divide(V, L, out=np.zeros_like(V), where=L > 0)
    N = np.stack([-U[:, 1], U[:, 0]], axis=1)
    lx = np.clip(P[:, [0]] + fr[None] * V[:, [0]] + lat[None] * N[:, [0]], -52.5, 52.5)
    ly = np.clip(P[:, [1]] + fr[None] * V[:, [1]] + lat[None] * N[:, [1]], -34, 34)
    return surface.goals(lx, ly).mean(axis=1)


# ---------------------------------------------------------------- passing options & outcomes
def load_passing_options(data_dir, match_ids=None):
    """Every passing option of every pass, with the passer's position and the option actually chosen."""
    dirs = sorted(glob.glob(f"{data_dir}/*"))
    out = []
    for d in dirs:
        mid = d.replace("\\", "/").split("/")[-1]
        if match_ids is not None and mid not in match_ids:
            continue
        ev = pd.read_csv(f"{d}/{mid}_dynamic_events.csv", low_memory=False, usecols=[
            "event_id", "event_type", "end_type", "team_id", "player_id", "frame_end", "x_end", "y_end",
            "associated_player_possession_event_id", "targeted", "xpass_completion", "xthreat",
            "pass_outcome", "overall_pressure_end", "game_state"])
        ps = ev[(ev["event_type"] == "player_possession") & (ev["end_type"] == "pass")][
            ["event_id", "team_id", "player_id", "frame_end", "x_end", "y_end", "pass_outcome", "overall_pressure_end", "game_state"]]
        ps = ps.rename(columns={"x_end": "px", "y_end": "py", "frame_end": "pass_frame"})
        o = ev[ev["event_type"] == "passing_option"][["associated_player_possession_event_id", "x_end", "y_end", "xpass_completion", "xthreat", "targeted"]]
        o = o.rename(columns={"associated_player_possession_event_id": "event_id", "x_end": "tx", "y_end": "ty", "xpass_completion": "xpass"})
        o = o.merge(ps, on="event_id")
        o["match_id"] = mid
        out.append(o)
    o = pd.concat(out, ignore_index=True)
    o["pass_key"] = o["match_id"] + "|" + o["event_id"]  # event ids repeat across matches
    return o


def pass_outcomes(data_dir, match_ids=None):
    """For each pass played: did the passing team shoot, or the rival shoot, within 15 s?"""
    rows = []
    for d in sorted(glob.glob(f"{data_dir}/*")):
        mid = d.replace("\\", "/").split("/")[-1]
        if match_ids is not None and mid not in match_ids:
            continue
        ev = pd.read_csv(f"{d}/{mid}_dynamic_events.csv", low_memory=False, usecols=["event_id", "event_type", "end_type", "team_id", "frame_start", "frame_end"])
        pp = ev[ev["event_type"] == "player_possession"].sort_values("frame_start")
        for _, r in pp[pp["end_type"] == "pass"].iterrows():
            after = pp[pp["frame_start"] > r["frame_end"]]
            rival, own = after[after["team_id"] != r["team_id"]], after[after["team_id"] == r["team_id"]]
            first_rival = rival["frame_start"].min() if len(rival) else np.inf
            back = own.loc[own["frame_start"] > first_rival, "frame_start"].min() if np.isfinite(first_rival) and len(own) else np.inf
            lim = r["frame_end"] + HORIZON_FRAMES
            rows.append({
                "pass_key": f"{mid}|{r['event_id']}",
                "rival_shot_15s": bool(((rival["end_type"] == "shot") & (rival["frame_end"] <= lim) & (rival["frame_start"] < back)).any()),
                "own_shot_15s": bool(((own["end_type"] == "shot") & (own["frame_end"] <= lim) & (own["frame_start"] < first_rival)).any()),
            })
    return pd.DataFrame(rows)


def score_options(options, surface, geometry, **kw):
    o = options.dropna(subset=["xpass", "xthreat"]).copy()
    o["loss_cost"] = expected_loss_cost(o["px"], o["py"], o["tx"], o["ty"], surface, geometry, **kw)
    o["risk"] = (1 - o["xpass"]) * o["loss_cost"]
    o["reward"] = o["xpass"] * o["xthreat"]
    return o
