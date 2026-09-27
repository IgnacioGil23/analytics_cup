"""Build the interactive Turnover Cost explorer from YOUR local copy of the SkillCorner open data.

    python scripts/build_explorer.py --data ../skillcorner-opendata/data/matches

Writes output/turnover_cost_explorer.html: a single self-contained page (no server, no internet).
The page embeds derived SkillCorner data (pass positions, SkillCorner's xpass/xthreat), so it is
gitignored like the data itself -- share screenshots, not the file, unless you have the rights to.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pass_risk import fit_lens, load_passing_options, pass_outcomes, score_options  # noqa: E402
from turnovers import load_player_roles  # noqa: E402


def build_payload(data_dir, cache_dir):
    surface, geometry = fit_lens(data_dir, cache_dir)
    opts = score_options(load_passing_options(data_dir), surface, geometry)
    outcomes = pass_outcomes(data_dir)

    meta, names = {}, {}
    for d in sorted(Path(data_dir).glob("*")):
        m = json.loads((d / f"{d.name}_match.json").read_text(encoding="utf-8"))
        h, a = m["home_team"], m["away_team"]
        meta[d.name] = {"label": f"{h['short_name']} {m['home_team_score']}-{m['away_team_score']} {a['short_name']}",
                        "date": m["date_time"][:10]}
        names.update({p["id"]: p["short_name"] for p in m["players"]})
        names.update({h["id"]: h["short_name"], a["id"]: a["short_name"]})
    roles = load_player_roles(data_dir).drop_duplicates(["match_id", "player_id"]).set_index(["match_id", "player_id"])["role_name"]

    passes = []
    for key, g in opts.groupby("pass_key", sort=False):
        chosen = g[g["targeted"] == True]
        if chosen.empty:
            continue
        r = g.iloc[0]
        passes.append({
            "k": key, "m": r["match_id"], "t": names.get(r["team_id"], str(r["team_id"])),
            "p": names.get(r["player_id"], "?"), "role": roles.get((r["match_id"], r["player_id"]), ""),
            "f": int(r["pass_frame"]), "gs": r["game_state"] if isinstance(r["game_state"], str) else "",
            "pr": r["overall_pressure_end"] if isinstance(r["overall_pressure_end"], str) else "",
            "px": round(float(r["px"]), 1), "py": round(float(r["py"]), 1),
            "o": [[round(float(o.tx), 1), round(float(o.ty), 1), round(float(o.xpass), 3),
                   round(float(o.risk) * 1000, 2), round(float(o.reward) * 1000, 2), int(bool(o.targeted))]
                  for o in g.itertuples()],
        })
    pdf = pd.DataFrame(passes)
    out = outcomes.set_index("pass_key")
    pdf["rs"] = pdf["k"].map(out["rival_shot_15s"]).fillna(False).astype(int)
    pdf["os"] = pdf["k"].map(out["own_shot_15s"]).fillna(False).astype(int)

    xs, ys = np.linspace(-52.5, 52.5, 43), np.linspace(-34, 34, 29)
    grid = [[round(float(surface.goals(x, y)) * 100, 3) for x in xs] for y in ys]

    chosen = opts[opts["targeted"] == True].drop_duplicates("pass_key")
    chosen = chosen.assign(team=chosen["team_id"].map(names), own=chosen["px"] < -17.5)
    tm = chosen.groupby("team").agg(matches=("match_id", "nunique"), passes=("pass_key", "size"), risk=("risk", "sum"))
    own = chosen[chosen["own"]].groupby("team")["risk"].sum()
    teams = [{"team": t, "matches": int(r.matches), "passes": int(r.passes),
              "risk_per_match": round(r.risk / r.matches, 3), "risk_per_1000": round(1000 * r.risk / r.passes, 3),
              "own_share": round(float(own.get(t, 0) / r.risk), 3)} for t, r in tm.iterrows()]

    return {"matches": meta, "passes": pdf.to_dict("records"), "grid": {"xs": xs.round(2).tolist(), "ys": ys.round(2).tolist(), "v": grid},
            "teams": sorted(teams, key=lambda t: -t["risk_per_1000"]), "goals_per_shot": round(surface.goals_per_shot, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "skillcorner-opendata" / "data" / "matches"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--out", default=str(ROOT / "output" / "turnover_cost_explorer.html"))
    ap.add_argument("--reuse-data", action="store_true", help="reuse the payload computed on the last run (template tweaks only)")
    a = ap.parse_args()
    payload_path = Path(a.out).with_name("explorer_payload.json")
    if a.reuse_data and payload_path.exists():
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    else:
        payload = build_payload(a.data, a.cache)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(json.dumps(payload, separators=(",", ":"), default=str), encoding="utf-8")
    html = (ROOT / "scripts" / "explorer_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"), default=str))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(html, encoding="utf-8")
    print(f"wrote {a.out} ({len(html) / 1e6:.1f} MB, {len(payload['passes']):,} passes)")


if __name__ == "__main__":
    main()
