# Data layout

Data is downloaded separately (not committed) from
[SkillCorner/opendata](https://github.com/SkillCorner/opendata) — Australia A-League 2024/25.

Expected layout:

```
data/raw/
  matches/              # match metadata
  tracking/             # XY tracking data (20 games)
  dynamic_events/       # Game Intelligence dynamic events (10 games)
  physical/             # season-level aggregated physical data (175 games)
  pose/                 # body pose data (2 games)
```

Clone the SkillCorner opendata repo and copy/symlink the relevant folders here, e.g.:

```bash
git clone https://github.com/SkillCorner/opendata ../../skillcorner-opendata
```
