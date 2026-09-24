# Analytics Cup 2.0 — Football (Defensive Positioning)

Entry for the [PySport Analytics Cup 2.0](https://pysport.org/analytics-cup), Football challenge,
using SkillCorner's Australia A-League 2024/25 tracking dataset.

## Question

_TBD — pick a specific question about defensive positioning: team shape, compactness,
pressure, marking, or defensive decision-making._

## Data

This project uses the SkillCorner open dataset (Australia A-League 2024/25). Data is **not**
committed to this repo — download it separately from
[SkillCorner/opendata](https://github.com/SkillCorner/opendata) and place it under `data/raw/`
(see `data/README.md` for the expected layout).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Project structure

```
data/           # raw/processed data (gitignored, see data/README.md)
notebooks/      # exploratory analysis
src/            # reusable analysis code
scripts/        # entry-point scripts (e.g. build figures, run pipeline)
```

## Reproducing results

_TBD — instructions to reproduce the final figures/results from a clean environment._

## License

MIT — see [LICENSE.md](LICENSE.md).
