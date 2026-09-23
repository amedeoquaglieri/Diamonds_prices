# Diamonds prices

Modelling diamond price from the Kaggle diamonds dataset (`diamonds.csv`,
53,940 rows).

Carat dominates price and confounds the other 4Cs, so the model controls for
it explicitly. See `planning/report.html` for the exploratory analysis and
`planning/PLAN.md` for the modelling plan and build steps.

## Setup

```
uv sync
```

## Layout

- `diamonds.csv` — raw dataset.
- `src/diamonds/` — package code.
- `planning/` — EDA report and modelling plan.
