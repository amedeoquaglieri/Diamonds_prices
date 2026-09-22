# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## Project overview

This repo builds a model of diamond price from the Kaggle diamonds dataset
(`diamonds.csv`, 53,940 rows, columns: `carat`, `cut`, `color`, `clarity`,
`depth`, `table`, `price`, `x`, `y`, `z`).

Current state: **exploratory analysis and planning only — no model code has
been implemented yet.**

- `diamonds.csv` — raw dataset.
- `planning/report.html` — exploratory data analysis (EDA) of the dataset.
- `planning/PLAN.md` — the modeling plan derived from that EDA. Read this
  file in full before writing any model code; it is the source of truth for
  *why* the approach below is structured this way.

## Key findings from the EDA/plan (see `planning/PLAN.md` for detail)

- **Carat is a confounder.** Carat correlates with price at r=0.92 (raw),
  r=0.97 in log-log space (`price ∝ carat^1.68`), and dominates every other
  feature. Raw, unconditional average price by cut/color/clarity moves in
  the *wrong* direction (e.g. better clarity looks cheaper) because
  higher-grade stones skew smaller in this dataset. Any model of price on
  cut/color/clarity/dimensions must explicitly control for carat, or the
  coefficients on the other 4Cs will be biased and can flip sign.
- **Model `log(price)`, not raw `price`** — matches the power-law
  relationship with carat and stabilizes the right-skewed price
  distribution (skew 1.62 on raw price).
- **x, y, z are near-redundant with carat** (r≥0.97 with each other and with
  carat). Use at most one of {carat, x, y, z} as the size feature to avoid
  multicollinearity.

## Data preparation (before modeling)

Apply these cleaning steps identified in the EDA (net result: ~53,766 usable
rows):

1. Drop/repair 20 rows with a zero x/y/z dimension.
2. Drop/repair 6 rows with an implausible y or z (>10mm) — decimal-point
   typos.
3. Deduplicate — 289 rows are involved in exact duplicates; keep one copy of
   each.
4. Handle extreme depth (≥75%) / table (≥80%) outlier rows.

## Modeling approach

Two complementary ways to control for carat (do both):

1. **Regression with carat as a covariate**: fit
   `log(price) ~ log(carat) + cut + color + clarity + depth + table`. Start
   with linear regression/OLS, then compare against a tree-based model.
2. **Carat-bucketed comparison**: bucket carat into bands (e.g. 0.2–0.4,
   0.4–0.7, 0.7–1.0, 1.0–1.5, 1.5+) and compute mean price / price-per-carat
   by cut/color/clarity within each band, as an interpretable sanity check
   against the regression.

Candidate model types, in order of complexity: linear regression on
log(price) (baseline) → regularized regression (Ridge/Lasso) if x/y/z are
kept alongside carat → gradient-boosted trees (XGBoost/LightGBM) as a
stronger predictive baseline.

Encode `cut`, `color`, `clarity` as ordinal (they have a natural
worst→best order); also compare against one-hot encoding since the true
price step between grades isn't necessarily linear.

## Evaluation

- 80/20 train/test split, stratified by carat band.
- Metrics: RMSE and MAE on price (converted back from log-price), R² on
  log(price). Report overall and broken out by carat band.
- Sanity checks: fitted coefficients (or partial-dependence plots for tree
  models) on cut/color/clarity should be monotonic in the expected
  direction once carat is controlled for; residuals vs. carat should not
  show a strong remaining trend.

## Working conventions

- There is no established source layout, package structure, or dependency
  manifest yet — when adding the first model code, choose a conventional,
  minimal structure (e.g. a `src/` or top-level package plus a
  `requirements.txt`/`pyproject.toml`) rather than inferring one from
  nonexistent precedent, and keep it consistent thereafter.
- Keep `planning/PLAN.md` and this file in sync: if the modeling approach
  changes materially during implementation, update both.
- Don't modify `planning/report.html` or `diamonds.csv`; they are fixed
  inputs (the EDA report and raw dataset, respectively).
