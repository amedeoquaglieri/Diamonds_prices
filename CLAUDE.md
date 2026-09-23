# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## Project overview

This repo builds a model of diamond price from the Kaggle diamonds dataset
(`diamonds.csv`, 53,940 rows, columns: `carat`, `cut`, `color`, `clarity`,
`depth`, `table`, `price`, `x`, `y`, `z`).

Current state: **build steps 1-4 of `planning/PLAN.md` are done** (project
scaffold, data loading and cleaning, feature engineering, train/test split).
No model code yet.

- `diamonds.csv` — raw dataset.
- `planning/report.html` — exploratory data analysis (EDA) of the dataset.
- `planning/PLAN.md` — the modeling plan derived from that EDA, including
  the ordered build steps and the testing conventions. Read this file in
  full before writing any model code; it is the source of truth for *why*
  the approach below is structured this way.
- `src/diamonds/` — package code. `data.py` loads and cleans the dataset;
  `features.py` builds the feature-set variants (size: carat or x/y/z;
  grade encoding: ordinal or one-hot) and the log-price target; `split.py`
  holds the shared carat-band-stratified train/test split.
- `tests/` — pytest suite, one module per source module.

## Commands

This project uses `uv`. Never call `python`/`pip` directly.

- `uv sync` — install dependencies.
- `uv run pytest` — run the test suite.
- `uv run python <script>` — run a script inside the project environment.
- `uv add <pkg>` / `uv add --dev <pkg>` — add a runtime / dev dependency.

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

## Testing

See "Testing" in `planning/PLAN.md` for the full rationale. In short:

- Every build step ships with tests; run `uv run pytest` before committing.
- Tests read the real `diamonds.csv` — it is small and fixed, so don't mock
  it.
- Test the data contract (cleaning rules, grade ordering, carat banding) and
  the EDA findings the model must reproduce, above all that cut/color/
  clarity coefficients come out in the *right* direction once carat is
  controlled for.
- Assert loose metric floors, not exact values, so the suite catches a
  broken pipeline without breaking on a hyperparameter or version change.

## Working conventions

- Source layout: `src/diamonds/` package, `tests/` mirroring it one module
  per source module, dependencies in `pyproject.toml` managed with `uv`.
  Keep to this structure.
- Keep `planning/PLAN.md` and this file in sync: if the modeling approach
  changes materially during implementation, update both — including
  marking build steps done as they land.
- Don't modify `planning/report.html` or `diamonds.csv`; they are fixed
  inputs (the EDA report and raw dataset, respectively).
