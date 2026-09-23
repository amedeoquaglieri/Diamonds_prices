# Diamond price model — plan

This plan follows on from `planning/report.html`, the exploratory analysis of
`diamonds.csv` (Kaggle diamonds dataset, 53,940 rows). It is a plan only —
no model code is implemented yet. See the report for full detail; the
relevant findings are summarized below as the reasons behind each decision.

## Why carat needs to be controlled for

- Carat correlates with price at r=0.92 (raw) and r=0.97 in log-log space,
  following roughly `price ∝ carat^1.68`. It dominates every other feature.
- Raw (unconditional) average price by cut/color/clarity moves in the
  *wrong* direction — e.g. IF clarity looks cheaper than I1 — because the
  best-clarity/best-color stones in this dataset skew smaller. Carat is a
  confounder between the 4Cs and price.
- Restricting to a narrow carat band (0.99–1.01 ct) removes the confound and
  recovers the expected monotonic price increase with clarity, color, and
  cut.
- Conclusion: any model of price on cut/color/clarity/dimensions must
  explicitly account for carat's effect, or its coefficients on the other
  4Cs will be biased (and can flip sign).

## Data preparation

1. Drop or repair known-bad rows (from the EDA data-quality pass):
   - 20 rows with a zero x/y/z dimension.
   - 6 rows with an implausible y or z (>10mm) — decimal-point typos.
   - 289 rows involved in exact duplicates — keep one copy of each.
   - A handful of extreme depth (≥75%) / table (≥80%) rows.
   - Net: ~53,766 usable rows.
2. Feature set:
   - `carat` (primary driver) — model in log space (`log(carat)`) given the
     power-law relationship found in the EDA.
   - `cut`, `color`, `clarity` — encode as ordered/ordinal (they have a
     natural worst→best order) rather than one-hot, so the model can learn
     a monotonic-ish quality effect; also try one-hot as a comparison since
     the true price step between grades isn't necessarily linear.
   - `depth`, `table` — cut-proportion features; weak direct correlation
     with price in the EDA, but may matter as interaction terms with cut
     grade rather than as standalone predictors.
   - `x`, `y`, `z` — nearly redundant with carat (r≥0.97 with each other and
     with carat); use at most one of {carat, x, y, z} as the size feature
     to avoid multicollinearity, or reduce them (e.g. estimated volume) as
     a single derived feature and compare against carat alone.
3. Target: model `log(price)` rather than `price` directly — matches the
   power-law relationship and stabilizes the right-skewed price
   distribution (skew 1.62 on raw price).

## Modeling approach — how carat gets "controlled for"

Two complementary ways to phrase "control for carat," both worth doing:

1. **Regression with carat as a covariate.** Fit
   `log(price) ~ log(carat) + cut + color + clarity + depth + table`
   (start with linear regression / OLS on log-price, then compare against
   a tree-based model). The coefficients on cut/color/clarity, holding
   carat fixed, give the actual quality premium — this directly answers
   "how much is better clarity/color/cut worth at a given size," which is
   what the raw averages got backwards on.
2. **Carat-bucketed comparison / price-per-carat.** As a simpler,
   more interpretable companion to the regression: bucket carat into bands
   (e.g. 0.2–0.4, 0.4–0.7, 0.7–1.0, 1.0–1.5, 1.5+) and compute mean price
   and price-per-carat by cut/color/clarity *within each band*. This is
   the same idea as the ~1 ct slice in the EDA, generalized across the
   whole carat range, and serves as a sanity check / baseline against the
   regression.

Candidate model types, in order of complexity:
- Linear regression on log(price) with the features above (baseline,
  most interpretable coefficients).
- Regularized regression (Ridge/Lasso) if x/y/z are kept alongside carat,
  to handle multicollinearity.
- Gradient-boosted trees (e.g. XGBoost/LightGBM) as a stronger predictive
  baseline — trees handle the carat non-linearity and cut/color/clarity
  interactions natively, at the cost of interpretability compared to the
  linear coefficients.

## Evaluation plan

- Train/test split (e.g. 80/20), stratified by carat band so both splits
  cover the full size range.
- Metrics: RMSE and MAE on price (converted back from log-price), plus
  R² on log(price). Report metrics overall and broken out by carat band,
  since error will likely scale with price magnitude.
- Sanity checks tied back to the EDA:
  - Fitted coefficients (or partial-dependence plots, for the tree model)
    on cut/color/clarity should be monotonic in the expected direction
    once carat is in the model — this is the key check that the confound
    has actually been controlled for.
  - Residuals vs. carat: should not show a strong remaining trend: if
    they do, the log-carat term needs a different functional form (e.g. a
    spline or polynomial term).

## Build steps

Concrete, ordered steps to go from this plan to a working model. Each step
should be a separate commit, and each step ships with tests covering the
invariant it establishes (see "Testing" below); keep this list and
`CLAUDE.md` updated if the approach changes materially along the way.

1. **Scaffold the project** — done
   - Add a dependency manifest (`requirements.txt` or `pyproject.toml`):
     pandas, numpy, scikit-learn, and a gradient-boosting library
     (xgboost or lightgbm). Add `pytest` as a dev dependency.
   - Create a `src/` package (e.g. `src/diamonds/`) rather than loose
     top-level scripts, per the working conventions in `CLAUDE.md`.

2. **Data loading & cleaning** (`src/diamonds/data.py`) — done
   - Load `diamonds.csv`.
   - Apply the cleaning steps from "Data preparation" above: drop rows with
     a zero x/y/z dimension, drop the implausible-y/z decimal-point-typo
     rows, deduplicate exact duplicates, drop extreme depth/table rows.
   - Return the cleaned ~53,766-row DataFrame, plus a carat-band column
     (0.2–0.4, 0.4–0.7, 0.7–1.0, 1.0–1.5, 1.5+) used later for
     stratification and the bucketed comparison.
   - Tests: cleaned row count, no bad rows surviving any of the four
     cleaning rules, grade categoricals in worst→best order, every row
     banded, every band populated enough to stratify on.

3. **Feature engineering** (`src/diamonds/features.py`) — done
   - Derive `log(carat)` and `log(price)`.
   - Ordinal-encode `cut`, `color`, `clarity` (worst→best); also build a
     one-hot-encoded variant of each for comparison against the ordinal
     encoding.
   - Assemble feature-set variants to compare: (a) `log(carat)` + `depth` +
     `table` + 4Cs, and (b) `x`/`y`/`z` + `depth` + `table` + 4Cs — never
     carat and x/y/z together, to avoid multicollinearity.
   - Tests: log transforms invert back to the original values, ordinal codes
     follow the worst→best order, one-hot columns match the grade levels,
     and no feature-set variant contains both carat and x/y/z.

4. **Train/test split** (`src/diamonds/split.py`) — done
   - 80/20 split, stratified by carat band.
   - Tests: split sizes, no row appearing in both sides, carat-band
     proportions preserved across train and test.

5. **Baseline: linear regression on log(price)** — done
   (`src/diamonds/models.py`, `src/diamonds/metrics.py`)
   - Fit OLS on `log(price) ~ log(carat) + cut + color + clarity + depth +
     table` (ordinal encoding first).
   - Record coefficients and metrics (see Evaluation plan).
   - Tests: the fitted `log(carat)` coefficient is positive and in a wide
     band around the EDA estimate, and the cut/color/clarity coefficients
     come out positive (better grade, higher price) now that carat is in
     the model — the confound check, as a test rather than a manual look.

   Result: the confound reverses as predicted. Grade coefficients are all
   positive (clarity +0.122, color +0.078, cut +0.030 per grade step), and
   under one-hot encoding — which tells the model nothing about grade order
   — every grade's coefficients still come out monotonically worst→best.
   Test R² 0.979 on log price, RMSE $923, MAE $461.

   One correction to this plan's expectation: the fitted `log(carat)`
   coefficient is **1.88, not 1.68**. The 1.68 figure was the
   *unconditional* log-log slope, which is biased downward by the same
   confound seen from the other side — higher-grade stones skew smaller, so
   leaving grades out understates how steeply price climbs with carat.
   Controlling for them raises the elasticity. The test therefore asserts a
   wide band (1.5-2.2) rather than pinning 1.68.

6. **Regularized regression comparison** — done (`src/diamonds/models.py`)
   - Fit Ridge/Lasso on the same target, including the x/y/z feature-set
     variant, to check how it handles the multicollinearity that plain OLS
     can't.
   - Both fit on standardized features (regularization is scale-sensitive);
     `Fitted.coefficients()` divides the scaling back out so every model's
     coefficients stay comparable with the OLS baseline.

   Result: **regularization turns out not to be needed here, and this step's
   premise was wrong.** Ridge reproduces the OLS fit to four decimals on
   every variant (test RMSE $922.78 vs $922.78; predictions agree within
   0.01 log units), and cross-validation picks small alphas (0.1-1.0).

   The premise was that near-collinear x/y/z (r>=0.97) would destabilize OLS.
   It doesn't, because n=43,012 against at most 22 features leaves plenty of
   information to separate them. Refitting on 10 bootstrap resamples gives
   OLS coefficient standard deviations of x 0.048 / y 0.061 / z 0.103 —
   no worse than ridge's 0.052 / 0.062 / 0.133. Collinearity inflates
   variance, but not enough to matter at this sample size.

   Lasso is slightly *worse* than OLS (RMSE $823.84 vs $773.85 on the best
   variant): with no excess variance to trade away, shrinkage only costs
   signal. It does zero out `table` on the carat variant, agreeing with the
   EDA that table carries almost no price information.

   Implication for step 10: prefer the simpler OLS. Regularization earns
   nothing on this dataset.

7. **Gradient-boosted trees** — done
   (`src/diamonds/models.py`, `src/diamonds/interpret.py`)
   - Fit LightGBM on `log(price)` with both feature-set variants.
   - Compute feature importances and partial dependence for
     cut/color/clarity — trees have no coefficients to read, so
     `interpret.partial_dependence` sweeps each grade worst-to-best,
     holding everything else fixed, and reports the predicted price at each
     level.
   - Tests: predictions have the right shape, importances cover every
     input feature and rank `log_carat` first, the GBM beats the OLS
     baseline, every feature-set variant clears the accuracy floor, and
     partial dependence is defined only for the ordinal encoding and comes
     out monotonically increasing worst-to-best for all three grades — the
     confound check restated for a model with no coefficients.

   Result: LightGBM clearly out-predicts OLS — RMSE $529 vs $923, R²(log)
   0.991 vs 0.979, on the carat+ordinal variant (its best; dimensions+
   ordinal is a close second at $537). `log_carat` is the single most
   important feature by a wide margin, matching every earlier step.
   Partial dependence confirms the confound reversal holds without needing
   a coefficient to read: predicted price rises strictly worst-to-best for
   cut, color and clarity alike, holding carat and the other grades fixed.

8. **Carat-bucketed comparison (sanity check)**
   - Compute mean price and price-per-carat by cut/color/clarity within each
     carat band (generalizing the ~1 ct slice from the EDA across the full
     range).
   - Compare its direction/shape against the regression and tree
     coefficients — they should agree once carat is controlled for.

9. **Evaluation**
   - Compute RMSE, MAE (back-transformed to price) and R² (log-price) for
     every model above, overall and broken out by carat band.
   - Check the sanity checks from "Evaluation plan": monotonic
     cut/color/clarity effect, no strong residual-vs-carat trend.
   - Tests: metric functions return known values on a small hand-built
     example (especially the log→price back-transformation, which is easy
     to get subtly wrong), and each model clears a loose R² floor so a
     future refactor that silently breaks training fails the suite.

10. **Compare and select a final model**
    - Summarize metrics and interpretability tradeoffs across the linear,
      regularized, and tree models; pick (or ensemble) a final model with a
      stated rationale.

11. **Write up results**
    - Produce a results report (mirroring `planning/report.html`'s format)
      covering final metrics, coefficients/importances, and the sanity
      checks.
    - Update `planning/PLAN.md` and `CLAUDE.md` if anything changed from
      this plan during implementation.

## Testing

Tests live in `tests/`, one module per source module (`tests/test_data.py`
covers `src/diamonds/data.py`). Run them with `uv run pytest`, and run the
suite before committing each build step.

What is worth testing here, and what isn't:

- **Do test the data contract.** The cleaning rules, grade ordering and
  carat banding are assumptions every later step silently depends on; a
  test is what stops a subtle change to one of them from quietly biasing a
  model later on.
- **Do test the findings the model must reproduce.** The EDA's central
  claim is that quality grades look *backwards* until carat is controlled
  for. Asserting the fitted coefficients come out in the right direction
  turns that from a thing we looked at once into a thing that stays true.
- **Do test transformations that are easy to get quietly wrong**, above all
  the log(price)→price back-transformation used for RMSE and MAE.
- **Don't pin exact metric values.** Assert loose floors (e.g. R² above a
  clearly-passing threshold) so the suite catches a broken pipeline without
  failing every time a hyperparameter or library version shifts a digit.
- **Don't mock the dataset.** It is a fixed 3MB file that loads in under a
  second; tests read the real thing, so they check the real data.

## Open questions for you

- Preferred model type to start with — plain interpretable regression, or
  go straight to gradient-boosted trees for accuracy?
- Do you want feature importance / partial-dependence explanations as a
  deliverable, or just a working price predictor?
- Any target for how the model will be used (one-off report vs. something
  that needs to run repeatedly / serve predictions)? This affects whether
  we keep it a notebook/script or wrap it as a small package.
