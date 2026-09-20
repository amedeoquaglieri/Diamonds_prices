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

## Open questions for you

- Preferred model type to start with — plain interpretable regression, or
  go straight to gradient-boosted trees for accuracy?
- Do you want feature importance / partial-dependence explanations as a
  deliverable, or just a working price predictor?
- Any target for how the model will be used (one-off report vs. something
  that needs to run repeatedly / serve predictions)? This affects whether
  we keep it a notebook/script or wrap it as a small package.
