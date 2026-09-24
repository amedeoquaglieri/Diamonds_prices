"""Regenerate planning/results.html from the current codebase.

Run with: uv run python planning/build_results.py

Recomputes every number and chart in the write-up from scratch (the model
comparison, the final model's importances and partial dependence, the
carat-bucketed check, and the OLS coefficients) rather than reading any
cached values, so the report never drifts from what the package actually
does.
"""

import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from diamonds import bucketed, data, features, final, interpret, metrics, models, split
from diamonds.data import CARAT_BAND_LABELS, CLARITY_ORDER, COLOR_ORDER, CUT_ORDER

OUT_PATH = Path(__file__).resolve().parent / "results.html"

SEQ_STEPS = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]
SERIES_1 = "#2a78d6"
MODEL_COLORS = {"ols": "#2a78d6", "ridge": "#eb6834", "lasso": "#1baf7a", "gbm": "#eda100"}
MODEL_LABELS = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso", "gbm": "LightGBM"}
FIT_FUNCS = {"ols": models.fit_linear, "ridge": models.fit_ridge, "lasso": models.fit_lasso, "gbm": models.fit_gbm}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def money(v):
    return f"${v:,.0f}"


def seq_colors_for(order):
    return [SEQ_STEPS[i] for i in range(len(order))]


TIP = "data-tip"


def bar_chart(cats, vals, *, fmt, colors, width=560, height=300, y_max=None, y_ticks=4, tip_fn=None, y_prefix=""):
    pad_l, pad_r, pad_t, pad_b = 18, 18, 30, 52
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    vmax = y_max if y_max is not None else max(vals) * 1.18
    n = len(cats)
    gap = plot_w / n * 0.28
    bw = plot_w / n - gap
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" aria-label="bar chart">']
    for i in range(y_ticks + 1):
        gy = pad_t + plot_h * (1 - i / y_ticks)
        gval = vmax * i / y_ticks
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" class="gridline"/>')
        parts.append(
            f'<text x="{pad_l - 6}" y="{gy + 4:.1f}" class="axis-label" text-anchor="end">{y_prefix}{gval:,.0f}</text>'
        )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h:.1f}" x2="{width - pad_r}" y2="{pad_t + plot_h:.1f}" class="baseline"/>'
    )
    for i, (cat, val) in enumerate(zip(cats, vals)):
        x = pad_l + i * (plot_w / n) + gap / 2
        bh = plot_h * (val / vmax) if vmax else 0
        y = pad_t + plot_h - bh
        color = colors[i] if isinstance(colors, list) else colors
        tip = esc(tip_fn(cat, val)) if tip_fn else f"{cat}: {val:,.0f}"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{color}" class="mark-bar" {TIP}="{tip}"/>'
        )
        parts.append(f'<text x="{x + bw / 2:.1f}" y="{y - 8:.1f}" class="bar-value" text-anchor="middle">{esc(fmt(val))}</text>')
        parts.append(
            f'<text x="{x + bw / 2:.1f}" y="{pad_t + plot_h + 20:.1f}" class="axis-label" text-anchor="middle">{esc(str(cat))}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def bar_chart_with_reference(cats, vals, reference, *, fmt, color=SERIES_1, width=560, height=300, y_ticks=4, ref_label=""):
    """Bar chart with a dashed horizontal reference line (the pooled/confounded value)."""
    pad_l, pad_r, pad_t, pad_b = 18, 18, 30, 52
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    vmax = max(max(vals), reference, 0) * 1.25
    vmin = min(min(vals), reference, 0) * 1.25
    span = vmax - vmin
    n = len(cats)
    gap = plot_w / n * 0.28
    bw = plot_w / n - gap

    def y_of(v):
        return pad_t + plot_h * (1 - (v - vmin) / span)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" aria-label="bar chart with reference line">']
    for i in range(y_ticks + 1):
        gval = vmin + span * i / y_ticks
        gy = y_of(gval)
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" class="gridline"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{gy + 4:.1f}" class="axis-label" text-anchor="end">{gval:.2f}</text>')
    zero_y = y_of(0)
    parts.append(f'<line x1="{pad_l}" y1="{zero_y:.1f}" x2="{width - pad_r}" y2="{zero_y:.1f}" class="baseline"/>')
    for i, (cat, val) in enumerate(zip(cats, vals)):
        x = pad_l + i * (plot_w / n) + gap / 2
        y_top = y_of(max(val, 0))
        y_bot = y_of(min(val, 0))
        bh = max(y_bot - y_top, 1)
        tip = esc(f"{cat}: {fmt(val)} (within this carat band)")
        parts.append(
            f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{color}" class="mark-bar" {TIP}="{tip}"/>'
        )
        label_y = y_top - 8 if val >= 0 else y_bot + 16
        parts.append(f'<text x="{x + bw / 2:.1f}" y="{label_y:.1f}" class="bar-value" text-anchor="middle">{esc(fmt(val))}</text>')
        parts.append(
            f'<text x="{x + bw / 2:.1f}" y="{pad_t + plot_h + 20:.1f}" class="axis-label" text-anchor="middle">{esc(str(cat))}</text>'
        )
    ref_y = y_of(reference)
    ref_tip = esc(f"{ref_label}: {fmt(reference)}")
    parts.append(
        f'<line x1="{pad_l}" y1="{ref_y:.1f}" x2="{width - pad_r}" y2="{ref_y:.1f}" stroke-dasharray="4 3" class="reference-line" {TIP}="{ref_tip}"/>'
    )
    parts.append(f'<text x="{width - pad_r}" y="{ref_y - 5:.1f}" class="reference-label" text-anchor="end">{esc(ref_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def model_comparison_chart(combos, *, width=680, height=560):
    pad_l, pad_r, pad_t, pad_b = 18, 18, 16, 130
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    vmax = max(r["rmse"] for r in combos) * 1.1
    n = len(combos)
    gap = plot_w / n * 0.22
    bw = plot_w / n - gap
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" aria-label="RMSE by model and feature variant">']
    for frac in (0.25, 0.5, 0.75, 1.0):
        gy = pad_t + plot_h * (1 - frac)
        gval = vmax * frac
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" class="gridline"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{gy + 4:.1f}" class="axis-label" text-anchor="end">${gval:,.0f}</text>')
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h:.1f}" x2="{width - pad_r}" y2="{pad_t + plot_h:.1f}" class="baseline"/>'
    )
    for i, r in enumerate(combos):
        x = pad_l + i * (plot_w / n) + gap / 2
        bh = plot_h * (r["rmse"] / vmax)
        y = pad_t + plot_h - bh
        color = MODEL_COLORS[r["model"]]
        is_final = r["model"] == "gbm" and r["size"] == "carat" and r["encoding"] == "ordinal"
        cls = "mark-bar mark-bar-final" if is_final else "mark-bar"
        tip = esc(
            f"{MODEL_LABELS[r['model']]} · {r['size']} · {r['encoding']}: RMSE {money(r['rmse'])}, R² {r['r2_log']:.3f}"
        )
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{color}" class="{cls}" {TIP}="{tip}"/>'
        )
        if is_final:
            parts.append(f'<text x="{x + bw / 2:.1f}" y="{y - 8:.1f}" class="bar-value" text-anchor="middle">selected</text>')
        size_label = "carat" if r["size"] == "carat" else "x/y/z"
        encoding_label = "1-hot" if r["encoding"] == "one-hot" else "ordinal"
        label = f"{size_label} {encoding_label}"
        parts.append(
            f'<text x="{x + bw / 2:.1f}" y="{pad_t + plot_h + 16:.1f}" class="axis-label-rot" '
            f'text-anchor="end" transform="rotate(-55 {x + bw / 2:.1f} {pad_t + plot_h + 16:.1f})">{label}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def legend(items):
    return "".join(
        f'<span class="legend-item"><span class="swatch" style="background:{color}"></span>{esc(label)}</span>'
        for label, color in items
    )


def main():
    df = features.add_log_columns(data.load())
    train, test = split.split(df)

    combos = []
    for name, fit in FIT_FUNCS.items():
        for size in ["carat", "dimensions"]:
            for encoding in ["ordinal", "one-hot"]:
                fitted = fit(train, size=size, encoding=encoding)
                scores = metrics.score(features.target(test), fitted.predict(test))
                combos.append({"model": name, "size": size, "encoding": encoding, **scores})
    combos.sort(key=lambda r: r["rmse"])
    best_linear_rmse = min(r["rmse"] for r in combos if r["model"] != "gbm")

    final_model = final.fit_final(train)
    final_scores = metrics.score(features.target(test), final_model.predict(test))
    final_by_band = metrics.score_by_band(test["carat_band"], features.target(test), final_model.predict(test))
    final_residual_corr = metrics.residual_carat_correlation(
        test["carat"], features.target(test), final_model.predict(test)
    )
    final_importances = final_model.importances().sort_values(ascending=False)

    pd_results = {grade: interpret.partial_dependence(final_model, train, grade) for grade in features.GRADES}

    prepared = bucketed.add_price_per_carat(data.load())
    bucket_results = {}
    for grade in ["cut", "color", "clarity"]:
        by_band = bucketed.rank_correlation_by_band(prepared, grade)
        pooled = bucketed.unbucketed_rank_correlation(prepared, grade)
        bucket_results[grade] = {"by_band": by_band, "pooled": pooled}

    ols = models.fit_linear(train).coefficients()

    model_chart = model_comparison_chart(combos)
    model_legend = legend([(MODEL_LABELS[m], MODEL_COLORS[m]) for m in ["ols", "ridge", "lasso", "gbm"]])

    importance_chart = bar_chart(
        list(final_importances.index),
        list(final_importances.values),
        fmt=lambda v: f"{v:.0f}",
        colors=SERIES_1,
        tip_fn=lambda cat, val: f"{cat}: {val:.0f} splits",
    )

    pd_charts = {}
    for grade, order in [("cut", CUT_ORDER), ("color", COLOR_ORDER), ("clarity", CLARITY_ORDER)]:
        series = pd_results[grade].reindex(order)
        pd_charts[grade] = bar_chart(
            order,
            list(series.values),
            fmt=money,
            colors=seq_colors_for(order),
            tip_fn=lambda cat, val: f"{cat}: {money(val)} predicted (carat and other grades held fixed)",
        )

    bucket_charts = {}
    for grade in ["cut", "color", "clarity"]:
        by_band = bucket_results[grade]["by_band"].reindex(CARAT_BAND_LABELS)
        pooled = bucket_results[grade]["pooled"]
        bucket_charts[grade] = bar_chart_with_reference(
            CARAT_BAND_LABELS,
            list(by_band.values),
            pooled,
            fmt=lambda v: f"{v:.2f}",
            ref_label="pooled (confounded)",
        )

    band_chart = bar_chart(
        CARAT_BAND_LABELS,
        list(final_by_band["rmse"]),
        fmt=money,
        colors=SERIES_1,
        tip_fn=lambda cat, val: f"{cat} ct: RMSE {money(val)}",
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Diamond price model &mdash; results</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --page-plane:     #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --muted:          #898781;
    --gridline:       #e1e0d9;
    --baseline:       #c3c2b7;
    --border:         rgba(11,11,11,0.10);
    --series-1:       #2a78d6;
    --success-text:   #006300;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) {{
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --page-plane:     #0d0d0d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --muted:          #898781;
      --gridline:       #2c2c2a;
      --baseline:       #383835;
      --border:         rgba(255,255,255,0.10);
      --series-1:       #3987e5;
      --success-text:   #0ca30c;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page-plane:     #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --muted:          #898781;
    --gridline:       #2c2c2a;
    --baseline:       #383835;
    --border:         rgba(255,255,255,0.10);
    --series-1:       #3987e5;
    --success-text:   #0ca30c;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--page-plane);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: 32px 16px 64px; }}
  header.hero h1 {{ font-size: 28px; margin: 0 0 8px; letter-spacing: -0.01em; }}
  header.hero p.lede {{ color: var(--text-secondary); font-size: 15px; line-height: 1.55; max-width: 66ch; margin: 0 0 24px; }}

  .stat-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 8px; }}
  .stat-tile {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
  .stat-tile .label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
  .stat-tile .value {{ font-size: 20px; font-weight: 600; margin-top: 4px; }}
  .stat-tile .value.small {{ font-size: 15px; }}

  section {{ margin-top: 44px; }}
  h2 {{ font-size: 19px; margin: 0 0 6px; }}
  p.section-note {{ color: var(--text-secondary); font-size: 14px; line-height: 1.6; max-width: 70ch; margin: 0 0 18px; }}

  .card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }}
  .card-title {{ font-size: 13px; color: var(--text-secondary); margin: 0 0 8px; font-weight: 600; }}

  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}

  .chart-svg {{ width: 100%; height: auto; display: block; overflow: visible; }}
  .gridline {{ stroke: var(--gridline); stroke-width: 1; }}
  .baseline {{ stroke: var(--baseline); stroke-width: 1; }}
  .reference-line {{ stroke: var(--muted); stroke-width: 1.5; cursor: pointer; }}
  .reference-label {{ font-size: 10px; fill: var(--muted); font-style: italic; }}
  .axis-label {{ font-size: 10.5px; fill: var(--muted); }}
  .axis-label-rot {{ font-size: 9.5px; fill: var(--muted); }}
  .bar-value {{ font-size: 11px; fill: var(--text-primary); font-weight: 600; }}
  .mark-bar, .mark-cell {{ cursor: pointer; transition: opacity .12s ease; }}
  .mark-bar:hover {{ opacity: 0.82; }}
  .mark-bar-final {{ stroke: var(--text-primary); stroke-width: 1.5; }}

  .legend {{ display: flex; flex-wrap: wrap; gap: 10px 14px; margin-top: 10px; font-size: 12px; color: var(--text-secondary); }}
  .legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
  .swatch {{ width: 11px; height: 11px; border-radius: 3px; display: inline-block; }}

  ul.findings {{ margin: 0; padding-left: 20px; font-size: 14px; line-height: 1.7; color: var(--text-primary); }}
  ul.findings li {{ margin-bottom: 6px; }}
  ul.findings b {{ font-weight: 600; }}

  table.coef-table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
  table.coef-table th, table.coef-table td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
  table.coef-table th {{ color: var(--muted); font-weight: 600; font-size: 11.5px; text-transform: uppercase; letter-spacing: .03em; }}
  table.coef-table td.num {{ font-variant-numeric: tabular-nums; }}
  .positive {{ color: var(--success-text); font-weight: 600; }}

  footer {{ margin-top: 52px; padding-top: 18px; border-top: 1px solid var(--border); color: var(--muted); font-size: 12.5px; }}

  #tooltip {{
    position: fixed; pointer-events: none; z-index: 50;
    background: var(--text-primary); color: var(--surface-1);
    font-size: 12px; padding: 5px 9px; border-radius: 6px;
    opacity: 0; transform: translate(-50%, -100%); transition: opacity .08s ease;
    white-space: nowrap; font-weight: 500;
  }}

  @media (max-width: 640px) {{
    .stat-row {{ grid-template-columns: repeat(2, 1fr); }}
    .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="wrap">

<header class="hero">
  <h1>Diamond price model &mdash; results</h1>
  <p class="lede">Follows on from <a href="report.html">the EDA</a> and <a href="PLAN.md">the plan</a>: sixteen
  model &times; feature-variant combinations were compared, and this is the final pick, its accuracy, and the
  checks that confirm it actually captures the 4Cs' effect rather than an artifact of carat.</p>

  <div class="stat-row">
    <div class="stat-tile"><div class="label">Final model</div><div class="value small">LightGBM, carat + ordinal</div></div>
    <div class="stat-tile"><div class="label">Test RMSE</div><div class="value">{money(final_scores['rmse'])}</div></div>
    <div class="stat-tile"><div class="label">Test R&sup2; (log)</div><div class="value">{final_scores['r2_log']:.3f}</div></div>
    <div class="stat-tile"><div class="label">Residual&ndash;carat corr.</div><div class="value">{final_residual_corr:.2f}</div></div>
  </div>
</header>

<section id="comparison">
  <h2>Choosing the final model</h2>
  <p class="section-note">Every model type (OLS, ridge, lasso, LightGBM) fit on every feature-set variant
  (carat or x/y/z; ordinal or one-hot grades), sorted by test RMSE. LightGBM wins outright regardless of
  variant &mdash; the gap to the best linear model (OLS, one-hot) is larger than the gap between any two
  linear models. The selected combination (carat + ordinal) is outlined.</p>
  <div class="card">
    {model_chart}
    <div class="legend">{model_legend}</div>
  </div>
</section>

<section id="importance">
  <h2>What the final model actually uses</h2>
  <p class="section-note">Feature importance (how often LightGBM splits on each feature). Carat dominates, as
  every step of this project found; clarity and color follow, matching their strength in the partial-dependence
  and carat-bucketed checks below. Table and depth &mdash; the cut-proportion measurements &mdash; contribute least,
  echoing the EDA's weak direct correlation for both.</p>
  <div class="card">
    {importance_chart}
  </div>
</section>

<section id="partial-dependence">
  <h2>The 4Cs' effect, with carat held fixed</h2>
  <p class="section-note">Partial dependence: predicted price as each grade sweeps worst&rarr;best, everything else
  (including carat) held at its training value. This is the tree-model equivalent of a linear coefficient &mdash;
  and like the OLS coefficients below, it comes out monotonically increasing for all three grades once carat's
  effect is isolated.</p>
  <div class="grid-3">
    <div class="card"><div class="card-title">Cut</div>{pd_charts['cut']}</div>
    <div class="card"><div class="card-title">Color</div>{pd_charts['color']}</div>
    <div class="card"><div class="card-title">Clarity</div>{pd_charts['clarity']}</div>
  </div>
</section>

<section id="bucketed">
  <h2>The confound, confirmed without a model</h2>
  <p class="section-note">Spearman correlation between grade and price-per-carat, computed separately within each
  carat band (bars) versus pooled across all carat sizes (dashed line). Pooled, the relationship is confounded to
  near zero &mdash; the EDA's "better quality looks cheaper" finding. Within any single band it's positive
  throughout, confirming the model's direction by a method that doesn't fit anything.</p>
  <div class="grid-3">
    <div class="card"><div class="card-title">Cut</div>{bucket_charts['cut']}</div>
    <div class="card"><div class="card-title">Color</div>{bucket_charts['color']}</div>
    <div class="card"><div class="card-title">Clarity</div>{bucket_charts['clarity']}</div>
  </div>
</section>

<section id="by-band">
  <h2>Error by carat size</h2>
  <p class="section-note">RMSE for the final model, computed separately within each carat band. Error grows with
  band size because a fixed percentage error is a bigger dollar error on a bigger stone, not because the model
  fits large stones worse in relative terms &mdash; and the residual shows essentially no leftover correlation with
  carat ({final_residual_corr:.2f} Spearman), confirming the log-carat term has captured the size effect.</p>
  <div class="card">
    {band_chart}
  </div>
</section>

<section id="coefficients">
  <h2>Reading the price mechanism directly</h2>
  <p class="section-note">LightGBM is the predictor; OLS's coefficients (same carat + ordinal variant) remain the
  readable explanation of the price mechanism &mdash; the reason this project didn't ensemble the two (see
  planning/PLAN.md, step 10).</p>
  <div class="card">
    <table class="coef-table">
      <thead><tr><th>Term</th><th>Coefficient</th><th>Reading</th></tr></thead>
      <tbody>
        <tr><td>log(carat)</td><td class="num positive">{ols['log_carat']:.3f}</td><td>Price &prop; carat<sup>{ols['log_carat']:.2f}</sup> once grades are held fixed &mdash; steeper than the EDA's unconditional 1.68</td></tr>
        <tr><td>clarity (per grade step)</td><td class="num positive">+{ols['clarity']:.3f}</td><td>&times;{math.exp(ols['clarity']):.3f} price per step up in clarity, holding carat and the other grades fixed</td></tr>
        <tr><td>color (per grade step)</td><td class="num positive">+{ols['color']:.3f}</td><td>&times;{math.exp(ols['color']):.3f} price per step up in color</td></tr>
        <tr><td>cut (per grade step)</td><td class="num positive">+{ols['cut']:.3f}</td><td>&times;{math.exp(ols['cut']):.3f} price per step up in cut &mdash; smallest and noisiest of the three (see the carat-bucketed check above)</td></tr>
        <tr><td>table</td><td class="num">{ols['table']:.4f}</td><td>Essentially zero &mdash; matches the EDA's weak table&ndash;price correlation</td></tr>
        <tr><td>depth</td><td class="num">{ols['depth']:.4f}</td><td>Essentially zero &mdash; same for depth</td></tr>
      </tbody>
    </table>
  </div>
</section>

<section id="takeaways">
  <h2>Takeaways</h2>
  <div class="card">
    <ul class="findings">
      <li><b>LightGBM on carat + ordinal grades is the final model</b>: RMSE {money(final_scores['rmse'])}, R&sup2; {final_scores['r2_log']:.3f}, beating the best linear model (OLS, one-hot, RMSE {money(best_linear_rmse)}) by {money(best_linear_rmse - final_scores['rmse'])}.</li>
      <li><b>Every check agrees on direction</b> once carat is controlled for: the OLS coefficients, the tree's partial dependence, and the model-free carat-bucketed correlation all show cut/color/clarity raising price &mdash; three independent methods converging on the same answer that raw averages get backwards.</li>
      <li><b>Cut is the weakest and noisiest of the 4Cs</b> throughout: the smallest OLS coefficient, the flattest partial-dependence curve, and the only grade with non-monotonic cell means in the carat-bucketed check.</li>
      <li><b>The model has no leftover size bias</b>: residual&ndash;carat correlation is {final_residual_corr:.2f}, confirming log(carat) fully captures the size effect this whole project is about.</li>
      <li><b>One-hot beats ordinal for every linear model</b>, a finding from the final comparison that didn't change the model choice (LightGBM wins either way) but explains why the readable OLS baseline (ordinal, for its single coefficient per grade) isn't also the most accurate linear variant.</li>
    </ul>
  </div>
</section>

<footer>
  Source: <code>src/diamonds/</code> (this repository). Reproduce with <code>uv run pytest</code> (every claim
  above is a test) or regenerate this page with <code>uv run python planning/build_results.py</code>. See
  <a href="PLAN.md">planning/PLAN.md</a> for the full build history and every intermediate result.
</footer>

</div>
<div id="tooltip"></div>
<script>
(function() {{
  var tip = document.getElementById('tooltip');
  document.addEventListener('mouseover', function(e) {{
    var t = e.target.closest('[data-tip]');
    if (!t) return;
    tip.textContent = t.getAttribute('data-tip');
    tip.style.opacity = '1';
  }});
  document.addEventListener('mousemove', function(e) {{
    var t = e.target.closest('[data-tip]');
    if (!t) {{ tip.style.opacity = '0'; return; }}
    tip.style.left = e.clientX + 'px';
    tip.style.top = (e.clientY - 10) + 'px';
  }});
  document.addEventListener('mouseout', function(e) {{
    if (e.target.closest('[data-tip]')) tip.style.opacity = '0';
  }});
}})();
</script>
</body>
</html>
"""

    OUT_PATH.write_text(html)
    print(f"wrote {OUT_PATH} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
