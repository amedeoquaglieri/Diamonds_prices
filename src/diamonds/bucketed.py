"""Carat-bucketed price comparison: the plan's non-model sanity check.

The regression and tree models control for carat by fitting it as a
feature; this checks the same claim about cut/color/clarity without fitting
anything, by looking directly at price within narrow carat bands. It
generalizes the EDA's single ~1 ct slice across the full carat range, as an
independent check that the models found a real effect rather than an
artifact of their functional form.
"""

import pandas as pd
from scipy.stats import spearmanr

from diamonds.data import CARAT_BAND_LABELS, GRADE_ORDERS
from diamonds.features import ordinal_codes

PRICE_PER_CARAT = "price_per_carat"


def add_price_per_carat(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[PRICE_PER_CARAT] = df["price"] / df["carat"]
    return df


def bucketed_means(
    df: pd.DataFrame, grade: str, metric: str = PRICE_PER_CARAT
) -> pd.DataFrame:
    """Mean `metric` by grade level, within each carat band.

    Rows run smallest to largest carat band, columns worst to best grade:
    read down a row for the confound this project is about (raw, pooled
    averages run backwards), and across a row for the corrected effect
    (within a band, quality should raise the average).
    """
    return (
        df.groupby(["carat_band", grade], observed=True)[metric]
        .mean()
        .unstack(grade)
        .reindex(columns=GRADE_ORDERS[grade])
        .reindex(CARAT_BAND_LABELS)
    )


def rank_correlation_by_band(
    df: pd.DataFrame, grade: str, metric: str = PRICE_PER_CARAT
) -> pd.Series:
    """Spearman correlation between grade and `metric`, within each carat band.

    Restates the confound check as one number per band rather than a
    monotonic run across grade levels: robust to the small-sample noise
    that can break strict ordering in a cell or two (cut, especially, is
    noisy - see planning/PLAN.md), while still requiring the relationship
    to run the right way overall.
    """
    codes = ordinal_codes(df)[grade]
    correlations = {}
    for band in CARAT_BAND_LABELS:
        mask = df["carat_band"] == band
        correlations[band] = spearmanr(codes[mask], df.loc[mask, metric]).statistic
    return pd.Series(correlations)


def unbucketed_rank_correlation(
    df: pd.DataFrame, grade: str, metric: str = PRICE_PER_CARAT
) -> float:
    """The same correlation pooled across all carat bands: the confound as it
    looks before controlling for carat, for direct contrast with
    rank_correlation_by_band."""
    return spearmanr(ordinal_codes(df)[grade], df[metric]).statistic
