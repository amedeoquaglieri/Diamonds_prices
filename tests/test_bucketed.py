"""Checks on the carat-bucketed comparison, the plan's non-model sanity check."""

import numpy as np
import pytest

from diamonds import bucketed, data
from diamonds.data import CARAT_BAND_LABELS, GRADE_ORDERS


@pytest.fixture(scope="module")
def prepared():
    return bucketed.add_price_per_carat(data.load())


def test_price_per_carat_is_price_over_carat(prepared):
    assert np.allclose(
        prepared[bucketed.PRICE_PER_CARAT], prepared["price"] / prepared["carat"]
    )


@pytest.mark.parametrize("grade", ["cut", "color", "clarity"])
def test_bucketed_means_columns_are_worst_to_best(prepared, grade):
    assert list(bucketed.bucketed_means(prepared, grade).columns) == GRADE_ORDERS[grade]


def test_bucketed_means_rows_are_smallest_to_largest_band(prepared):
    assert list(bucketed.bucketed_means(prepared, "clarity").index) == CARAT_BAND_LABELS


@pytest.mark.parametrize("grade", ["cut", "color", "clarity"])
def test_every_band_shows_a_positive_grade_effect(prepared, grade):
    """Within any single carat band, better grade should mean a higher
    price-per-carat - the confound check restated as a correlation rather
    than a strict ordering, so a little small-sample noise in one grade
    level doesn't fail it (cut is the noisiest of the three - see
    planning/PLAN.md)."""
    by_band = bucketed.rank_correlation_by_band(prepared, grade)
    assert (by_band > 0).all()


@pytest.mark.parametrize("grade", ["cut", "color", "clarity"])
def test_bucketing_reveals_the_confound(prepared, grade):
    """The whole point of this step: pooled across all carat sizes, the
    grade-price relationship is confounded to near zero (or worse); within
    any single carat band, it's real and positive."""
    by_band = bucketed.rank_correlation_by_band(prepared, grade)
    pooled = bucketed.unbucketed_rank_correlation(prepared, grade)
    assert pooled < by_band.min()
