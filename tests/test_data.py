"""Checks that loading and cleaning match the EDA's data-quality findings."""

import pytest

from diamonds import data

RAW_ROWS = 53940
CLEAN_ROWS = 53766


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()


@pytest.fixture(scope="module")
def clean():
    return data.load()


def test_raw_shape(raw):
    assert len(raw) == RAW_ROWS
    assert "Unnamed: 0" not in raw.columns


def test_clean_row_count(clean):
    assert len(clean) == CLEAN_ROWS


def test_no_zero_dimensions(clean):
    assert (clean[data.DIMENSIONS] > 0).all().all()


def test_no_oversized_dimensions(clean):
    assert clean["y"].max() < data.MAX_DIMENSION_MM
    assert clean["z"].max() < data.MAX_DIMENSION_MM


def test_no_extreme_proportions(clean):
    assert clean["depth"].max() < data.MAX_DEPTH_PCT
    assert clean["table"].max() < data.MAX_TABLE_PCT


def test_no_duplicates(clean):
    assert not clean.duplicated().any()


def test_grades_are_ordered_categoricals(clean):
    for column, order in data.GRADE_ORDERS.items():
        assert list(clean[column].cat.categories) == order
        assert clean[column].cat.ordered


def test_every_row_gets_a_carat_band(clean):
    assert not clean["carat_band"].isna().any()
    assert list(clean["carat_band"].cat.categories) == data.CARAT_BAND_LABELS


def test_carat_bands_are_left_closed(clean):
    """The 0.2 minimum and round-number carats fall in the band above."""
    smallest = clean.loc[clean["carat"].idxmin()]
    assert smallest["carat_band"] == "0.2-0.4"
    assert (clean.loc[clean["carat"] == 1.0, "carat_band"] == "1.0-1.5").all()


def test_every_band_is_populated(clean):
    """Stratifying the train/test split needs a usable count in each band."""
    assert clean["carat_band"].value_counts().min() > 1000
