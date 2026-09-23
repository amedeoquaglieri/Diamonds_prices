"""Checks on the train/test split every modelling step shares."""

import pytest

from diamonds import data, split
from diamonds.data import CARAT_BAND_LABELS


@pytest.fixture(scope="module")
def frame():
    return data.load()


@pytest.fixture(scope="module")
def parts(frame):
    return split.split(frame)


def test_split_sizes(frame, parts):
    train, test = parts
    assert len(train) + len(test) == len(frame)
    assert len(test) == pytest.approx(len(frame) * split.TEST_SIZE, rel=0.01)


def test_no_row_is_in_both_sides(parts):
    train, test = parts
    assert train.index.intersection(test.index).empty


def test_carat_band_proportions_are_preserved(frame, parts):
    """Stratification is the point of this module, so pin it."""
    train, test = parts
    overall = frame["carat_band"].value_counts(normalize=True)
    for part in (train, test):
        proportions = part["carat_band"].value_counts(normalize=True)
        assert (proportions - overall).abs().max() < 0.01


def test_every_band_appears_on_both_sides(parts):
    """Step 9 reports metrics per band, so no band may be empty."""
    train, test = parts
    for part in (train, test):
        counts = part["carat_band"].value_counts()
        assert set(counts[counts > 0].index) == set(CARAT_BAND_LABELS)


def test_split_is_deterministic(frame):
    """Models are compared across steps, so the split must not move."""
    first_train, first_test = split.split(frame)
    second_train, second_test = split.split(frame)
    assert first_train.index.equals(second_train.index)
    assert first_test.index.equals(second_test.index)


def test_split_keeps_all_columns(frame, parts):
    train, test = parts
    assert list(train.columns) == list(frame.columns)
    assert list(test.columns) == list(frame.columns)
