"""Checks on the feature-set variants the modelling steps choose between."""

import numpy as np
import pytest

from diamonds import data, features
from diamonds.data import GRADE_ORDERS


@pytest.fixture(scope="module")
def prepared():
    return features.add_log_columns(data.load())


def test_log_columns_have_no_missing_values(prepared):
    assert not prepared[[features.LOG_CARAT, features.LOG_PRICE]].isna().any().any()


def test_log_price_inverts_back_to_price(prepared):
    """to_price is what converts model error back into dollars."""
    recovered = features.to_price(prepared[features.LOG_PRICE])
    assert np.allclose(recovered, prepared["price"])


def test_log_carat_inverts_back_to_carat(prepared):
    assert np.allclose(np.exp(prepared[features.LOG_CARAT]), prepared["carat"])


def test_ordinal_codes_follow_worst_to_best_order(prepared):
    codes = features.ordinal_codes(prepared)
    for grade, order in GRADE_ORDERS.items():
        mapping = (
            prepared[[grade]]
            .assign(code=codes[grade])
            .drop_duplicates()
            .set_index(grade)["code"]
        )
        assert [mapping[level] for level in order] == list(range(len(order)))


def test_one_hot_keeps_every_level_but_the_worst(prepared):
    """The worst grade is the dropped reference level, so coefficients read
    as a premium over it and the matrix stays full rank with an intercept."""
    dummies = features.one_hot_codes(prepared)
    for grade, order in GRADE_ORDERS.items():
        columns = [c for c in dummies.columns if c.startswith(f"{grade}_")]
        assert columns == [f"{grade}_{level}" for level in order[1:]]


@pytest.mark.parametrize("size", list(features.SIZE_FEATURES))
@pytest.mark.parametrize("encoding", features.ENCODINGS)
def test_variant_never_mixes_carat_with_dimensions(prepared, size, encoding):
    """Carat and x/y/z correlate at r>=0.97; a variant uses one or the other."""
    columns = features.build_features(prepared, size=size, encoding=encoding).columns
    has_carat = features.LOG_CARAT in columns
    has_dimensions = any(dimension in columns for dimension in features.DIMENSIONS)
    assert has_carat != has_dimensions


@pytest.mark.parametrize("size", list(features.SIZE_FEATURES))
@pytest.mark.parametrize("encoding", features.ENCODINGS)
def test_variant_is_numeric_and_complete(prepared, size, encoding):
    """Every variant must be model-ready: all numeric, no missing values."""
    X = features.build_features(prepared, size=size, encoding=encoding)
    assert len(X) == len(prepared)
    assert not X.isna().any().any()
    assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes)


@pytest.mark.parametrize("size", list(features.SIZE_FEATURES))
def test_one_hot_is_wider_than_ordinal(prepared, size):
    ordinal = features.build_features(prepared, size=size, encoding="ordinal")
    one_hot = features.build_features(prepared, size=size, encoding="one-hot")
    assert one_hot.shape[1] > ordinal.shape[1]


def test_target_is_log_price(prepared):
    assert features.target(prepared).equals(prepared[features.LOG_PRICE])
