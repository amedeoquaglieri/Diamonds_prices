"""Interpret a fitted model's effect on price, beyond a single accuracy number.

Partial dependence is the tree-model analogue of a linear coefficient: it
sweeps one grade from worst to best, holding everything else fixed, and
reports the predicted price at each level. It only makes sense for the
ordinal encoding, where a grade is a single numeric column running
worst-to-best; one-hot spreads each grade over several dummy columns with
no single column to sweep.
"""

import pandas as pd
from sklearn.inspection import partial_dependence as sk_partial_dependence

from diamonds.data import GRADE_ORDERS
from diamonds.features import build_features, to_price
from diamonds.models import Fitted


def partial_dependence(fitted: Fitted, train: pd.DataFrame, grade: str) -> pd.Series:
    """Average predicted price as `grade` sweeps worst to best.

    This is the confound check from the OLS baseline, restated for a model
    with no coefficients to read directly: price should rise from the worst
    grade to the best once carat's effect is isolated.
    """
    if fitted.encoding != "ordinal":
        raise ValueError("partial dependence needs the ordinal encoding")
    X = build_features(train, fitted.size, fitted.encoding)
    result = sk_partial_dependence(fitted.estimator, X, features=[grade], kind="average")
    return pd.Series(to_price(result["average"][0]), index=GRADE_ORDERS[grade])
