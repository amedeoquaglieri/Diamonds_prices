"""The final model this project selects, and why.

See "Compare and select a final model" in planning/PLAN.md for the full
comparison table. In short: LightGBM on the carat + ordinal-encoding
variant has the best accuracy of every model and every variant tried by a
wide margin, needs no functional-form choice the linear models are
sensitive to (log-carat, ordinal vs one-hot), and its partial dependence
(interpret.py) already gives this project the interpretability it needs -
a linear coefficient beyond that would be convenient but was never a
requirement of the plan.
"""

import pandas as pd

from diamonds.models import Fitted, fit_gbm

SIZE = "carat"
ENCODING = "ordinal"


def fit_final(train: pd.DataFrame) -> Fitted:
    """Fit the model this project recommends for predicting price."""
    return fit_gbm(train, size=SIZE, encoding=ENCODING)
