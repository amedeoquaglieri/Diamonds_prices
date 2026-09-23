"""Fit models of log price.

A fitted model is kept together with the feature variant it was trained on,
so scoring it on new rows cannot silently use a different feature set.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from diamonds.features import build_features, target


@dataclass(frozen=True)
class Fitted:
    """An estimator plus the feature variant it was fitted on."""

    estimator: object
    size: str
    encoding: str

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict log price for rows built into this model's feature variant."""
        return self.estimator.predict(build_features(df, self.size, self.encoding))

    def coefficients(self) -> pd.Series:
        """Fitted coefficients, named by feature."""
        return pd.Series(
            self.estimator.coef_, index=self.estimator.feature_names_in_
        ).sort_values(ascending=False)


def fit_linear(
    train: pd.DataFrame, size: str = "carat", encoding: str = "ordinal"
) -> Fitted:
    """Fit the OLS baseline: log price on log carat, proportions and grades.

    Deliberately unscaled, so the log-carat coefficient reads directly as the
    price-to-carat elasticity the EDA estimated at about 1.68.
    """
    X = build_features(train, size, encoding)
    return Fitted(LinearRegression().fit(X, target(train)), size, encoding)
