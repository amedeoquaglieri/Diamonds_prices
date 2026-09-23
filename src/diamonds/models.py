"""Fit models of log price.

A fitted model is kept together with the feature variant it was trained on,
so scoring it on new rows cannot silently use a different feature set.
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diamonds.features import build_features, target

RIDGE_ALPHAS = np.logspace(-3, 3, 13)
CV_FOLDS = 5
LASSO_MAX_ITER = 50_000
GBM_PARAMS = {"random_state": 0, "verbosity": -1}


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
        """Fitted coefficients, in the original feature units.

        Regularized models are fitted on standardized features, so their raw
        coefficients are per standard deviation. Dividing by the scale puts
        every model's coefficients on the same footing as the OLS baseline.
        """
        final = self.estimator[-1] if isinstance(self.estimator, Pipeline) else self.estimator
        values = final.coef_
        if isinstance(self.estimator, Pipeline):
            values = values / self.estimator.named_steps["scaler"].scale_
        return pd.Series(values, index=self.estimator.feature_names_in_).sort_values(
            ascending=False
        )

    @property
    def alpha(self) -> float:
        """The regularization strength chosen by cross-validation."""
        return self.estimator[-1].alpha_

    def importances(self) -> pd.Series:
        """Feature importances, sorted descending.

        Unlike coefficients(), these carry no sign or unit: a tree's
        importance is how often a feature is split on, not the direction or
        size of its effect on price. Use interpret.partial_dependence for
        that.
        """
        final = self.estimator[-1] if isinstance(self.estimator, Pipeline) else self.estimator
        return pd.Series(
            final.feature_importances_, index=self.estimator.feature_names_in_
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


def _fit_scaled(train: pd.DataFrame, size: str, encoding: str, regressor) -> Fitted:
    """Fit a regularized regressor on standardized features.

    Regularization penalizes coefficient size, so it is only meaningful once
    the features share a scale.
    """
    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", regressor)])
    X = build_features(train, size, encoding)
    return Fitted(pipeline.fit(X, target(train)), size, encoding)


def fit_ridge(
    train: pd.DataFrame, size: str = "carat", encoding: str = "ordinal"
) -> Fitted:
    """Fit ridge regression, choosing alpha by cross-validation.

    Ridge shrinks correlated features together rather than picking between
    them, which is what the near-collinear x/y/z variant needs.
    """
    return _fit_scaled(train, size, encoding, RidgeCV(alphas=RIDGE_ALPHAS))


def fit_lasso(
    train: pd.DataFrame, size: str = "carat", encoding: str = "ordinal"
) -> Fitted:
    """Fit lasso regression, choosing alpha by cross-validation.

    Lasso drives redundant coefficients to zero, so it shows which of the
    correlated size features it considers dispensable.
    """
    return _fit_scaled(
        train,
        size,
        encoding,
        LassoCV(cv=CV_FOLDS, random_state=0, max_iter=LASSO_MAX_ITER),
    )


def fit_gbm(
    train: pd.DataFrame, size: str = "carat", encoding: str = "ordinal", **params
) -> Fitted:
    """Fit a gradient-boosted tree ensemble on log price.

    Trees pick up the carat non-linearity and any grade interactions on
    their own, without the log-carat transform or the ordinal-vs-one-hot
    choice mattering the way it does to a linear model; this is here as a
    stronger predictive baseline, traded off against the coefficients'
    interpretability.
    """
    X = build_features(train, size, encoding)
    model = lgb.LGBMRegressor(**{**GBM_PARAMS, **params})
    model.fit(X, target(train))
    return Fitted(model, size, encoding)
