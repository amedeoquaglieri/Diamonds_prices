"""Score predictions of log price.

Models are fitted on log price, but error is only meaningful to a reader in
dollars, so RMSE and MAE are reported after converting back. R2 stays on the
log scale, where the model actually fits. score_by_band and
residual_carat_correlation are the two evaluation-plan checks from
planning/PLAN.md: error broken out by carat band, and whether a model's
error still trends with size after fitting.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from diamonds.features import to_price


def score(log_true: pd.Series | np.ndarray, log_pred: np.ndarray) -> dict[str, float]:
    """RMSE and MAE in dollars, R2 on log price."""
    price_true = to_price(log_true)
    price_pred = to_price(log_pred)
    return {
        "rmse": float(root_mean_squared_error(price_true, price_pred)),
        "mae": float(mean_absolute_error(price_true, price_pred)),
        "r2_log": float(r2_score(log_true, log_pred)),
    }


def score_by_band(
    carat_band: pd.Series, log_true: pd.Series | np.ndarray, log_pred: np.ndarray
) -> pd.DataFrame:
    """`score`, computed separately within each carat band.

    A band is a much narrower carat range than the whole dataset, so R2 is
    expected to come out lower here than the headline score: most of what
    R2 measures overall is carat itself, and that barely varies within a
    band. `carat_band.values` (not a plain array) is what keeps the bands
    in smallest-to-largest order rather than alphabetical.
    """
    frame = pd.DataFrame(
        {
            "carat_band": carat_band.values,
            "log_true": np.asarray(log_true),
            "log_pred": np.asarray(log_pred),
        }
    )
    return frame.groupby("carat_band", observed=True).apply(
        lambda g: pd.Series(score(g["log_true"], g["log_pred"]))
    )


def residual_carat_correlation(
    carat: pd.Series | np.ndarray,
    log_true: pd.Series | np.ndarray,
    log_pred: np.ndarray,
) -> float:
    """Spearman correlation between carat and the log-scale residual.

    Should be small: a strong trend would mean error still depends on size
    after fitting, e.g. a carat non-linearity the model hasn't captured.
    """
    residual = np.asarray(log_true) - np.asarray(log_pred)
    return float(spearmanr(carat, residual).statistic)
