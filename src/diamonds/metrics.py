"""Score predictions of log price.

Models are fitted on log price, but error is only meaningful to a reader in
dollars, so RMSE and MAE are reported after converting back. R2 stays on the
log scale, where the model actually fits.
"""

import numpy as np
import pandas as pd
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
