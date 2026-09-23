"""Build model features from the cleaned dataset.

Two choices are deliberately left open so later steps can compare them:
the size feature (carat or the x/y/z dimensions, never both, since they
correlate at r>=0.97) and the grade encoding (ordinal or one-hot).
"""

import numpy as np
import pandas as pd

from diamonds.data import GRADE_ORDERS

GRADES = list(GRADE_ORDERS)
PROPORTIONS = ["depth", "table"]
DIMENSIONS = ["x", "y", "z"]

LOG_CARAT = "log_carat"
LOG_PRICE = "log_price"

SIZE_FEATURES = {"carat": [LOG_CARAT], "dimensions": DIMENSIONS}
ENCODINGS = ("ordinal", "one-hot")


def add_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log_carat and log_price, the power-law scale the EDA found."""
    df = df.copy()
    df[LOG_CARAT] = np.log(df["carat"])
    df[LOG_PRICE] = np.log(df["price"])
    return df


def ordinal_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Encode each grade as its worst-to-best rank (0 is the worst grade)."""
    return pd.DataFrame(
        {grade: df[grade].cat.codes for grade in GRADES}, index=df.index
    ).astype(float)


def one_hot_codes(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the grades, dropping the worst level of each.

    The dropped level is the reference category, so a coefficient reads as
    the premium over the worst grade and the design matrix stays full rank
    alongside an intercept.
    """
    return pd.get_dummies(df[GRADES], drop_first=True).astype(float)


def build_features(
    df: pd.DataFrame, size: str = "carat", encoding: str = "ordinal"
) -> pd.DataFrame:
    """Assemble one feature-set variant: size + proportions + encoded grades."""
    size_columns = SIZE_FEATURES[size]
    grades = ordinal_codes(df) if encoding == "ordinal" else one_hot_codes(df)
    return pd.concat([df[size_columns + PROPORTIONS], grades], axis=1)


def target(df: pd.DataFrame) -> pd.Series:
    """The modelled target: log price."""
    return df[LOG_PRICE]


def to_price(log_price: pd.Series | np.ndarray) -> np.ndarray:
    """Invert the log target back to price, for reporting error in dollars."""
    return np.exp(log_price)
