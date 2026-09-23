"""Load and clean the Kaggle diamonds dataset.

Cleaning follows the data-quality findings in planning/report.html: rows with
impossible dimensions, decimal-point typos, exact duplicates and extreme cut
proportions are dropped, leaving ~53,766 of 53,940 rows.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "diamonds.csv"

CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

GRADE_ORDERS = {"cut": CUT_ORDER, "color": COLOR_ORDER, "clarity": CLARITY_ORDER}

CARAT_BAND_EDGES = [0.2, 0.4, 0.7, 1.0, 1.5, np.inf]
CARAT_BAND_LABELS = ["0.2-0.4", "0.4-0.7", "0.7-1.0", "1.0-1.5", "1.5+"]

MAX_DIMENSION_MM = 10.0
MAX_DEPTH_PCT = 75.0
MAX_TABLE_PCT = 80.0

DIMENSIONS = ["x", "y", "z"]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV, dropping its unnamed row-number column."""
    return pd.read_csv(path, index_col=0)


def add_grade_order(df: pd.DataFrame) -> pd.DataFrame:
    """Type cut, color and clarity as ordered categoricals (worst to best)."""
    df = df.copy()
    for column, order in GRADE_ORDERS.items():
        df[column] = pd.Categorical(df[column], categories=order, ordered=True)
    return df


def add_carat_band(df: pd.DataFrame) -> pd.DataFrame:
    """Add a carat_band column used for stratification and bucketed comparison."""
    df = df.copy()
    df["carat_band"] = pd.cut(
        df["carat"],
        bins=CARAT_BAND_EDGES,
        labels=CARAT_BAND_LABELS,
        right=False,
        ordered=True,
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the rows the EDA flagged as unusable.

    Removes zero dimensions (impossible measurements), y/z above 10mm
    (decimal-point typos), exact duplicates, and extreme depth/table.
    """
    usable = (
        (df[DIMENSIONS] > 0).all(axis=1)
        & (df["y"] < MAX_DIMENSION_MM)
        & (df["z"] < MAX_DIMENSION_MM)
        & (df["depth"] < MAX_DEPTH_PCT)
        & (df["table"] < MAX_TABLE_PCT)
    )
    return df[usable].drop_duplicates()


def load(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset cleaned, grade-ordered and carat-banded."""
    return add_carat_band(add_grade_order(clean(load_raw(path))))
