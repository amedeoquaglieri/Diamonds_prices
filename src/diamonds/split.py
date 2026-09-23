"""Split the dataset into train and test sets.

Stratifying by carat band matters here: carat drives price, and the bands
are very unevenly sized, so an unstratified draw can leave the large-stone
bands thin enough to distort per-band error.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY_COLUMN = "carat_band"


def split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and test, stratified by carat band."""
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[STRATIFY_COLUMN],
    )
