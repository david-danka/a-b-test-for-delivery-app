"""Loading and preparing the A/A and A/B test datasets."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CONTROL = "Control"
EXPERIMENTAL = "Experimental"

OUTLIER_COLUMNS = ["order_value", "session_duration"]
OUTLIER_PERCENTILE = 99


def load_aa_test(path: Path = DATA_DIR / "aa_test.csv") -> pd.DataFrame:
    """Load the A/A test: two samples drawn before the new interface shipped."""
    return pd.read_csv(path)


def load_ab_test(path: Path = DATA_DIR / "ab_test.csv") -> pd.DataFrame:
    """Load the A/B test: one row per session, tagged with its group."""
    return pd.read_csv(path)


def remove_outliers(
    df: pd.DataFrame,
    columns: list[str],
    percentile: float,
) -> pd.DataFrame:
    """Drop rows that reach the given percentile in any of the columns."""
    thresholds = df[columns].quantile(percentile / 100)
    return df[(df[columns] < thresholds).all(axis=1)].copy()


def split_groups(df: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    """Return the control and experimental values of a column."""
    groups = df.groupby("group")[column]
    return groups.get_group(CONTROL), groups.get_group(EXPERIMENTAL)


def add_log_order_value(df: pd.DataFrame) -> pd.DataFrame:
    """Add the natural log of the order value to reduce its skew."""
    return df.assign(log_order_value=np.log(df["order_value"]))
