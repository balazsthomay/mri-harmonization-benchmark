"""Feature matrix CSV I/O."""

from pathlib import Path

import pandas as pd


def save_feature_matrix(df: pd.DataFrame, path: Path) -> None:
    """Save a feature matrix DataFrame to CSV.

    Args:
        df: Feature matrix with subject_id, site, and feature columns.
        path: Output CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_feature_matrix(path: Path) -> pd.DataFrame:
    """Load a feature matrix from CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with feature data.
    """
    return pd.read_csv(path)
