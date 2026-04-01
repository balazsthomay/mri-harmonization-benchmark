"""Coefficient of Variation computation."""

import numpy as np
import pandas as pd


def compute_cv(values: pd.Series) -> float:
    """Compute the Coefficient of Variation.

    Args:
        values: Feature values.

    Returns:
        CV = std / mean. Returns 0 if mean is 0 or all values are identical.
    """
    mean = values.mean()
    std = values.std(ddof=1)

    if mean == 0 or std == 0:
        return 0.0

    return float(std / abs(mean))
