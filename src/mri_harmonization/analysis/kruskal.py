"""Kruskal-Wallis test for site effects."""

import pandas as pd
from scipy import stats


def compute_kruskal_wallis(
    values: pd.Series, groups: pd.Series
) -> tuple[float, float]:
    """Compute Kruskal-Wallis H-test for site differences.

    Args:
        values: Feature values for all subjects.
        groups: Group (site) labels.

    Returns:
        Tuple of (test statistic, p-value).
    """
    group_data = [
        values[groups == g].values for g in groups.unique()
    ]

    # Need at least 2 groups with data
    group_data = [g for g in group_data if len(g) > 0]
    if len(group_data) < 2:
        return 0.0, 1.0

    stat, pvalue = stats.kruskal(*group_data)
    return float(stat), float(pvalue)
