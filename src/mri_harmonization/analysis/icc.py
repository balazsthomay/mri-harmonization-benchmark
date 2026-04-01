"""Site-effect metrics for cross-sectional multi-site data.

Standard ICC requires the same subjects measured by each rater. In our case,
different subjects are scanned at different sites, so we use a variance-ratio
approach: the proportion of total variance NOT explained by site.

High value (close to 1) = features are reproducible across sites.
Low value (close to 0) = features are dominated by site effects.
"""

import numpy as np
import pandas as pd


def compute_icc(
    values: pd.Series,
    groups: pd.Series,
) -> float:
    """Compute a reproducibility index based on one-way ANOVA decomposition.

    Returns 1 - eta_squared, where eta_squared is the proportion of variance
    explained by site. This gives:
    - 1.0 = no site effect (perfect reproducibility)
    - 0.0 = all variance is between sites (no reproducibility)

    Args:
        values: Feature values for all subjects.
        groups: Group (site) labels for each subject.

    Returns:
        Reproducibility index in [0, 1].
    """
    values = values.values.astype(float)
    groups = groups.values

    grand_mean = np.mean(values)
    ss_total = np.sum((values - grand_mean) ** 2)

    if ss_total == 0:
        return 1.0  # No variance at all = perfectly reproducible

    # Between-group sum of squares
    ss_between = 0.0
    for g in np.unique(groups):
        mask = groups == g
        group_mean = np.mean(values[mask])
        ss_between += np.sum(mask) * (group_mean - grand_mean) ** 2

    # eta_squared = proportion of variance explained by site
    eta_squared = ss_between / ss_total

    # Reproducibility = 1 - eta_squared
    return float(max(0.0, 1.0 - eta_squared))
