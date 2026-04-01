"""Feature distribution visualization."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def plot_feature_distributions(
    df: pd.DataFrame,
    features: list[str],
    title: str = "Feature Distributions by Site",
) -> Figure:
    """Plot violin plots of selected features across sites.

    Args:
        df: Feature DataFrame with site column.
        features: Feature column names to plot.
        title: Overall title.

    Returns:
        matplotlib Figure.
    """
    n_features = len(features)
    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 6))

    if n_features == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        sns.violinplot(data=df, x="site", y=feat, ax=ax, inner="box")
        ax.set_title(feat)
        ax.set_xlabel("Site")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    return fig
