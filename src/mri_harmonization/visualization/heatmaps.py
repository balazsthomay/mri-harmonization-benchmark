"""ICC heatmap visualization."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def plot_icc_heatmap(
    icc_data: pd.DataFrame,
    title: str = "ICC by Feature and Harmonization Condition",
) -> Figure:
    """Plot ICC heatmap: features x conditions.

    Args:
        icc_data: DataFrame with features as rows, conditions as columns.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(max(10, len(icc_data.columns) * 2), max(6, len(icc_data) * 0.3)))

    sns.heatmap(
        icc_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title(title)
    ax.set_ylabel("Feature")
    ax.set_xlabel("Condition")
    fig.tight_layout()

    return fig
