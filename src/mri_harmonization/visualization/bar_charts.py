"""Reproducibility summary bar charts."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_reproducibility_summary(
    summary_df: pd.DataFrame,
    title: str = "Percentage of Reproducible Features (ICC > 0.75)",
) -> Figure:
    """Plot bar chart of reproducible feature percentage per condition.

    Args:
        summary_df: DataFrame with condition and pct_reproducible columns.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = []
    for cond in summary_df["condition"]:
        if "combat" in cond.lower():
            colors.append("#2196F3")
        else:
            colors.append("#FF9800")

    bars = ax.bar(
        summary_df["condition"],
        summary_df["pct_reproducible"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_ylabel("% Reproducible Features (ICC > 0.75)")
    ax.set_xlabel("Harmonization Condition")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% threshold")

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    return fig
