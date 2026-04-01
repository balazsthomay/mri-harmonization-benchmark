"""Intensity histogram visualization."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_intensity_histograms(
    intensity_data: dict[str, np.ndarray],
    title: str = "Intensity Distributions by Site",
    bins: int = 100,
) -> Figure:
    """Plot overlapping intensity histograms for each site.

    Args:
        intensity_data: Dict mapping site name to intensity values.
        title: Plot title.
        bins: Number of histogram bins.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Guys": "#2196F3", "HH": "#FF9800", "IOP": "#4CAF50"}

    for site_name, values in intensity_data.items():
        color = colors.get(site_name, None)
        ax.hist(values, bins=bins, alpha=0.5, label=site_name, color=color, density=True)

    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig
