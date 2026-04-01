"""Tests for visualization modules."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mri_harmonization.visualization.histograms import plot_intensity_histograms
from mri_harmonization.visualization.distributions import plot_feature_distributions
from mri_harmonization.visualization.heatmaps import plot_icc_heatmap
from mri_harmonization.visualization.bar_charts import plot_reproducibility_summary


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "subject_id": [f"S{i}" for i in range(30)],
        "site": ["Guys"] * 10 + ["HH"] * 10 + ["IOP"] * 10,
        "firstorder_Mean": rng.normal(100, 10, 30),
        "firstorder_Variance": rng.normal(50, 5, 30),
        "glcm_Correlation": rng.normal(0.5, 0.1, 30),
    })


@pytest.fixture
def sample_summary_df() -> pd.DataFrame:
    return pd.DataFrame({
        "condition": ["none", "zscore", "whitestripe", "nyul", "none+combat", "zscore+combat"],
        "pct_reproducible": [30.0, 35.0, 40.0, 45.0, 70.0, 75.0],
        "median_icc": [0.4, 0.45, 0.5, 0.55, 0.7, 0.75],
    })


class TestIntensityHistograms:
    def test_creates_figure(self, tmp_path: Path) -> None:
        intensity_data = {
            "Guys": np.random.RandomState(42).normal(100, 20, 1000),
            "HH": np.random.RandomState(43).normal(120, 25, 1000),
            "IOP": np.random.RandomState(44).normal(90, 15, 1000),
        }

        fig = plot_intensity_histograms(intensity_data, title="Before Harmonization")
        assert fig is not None
        plt.close(fig)

    def test_save_to_file(self, tmp_path: Path) -> None:
        intensity_data = {
            "Guys": np.random.normal(100, 20, 100),
            "HH": np.random.normal(120, 25, 100),
        }
        output = tmp_path / "hist.png"
        fig = plot_intensity_histograms(intensity_data)
        fig.savefig(output)
        plt.close(fig)
        assert output.exists()


class TestFeatureDistributions:
    def test_creates_figure(self, sample_feature_df: pd.DataFrame) -> None:
        fig = plot_feature_distributions(
            sample_feature_df,
            features=["firstorder_Mean", "glcm_Correlation"],
        )
        assert fig is not None
        plt.close(fig)


class TestICCHeatmap:
    def test_creates_figure(self) -> None:
        icc_data = pd.DataFrame({
            "none": [0.3, 0.5, 0.8],
            "zscore": [0.4, 0.6, 0.85],
            "combat": [0.7, 0.8, 0.9],
        }, index=["feat1", "feat2", "feat3"])

        fig = plot_icc_heatmap(icc_data)
        assert fig is not None
        plt.close(fig)


class TestReproducibilitySummary:
    def test_creates_figure(self, sample_summary_df: pd.DataFrame) -> None:
        fig = plot_reproducibility_summary(sample_summary_df)
        assert fig is not None
        plt.close(fig)
