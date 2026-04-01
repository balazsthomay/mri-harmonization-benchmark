"""Tests for reproducibility analysis metrics."""

import numpy as np
import pandas as pd
import pytest

from mri_harmonization.analysis.icc import compute_icc
from mri_harmonization.analysis.cv import compute_cv
from mri_harmonization.analysis.kruskal import compute_kruskal_wallis
from mri_harmonization.analysis.summary import (
    ReproducibilityResult,
    analyze_reproducibility,
    summarize_conditions,
)


class TestICC:
    def test_no_variance(self) -> None:
        """Identical values = perfectly reproducible."""
        values = pd.Series([10.0] * 20)
        sites = pd.Series(["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5)
        icc = compute_icc(values, sites)
        assert icc == 1.0

    def test_high_site_effect_gives_low_icc(self) -> None:
        """Large site differences should give low reproducibility index."""
        rng = np.random.RandomState(42)
        values = pd.Series(
            np.concatenate([
                rng.normal(10, 0.1, 10),  # Site A
                rng.normal(100, 0.1, 10),  # Site B - very different
                rng.normal(500, 0.1, 10),  # Site C - very different
            ])
        )
        sites = pd.Series(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
        icc = compute_icc(values, sites)
        assert icc < 0.05  # Almost all variance is between sites

    def test_no_site_effect_gives_high_icc(self) -> None:
        """Same distribution across sites = high reproducibility."""
        rng = np.random.RandomState(42)
        n = 50
        values = pd.Series(rng.normal(100, 30, n * 3))
        sites = pd.Series(["A"] * n + ["B"] * n + ["C"] * n)
        icc = compute_icc(values, sites)
        assert icc > 0.9  # Almost no variance due to site


class TestCV:
    def test_zero_variation(self) -> None:
        """All same values should give CV=0."""
        values = pd.Series([5.0, 5.0, 5.0])
        cv = compute_cv(values)
        assert cv == 0.0

    def test_known_cv(self) -> None:
        """Test CV computation against known value."""
        values = pd.Series([10.0, 20.0, 30.0])
        cv = compute_cv(values)
        expected = np.std([10, 20, 30], ddof=1) / np.mean([10, 20, 30])
        assert abs(cv - expected) < 1e-10

    def test_positive_result(self) -> None:
        values = pd.Series([1.0, 2.0, 3.0, 4.0])
        cv = compute_cv(values)
        assert cv > 0


class TestKruskalWallis:
    def test_identical_groups(self) -> None:
        """Same distribution should give high p-value (no significant difference)."""
        rng = np.random.RandomState(123)  # Seed chosen to avoid unlucky draws
        n = 50
        values = pd.Series(rng.normal(100, 10, n * 3))
        sites = pd.Series(["A"] * n + ["B"] * n + ["C"] * n)

        stat, pvalue = compute_kruskal_wallis(values, sites)
        assert pvalue > 0.05  # Not significant

    def test_different_groups(self) -> None:
        """Clearly different distributions should give low p-value."""
        values = pd.Series(
            [1.0] * 10 + [100.0] * 10 + [500.0] * 10
        )
        sites = pd.Series(["A"] * 10 + ["B"] * 10 + ["C"] * 10)

        stat, pvalue = compute_kruskal_wallis(values, sites)
        assert pvalue < 0.05  # Significant


class TestSummary:
    def test_analyze_reproducibility(self) -> None:
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "subject_id": [f"S{i}" for i in range(30)],
            "site": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "feat1": rng.normal(100, 10, 30),
            "feat2": np.concatenate([
                rng.normal(10, 1, 10),
                rng.normal(50, 1, 10),
                rng.normal(90, 1, 10),
            ]),
        })

        results = analyze_reproducibility(df)
        assert len(results) == 2
        assert all(isinstance(r, ReproducibilityResult) for r in results)

    def test_summarize_conditions(self) -> None:
        results = {
            "none": [
                ReproducibilityResult("f1", icc=0.8, cv=0.1, kw_statistic=1.0, kw_pvalue=0.5, is_reproducible=True),
                ReproducibilityResult("f2", icc=0.3, cv=0.5, kw_statistic=10.0, kw_pvalue=0.001, is_reproducible=False),
            ],
            "zscore": [
                ReproducibilityResult("f1", icc=0.9, cv=0.05, kw_statistic=0.5, kw_pvalue=0.8, is_reproducible=True),
                ReproducibilityResult("f2", icc=0.85, cv=0.08, kw_statistic=1.0, kw_pvalue=0.3, is_reproducible=True),
            ],
        }

        summary = summarize_conditions(results)
        assert len(summary) == 2
        assert "pct_reproducible" in summary.columns
        assert "median_icc" in summary.columns

        # zscore should have higher % reproducible
        none_row = summary[summary["condition"] == "none"].iloc[0]
        zscore_row = summary[summary["condition"] == "zscore"].iloc[0]
        assert zscore_row["pct_reproducible"] > none_row["pct_reproducible"]
