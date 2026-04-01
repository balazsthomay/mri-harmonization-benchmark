"""Tests for feature-level ComBat harmonization."""

import numpy as np
import pandas as pd
import pytest

from mri_harmonization.harmonization.feature_level import ComBatHarmonizer


@pytest.fixture
def feature_df_with_site_effect() -> pd.DataFrame:
    """Create feature DataFrame with clear site effects."""
    rng = np.random.RandomState(42)
    n_per_site = 20

    data = {
        "subject_id": [f"S{i:03d}" for i in range(n_per_site * 3)],
        "site": ["Guys"] * n_per_site + ["HH"] * n_per_site + ["IOP"] * n_per_site,
    }

    # Features with site-dependent shifts
    for feat_name in ["feature_a", "feature_b", "feature_c"]:
        guys_vals = rng.normal(10, 2, n_per_site)
        hh_vals = rng.normal(20, 2, n_per_site)  # Different mean = site effect
        iop_vals = rng.normal(30, 2, n_per_site)
        data[feat_name] = np.concatenate([guys_vals, hh_vals, iop_vals])

    return pd.DataFrame(data)


class TestComBatHarmonizer:
    def test_output_shape_matches_input(
        self, feature_df_with_site_effect: pd.DataFrame
    ) -> None:
        harmonizer = ComBatHarmonizer()
        result = harmonizer.harmonize(feature_df_with_site_effect)

        assert result.shape == feature_df_with_site_effect.shape

    def test_preserves_metadata_columns(
        self, feature_df_with_site_effect: pd.DataFrame
    ) -> None:
        harmonizer = ComBatHarmonizer()
        result = harmonizer.harmonize(feature_df_with_site_effect)

        assert "subject_id" in result.columns
        assert "site" in result.columns

    def test_reduces_site_effect(
        self, feature_df_with_site_effect: pd.DataFrame
    ) -> None:
        harmonizer = ComBatHarmonizer()
        result = harmonizer.harmonize(feature_df_with_site_effect)

        # After ComBat, site means should be closer
        for feat in ["feature_a", "feature_b", "feature_c"]:
            before_means = feature_df_with_site_effect.groupby("site")[feat].mean()
            after_means = result.groupby("site")[feat].mean()

            before_range = before_means.max() - before_means.min()
            after_range = after_means.max() - after_means.min()

            assert after_range < before_range, (
                f"{feat}: site effect not reduced. Before: {before_range:.2f}, After: {after_range:.2f}"
            )

    def test_values_are_finite(
        self, feature_df_with_site_effect: pd.DataFrame
    ) -> None:
        harmonizer = ComBatHarmonizer()
        result = harmonizer.harmonize(feature_df_with_site_effect)

        feature_cols = [c for c in result.columns if c not in ("subject_id", "site")]
        for col in feature_cols:
            assert result[col].notna().all(), f"NaN values in {col}"
            assert np.all(np.isfinite(result[col])), f"Non-finite values in {col}"

    def test_with_biological_covariates(
        self, feature_df_with_site_effect: pd.DataFrame
    ) -> None:
        rng = np.random.RandomState(99)
        df = feature_df_with_site_effect.copy()
        df["age"] = rng.normal(50, 15, len(df))
        df["sex"] = rng.choice(["M", "F"], len(df))

        harmonizer = ComBatHarmonizer(
            continuous_cols=["age"],
            categorical_cols=["sex"],
        )
        result = harmonizer.harmonize(df)

        assert result.shape[0] == df.shape[0]
