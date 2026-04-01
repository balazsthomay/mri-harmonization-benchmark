"""Tests for feature matrix I/O."""

from pathlib import Path

import pandas as pd

from mri_harmonization.features.io import load_feature_matrix, save_feature_matrix


class TestFeatureIO:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "subject_id": ["IXI002", "IXI025", "IXI530"],
                "site": ["Guys", "HH", "IOP"],
                "feature_a": [1.0, 2.0, 3.0],
                "feature_b": [4.0, 5.0, 6.0],
            }
        )

        csv_path = tmp_path / "features.csv"
        save_feature_matrix(df, csv_path)
        loaded = load_feature_matrix(csv_path)

        pd.testing.assert_frame_equal(df, loaded)

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1.0]})
        csv_path = tmp_path / "sub" / "dir" / "features.csv"
        save_feature_matrix(df, csv_path)
        assert csv_path.exists()

    def test_preserves_dtypes(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "subject_id": ["IXI002"],
                "site": ["Guys"],
                "int_feature": [42.0],
                "float_feature": [3.14],
            }
        )

        csv_path = tmp_path / "features.csv"
        save_feature_matrix(df, csv_path)
        loaded = load_feature_matrix(csv_path)

        assert loaded["float_feature"].iloc[0] == 3.14
