"""Tests for IXI demographics parsing."""

from pathlib import Path

import pandas as pd
import pytest

from mri_harmonization.acquisition.demographics import load_demographics, merge_demographics
from mri_harmonization.types import Site, Subject


@pytest.fixture
def fake_demographics_xls(tmp_path: Path) -> Path:
    """Create a fake IXI demographics spreadsheet."""
    df = pd.DataFrame(
        {
            "IXI_ID": [2, 25, 530],
            "SEX_ID (1=m, 2=f)": [1, 2, 1],
            "AGE": [30.5, 45.0, 62.3],
        }
    )
    path = tmp_path / "IXI.xls"
    df.to_excel(path, index=False)
    return path


class TestLoadDemographics:
    def test_load(self, fake_demographics_xls: Path) -> None:
        demographics = load_demographics(fake_demographics_xls)
        assert len(demographics) == 3
        assert demographics["IXI002"]["age"] == 30.5
        assert demographics["IXI002"]["sex"] == "M"
        assert demographics["IXI025"]["sex"] == "F"
        assert demographics["IXI530"]["age"] == 62.3

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_demographics(tmp_path / "nonexistent.xls")


class TestMergeDemographics:
    def test_merge(self, tmp_path: Path) -> None:
        subjects = [
            Subject(id="IXI002", site=Site.GUYS, image_path=tmp_path / "a.nii.gz"),
            Subject(id="IXI025", site=Site.HH, image_path=tmp_path / "b.nii.gz"),
        ]
        demographics = {
            "IXI002": {"age": 30.5, "sex": "M"},
            "IXI025": {"age": 45.0, "sex": "F"},
        }

        merged = merge_demographics(subjects, demographics)
        assert merged[0].age == 30.5
        assert merged[0].sex == "M"
        assert merged[1].age == 45.0

    def test_merge_missing_demographics(self, tmp_path: Path) -> None:
        subjects = [
            Subject(id="IXI999", site=Site.GUYS, image_path=tmp_path / "a.nii.gz"),
        ]
        demographics = {"IXI002": {"age": 30.5, "sex": "M"}}

        merged = merge_demographics(subjects, demographics)
        assert merged[0].age is None
        assert merged[0].sex is None
