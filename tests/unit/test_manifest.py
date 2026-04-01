"""Tests for subject manifest management."""

from pathlib import Path

import pytest

from mri_harmonization.acquisition.manifest import (
    build_manifest_from_directory,
    load_manifest,
    parse_ixi_filename,
    save_manifest,
)
from mri_harmonization.types import Site, Subject


class TestParseIxiFilename:
    def test_valid_filename(self) -> None:
        site, subject_id = parse_ixi_filename("IXI002-Guys-0828-T1.nii.gz")
        assert site == Site.GUYS
        assert subject_id == "IXI002"

    def test_hh_site(self) -> None:
        site, subject_id = parse_ixi_filename("IXI025-HH-1187-T1.nii.gz")
        assert site == Site.HH
        assert subject_id == "IXI025"

    def test_iop_site(self) -> None:
        site, subject_id = parse_ixi_filename("IXI530-IOP-0986-T1.nii.gz")
        assert site == Site.IOP
        assert subject_id == "IXI530"

    def test_invalid_filename(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_ixi_filename("invalid.nii.gz")

    def test_path_object(self) -> None:
        site, subject_id = parse_ixi_filename(Path("some/dir/IXI002-Guys-0828-T1.nii.gz"))
        assert site == Site.GUYS
        assert subject_id == "IXI002"


class TestBuildManifest:
    def test_build_from_directory(self, tmp_path: Path) -> None:
        # Create fake NIfTI files
        for name in [
            "IXI002-Guys-0828-T1.nii.gz",
            "IXI025-HH-1187-T1.nii.gz",
            "IXI530-IOP-0986-T1.nii.gz",
        ]:
            (tmp_path / name).touch()

        subjects = build_manifest_from_directory(tmp_path)
        assert len(subjects) == 3
        sites = {s.site for s in subjects}
        assert sites == {Site.GUYS, Site.HH, Site.IOP}

    def test_ignores_non_nifti_files(self, tmp_path: Path) -> None:
        (tmp_path / "IXI002-Guys-0828-T1.nii.gz").touch()
        (tmp_path / "README.txt").touch()
        (tmp_path / "data.csv").touch()

        subjects = build_manifest_from_directory(tmp_path)
        assert len(subjects) == 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        subjects = build_manifest_from_directory(tmp_path)
        assert len(subjects) == 0


class TestManifestIO:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        subjects = [
            Subject(
                id="IXI002",
                site=Site.GUYS,
                image_path=tmp_path / "IXI002.nii.gz",
                age=30.5,
                sex="M",
            ),
            Subject(
                id="IXI025",
                site=Site.HH,
                image_path=tmp_path / "IXI025.nii.gz",
                mask_path=tmp_path / "IXI025_mask.nii.gz",
                age=45.0,
                sex="F",
            ),
        ]

        csv_path = tmp_path / "manifest.csv"
        save_manifest(subjects, csv_path)
        loaded = load_manifest(csv_path)

        assert len(loaded) == 2
        assert loaded[0].id == "IXI002"
        assert loaded[0].site == Site.GUYS
        assert loaded[0].age == 30.5
        assert loaded[0].sex == "M"
        assert loaded[1].mask_path == tmp_path / "IXI025_mask.nii.gz"

    def test_save_and_load_with_none_fields(self, tmp_path: Path) -> None:
        subjects = [
            Subject(
                id="IXI002",
                site=Site.GUYS,
                image_path=tmp_path / "IXI002.nii.gz",
            ),
        ]

        csv_path = tmp_path / "manifest.csv"
        save_manifest(subjects, csv_path)
        loaded = load_manifest(csv_path)

        assert loaded[0].mask_path is None
        assert loaded[0].age is None
        assert loaded[0].sex is None
