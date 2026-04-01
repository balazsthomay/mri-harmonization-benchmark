"""Tests for IXI dataset download."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mri_harmonization.acquisition.download import (
    extract_subjects_from_tar,
    select_subjects_per_site,
)
from mri_harmonization.types import Site


class TestSelectSubjectsPerSite:
    def test_select_balanced(self) -> None:
        filenames = [
            f"IXI{i:03d}-Guys-0001-T1.nii.gz" for i in range(10)
        ] + [
            f"IXI{i:03d}-HH-0001-T1.nii.gz" for i in range(100, 110)
        ] + [
            f"IXI{i:03d}-IOP-0001-T1.nii.gz" for i in range(200, 210)
        ]

        selected = select_subjects_per_site(filenames, subjects_per_site=5)
        assert len(selected) == 15

        # Check balance across sites
        sites = [name.split("-")[1] for name in selected]
        assert sites.count("Guys") == 5
        assert sites.count("HH") == 5
        assert sites.count("IOP") == 5

    def test_select_fewer_than_available(self) -> None:
        filenames = [f"IXI{i:03d}-Guys-0001-T1.nii.gz" for i in range(3)]
        selected = select_subjects_per_site(filenames, subjects_per_site=10)
        # Should return all 3 since fewer available than requested
        assert len(selected) == 3

    def test_deterministic_selection(self) -> None:
        filenames = [f"IXI{i:03d}-Guys-0001-T1.nii.gz" for i in range(20)]
        selected1 = select_subjects_per_site(filenames, subjects_per_site=5)
        selected2 = select_subjects_per_site(filenames, subjects_per_site=5)
        assert selected1 == selected2


class TestExtractSubjectsFromTar:
    def test_extract_creates_files(self, tmp_path: Path) -> None:
        import io
        import tarfile

        # Create a small tar with fake NIfTI files
        tar_path = tmp_path / "test.tar"
        with tarfile.open(tar_path, "w") as tar:
            for name in ["IXI002-Guys-0828-T1.nii.gz", "IXI025-HH-1187-T1.nii.gz"]:
                info = tarfile.TarInfo(name=name)
                data = b"fake nifti data"
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

        output_dir = tmp_path / "output"
        filenames_to_extract = ["IXI002-Guys-0828-T1.nii.gz", "IXI025-HH-1187-T1.nii.gz"]
        extract_subjects_from_tar(tar_path, output_dir, filenames_to_extract)

        assert (output_dir / "IXI002-Guys-0828-T1.nii.gz").exists()
        assert (output_dir / "IXI025-HH-1187-T1.nii.gz").exists()

    def test_extract_only_selected(self, tmp_path: Path) -> None:
        import io
        import tarfile

        tar_path = tmp_path / "test.tar"
        with tarfile.open(tar_path, "w") as tar:
            for name in [
                "IXI002-Guys-0828-T1.nii.gz",
                "IXI025-HH-1187-T1.nii.gz",
                "IXI530-IOP-0986-T1.nii.gz",
            ]:
                info = tarfile.TarInfo(name=name)
                data = b"fake data"
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

        output_dir = tmp_path / "output"
        extract_subjects_from_tar(
            tar_path, output_dir, ["IXI002-Guys-0828-T1.nii.gz"]
        )

        assert (output_dir / "IXI002-Guys-0828-T1.nii.gz").exists()
        assert not (output_dir / "IXI025-HH-1187-T1.nii.gz").exists()
