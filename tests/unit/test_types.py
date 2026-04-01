"""Tests for domain types."""

from pathlib import Path

from mri_harmonization.types import (
    HarmonizationMethod,
    Site,
    Subject,
)


class TestSite:
    def test_site_values(self) -> None:
        assert Site.GUYS.value == "Guys"
        assert Site.HH.value == "HH"
        assert Site.IOP.value == "IOP"

    def test_site_from_string(self) -> None:
        assert Site.from_string("Guys") == Site.GUYS
        assert Site.from_string("HH") == Site.HH
        assert Site.from_string("IOP") == Site.IOP

    def test_site_from_string_invalid(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Unknown site"):
            Site.from_string("Unknown")

    def test_all_sites(self) -> None:
        assert len(Site) == 3


class TestSubject:
    def test_creation(self, tmp_path: Path) -> None:
        img = tmp_path / "test.nii.gz"
        subject = Subject(
            id="IXI002",
            site=Site.GUYS,
            image_path=img,
        )
        assert subject.id == "IXI002"
        assert subject.site == Site.GUYS
        assert subject.image_path == img
        assert subject.mask_path is None
        assert subject.age is None
        assert subject.sex is None

    def test_creation_with_all_fields(self, tmp_path: Path) -> None:
        img = tmp_path / "test.nii.gz"
        mask = tmp_path / "mask.nii.gz"
        subject = Subject(
            id="IXI002",
            site=Site.GUYS,
            image_path=img,
            mask_path=mask,
            age=35.5,
            sex="M",
        )
        assert subject.mask_path == mask
        assert subject.age == 35.5
        assert subject.sex == "M"

    def test_frozen(self, tmp_path: Path) -> None:
        import pytest

        subject = Subject(
            id="IXI002",
            site=Site.GUYS,
            image_path=tmp_path / "test.nii.gz",
        )
        with pytest.raises(AttributeError):
            subject.id = "IXI003"  # type: ignore[misc]


class TestHarmonizationMethod:
    def test_values(self) -> None:
        assert HarmonizationMethod.NONE.value == "none"
        assert HarmonizationMethod.ZSCORE.value == "zscore"
        assert HarmonizationMethod.WHITESTRIPE.value == "whitestripe"
        assert HarmonizationMethod.NYUL.value == "nyul"

    def test_image_level_methods(self) -> None:
        image_methods = HarmonizationMethod.image_level_methods()
        assert HarmonizationMethod.ZSCORE in image_methods
        assert HarmonizationMethod.WHITESTRIPE in image_methods
        assert HarmonizationMethod.NYUL in image_methods
        assert HarmonizationMethod.NONE not in image_methods
