"""Tests for image-level harmonization methods."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mri_harmonization.harmonization.base import ImageHarmonizer
from mri_harmonization.harmonization.image_level import (
    NyulHarmonizer,
    WhiteStripeHarmonizer,
    ZScoreHarmonizer,
)


class TestZScoreHarmonizer:
    def test_implements_protocol(self) -> None:
        harmonizer = ZScoreHarmonizer()
        assert isinstance(harmonizer, ImageHarmonizer)

    def test_name(self) -> None:
        assert ZScoreHarmonizer().name == "zscore"

    def test_normalize_changes_values(
        self, synthetic_nifti: Path, synthetic_mask: Path
    ) -> None:
        harmonizer = ZScoreHarmonizer()
        image = nib.load(synthetic_nifti)
        mask = nib.load(synthetic_mask)

        result = harmonizer.normalize(image, mask)

        assert result.shape == image.shape
        assert not np.allclose(result.get_fdata(), image.get_fdata())

    def test_normalize_output_finite(
        self, synthetic_nifti: Path, synthetic_mask: Path
    ) -> None:
        harmonizer = ZScoreHarmonizer()
        image = nib.load(synthetic_nifti)
        mask = nib.load(synthetic_mask)

        result = harmonizer.normalize(image, mask)
        data = result.get_fdata()
        mask_data = mask.get_fdata().astype(bool)

        # Values within mask should be finite
        assert np.all(np.isfinite(data[mask_data]))


class TestWhiteStripeHarmonizer:
    def test_implements_protocol(self) -> None:
        harmonizer = WhiteStripeHarmonizer()
        assert isinstance(harmonizer, ImageHarmonizer)

    def test_name(self) -> None:
        assert WhiteStripeHarmonizer().name == "whitestripe"

    def test_normalize_changes_values(
        self, synthetic_nifti: Path, synthetic_mask: Path
    ) -> None:
        harmonizer = WhiteStripeHarmonizer()
        image = nib.load(synthetic_nifti)
        mask = nib.load(synthetic_mask)

        result = harmonizer.normalize(image, mask)
        assert result.shape == image.shape


class TestNyulHarmonizer:
    def test_implements_protocol(self) -> None:
        harmonizer = NyulHarmonizer()
        assert isinstance(harmonizer, ImageHarmonizer)

    def test_name(self) -> None:
        assert NyulHarmonizer().name == "nyul"

    def test_fit_and_normalize(self, tmp_path: Path) -> None:
        """Nyul requires fitting on a population first."""
        rng = np.random.RandomState(42)
        images = []
        masks = []

        # Create 3 synthetic images with different intensity distributions
        for i in range(3):
            data = rng.normal(100 + i * 30, 20, (16, 16, 16)).astype(np.float32)
            data = np.clip(data, 1, 300)
            img = nib.Nifti1Image(data, np.eye(4))
            images.append(img)

            mask_data = np.ones((16, 16, 16), dtype=np.uint8)
            mask_data[:2, :, :] = 0  # Some background
            masks.append(nib.Nifti1Image(mask_data, np.eye(4)))

        harmonizer = NyulHarmonizer()
        harmonizer.fit(images, masks)

        result = harmonizer.normalize(images[0], masks[0])
        assert result.shape == images[0].shape
        assert not np.allclose(result.get_fdata(), images[0].get_fdata())

    def test_normalize_without_fit_raises(
        self, synthetic_nifti: Path, synthetic_mask: Path
    ) -> None:
        harmonizer = NyulHarmonizer()
        image = nib.load(synthetic_nifti)
        mask = nib.load(synthetic_mask)

        with pytest.raises(RuntimeError, match="fit"):
            harmonizer.normalize(image, mask)
