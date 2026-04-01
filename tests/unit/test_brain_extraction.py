"""Tests for brain extraction wrapper."""

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from mri_harmonization.preprocessing.brain_extraction import extract_brain


class TestBrainExtraction:
    def _mock_hd_bet(self, input_file: str, output_file: str, **kwargs) -> None:
        """Mock HD-BET v2 by creating a simple threshold-based mask.

        HD-BET v2 writes to:
        - {output_file} (brain-extracted image, must end with .nii.gz)
        - {output_file[:-7]}_bet.nii.gz (brain mask)
        """
        img = nib.load(input_file)
        data = img.get_fdata()
        mask = (data > data.mean()).astype(np.uint8)
        brain = data * mask

        nib.save(nib.Nifti1Image(brain, img.affine), output_file)
        mask_path = output_file[:-7] + "_bet.nii.gz"
        nib.save(nib.Nifti1Image(mask, img.affine), mask_path)

    @patch("mri_harmonization.preprocessing.brain_extraction._run_hd_bet")
    def test_produces_brain_and_mask(
        self, mock_bet, synthetic_nifti: Path, tmp_path: Path
    ) -> None:
        brain_path = tmp_path / "brain.nii.gz"
        mask_path = tmp_path / "brain_mask.nii.gz"

        mock_bet.side_effect = lambda inp, out, **kw: self._mock_hd_bet(inp, out, **kw)

        extract_brain(synthetic_nifti, brain_path, mask_path)

        assert brain_path.exists()
        assert mask_path.exists()

    @patch("mri_harmonization.preprocessing.brain_extraction._run_hd_bet")
    def test_mask_is_binary(
        self, mock_bet, synthetic_nifti: Path, tmp_path: Path
    ) -> None:
        brain_path = tmp_path / "brain.nii.gz"
        mask_path = tmp_path / "brain_mask.nii.gz"

        mock_bet.side_effect = lambda inp, out, **kw: self._mock_hd_bet(inp, out, **kw)

        extract_brain(synthetic_nifti, brain_path, mask_path)

        mask_data = nib.load(mask_path).get_fdata()
        unique_values = set(np.unique(mask_data))
        assert unique_values <= {0, 1}

    @patch("mri_harmonization.preprocessing.brain_extraction._run_hd_bet")
    def test_shapes_match(
        self, mock_bet, synthetic_nifti: Path, tmp_path: Path
    ) -> None:
        brain_path = tmp_path / "brain.nii.gz"
        mask_path = tmp_path / "brain_mask.nii.gz"

        mock_bet.side_effect = lambda inp, out, **kw: self._mock_hd_bet(inp, out, **kw)

        extract_brain(synthetic_nifti, brain_path, mask_path)

        input_shape = nib.load(synthetic_nifti).shape
        brain_shape = nib.load(brain_path).shape
        mask_shape = nib.load(mask_path).shape

        assert brain_shape == input_shape
        assert mask_shape == input_shape
