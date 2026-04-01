"""Tests for N4 bias field correction."""

from pathlib import Path

import nibabel as nib
import numpy as np

from mri_harmonization.preprocessing.bias_correction import apply_n4_correction


class TestN4BiasCorrection:
    def test_output_shape_matches_input(
        self, synthetic_nifti_with_bias: Path, synthetic_mask: Path, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "corrected.nii.gz"
        apply_n4_correction(synthetic_nifti_with_bias, synthetic_mask, output_path)

        input_img = nib.load(synthetic_nifti_with_bias)
        output_img = nib.load(output_path)

        assert output_img.shape == input_img.shape

    def test_output_values_changed(
        self, synthetic_nifti_with_bias: Path, synthetic_mask: Path, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "corrected.nii.gz"
        apply_n4_correction(synthetic_nifti_with_bias, synthetic_mask, output_path)

        input_data = nib.load(synthetic_nifti_with_bias).get_fdata()
        output_data = nib.load(output_path).get_fdata()

        # N4 should modify the image (not identical)
        assert not np.allclose(input_data, output_data)

    def test_output_is_finite(
        self, synthetic_nifti_with_bias: Path, synthetic_mask: Path, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "corrected.nii.gz"
        apply_n4_correction(synthetic_nifti_with_bias, synthetic_mask, output_path)

        output_data = nib.load(output_path).get_fdata()
        assert np.all(np.isfinite(output_data))

    def test_creates_output_file(
        self, synthetic_nifti_with_bias: Path, synthetic_mask: Path, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "sub" / "corrected.nii.gz"
        apply_n4_correction(synthetic_nifti_with_bias, synthetic_mask, output_path)
        assert output_path.exists()
