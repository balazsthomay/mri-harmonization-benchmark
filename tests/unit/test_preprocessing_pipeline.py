"""Tests for the preprocessing pipeline orchestration."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import nibabel as nib
import numpy as np

from mri_harmonization.preprocessing.pipeline import preprocess_subject
from mri_harmonization.types import Site, Subject


class TestPreprocessSubject:
    def _mock_n4(self, input_path, mask_path, output_path):
        """Mock N4 by copying the input."""
        import shutil
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(input_path, output_path)

    def _mock_extract_brain(self, input_path, brain_path, mask_path, **kwargs):
        """Mock brain extraction."""
        img = nib.load(input_path)
        data = img.get_fdata()
        mask = (data > data.mean()).astype(np.uint8)

        brain_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data * mask, img.affine), brain_path)
        nib.save(nib.Nifti1Image(mask, img.affine), mask_path)

    @patch("mri_harmonization.preprocessing.pipeline.extract_brain")
    @patch("mri_harmonization.preprocessing.pipeline.apply_n4_correction")
    def test_produces_preprocessed_subject(
        self,
        mock_n4,
        mock_extract,
        synthetic_nifti: Path,
        tmp_path: Path,
    ) -> None:
        mock_n4.side_effect = self._mock_n4
        mock_extract.side_effect = self._mock_extract_brain

        subject = Subject(
            id="IXI002",
            site=Site.GUYS,
            image_path=synthetic_nifti,
        )
        output_dir = tmp_path / "preprocessed"

        result = preprocess_subject(subject, output_dir)

        assert result.mask_path is not None
        assert result.site == Site.GUYS
        assert result.id == "IXI002"
        # Verify N4 was called
        mock_n4.assert_called_once()
        # Verify brain extraction was called
        mock_extract.assert_called_once()

    @patch("mri_harmonization.preprocessing.pipeline.extract_brain")
    @patch("mri_harmonization.preprocessing.pipeline.apply_n4_correction")
    def test_output_organized_by_site(
        self,
        mock_n4,
        mock_extract,
        synthetic_nifti: Path,
        tmp_path: Path,
    ) -> None:
        mock_n4.side_effect = self._mock_n4
        mock_extract.side_effect = self._mock_extract_brain

        subject = Subject(
            id="IXI002",
            site=Site.GUYS,
            image_path=synthetic_nifti,
        )
        output_dir = tmp_path / "preprocessed"

        result = preprocess_subject(subject, output_dir)

        # Check files are in site subdirectory
        assert "Guys" in str(result.image_path)
