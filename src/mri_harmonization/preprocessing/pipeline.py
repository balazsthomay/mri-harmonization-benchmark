"""Preprocessing pipeline orchestration."""

import logging
from pathlib import Path

from mri_harmonization.preprocessing.bias_correction import apply_n4_correction
from mri_harmonization.preprocessing.brain_extraction import extract_brain
from mri_harmonization.types import Subject

logger = logging.getLogger(__name__)


def preprocess_subject(
    subject: Subject,
    output_dir: Path,
    device: str = "cpu",
) -> Subject:
    """Run full preprocessing on a single subject.

    Pipeline: N4 bias field correction -> brain extraction (skull stripping).

    Args:
        subject: Subject with raw image path.
        output_dir: Base output directory for preprocessed files.
        device: Device for HD-BET ('cpu', 'cuda', 'mps').

    Returns:
        New Subject with updated image_path and mask_path.
    """
    site_dir = output_dir / subject.site.value
    site_dir.mkdir(parents=True, exist_ok=True)

    n4_path = site_dir / f"{subject.id}_T1_n4.nii.gz"
    brain_path = site_dir / f"{subject.id}_T1_brain.nii.gz"
    mask_path = site_dir / f"{subject.id}_T1_mask.nii.gz"

    # Step 1: Create a simple initial mask for N4 (threshold-based)
    # For N4, we use a rough mask; the real mask comes from brain extraction
    _create_initial_mask(subject.image_path, site_dir / f"{subject.id}_T1_init_mask.nii.gz")
    init_mask_path = site_dir / f"{subject.id}_T1_init_mask.nii.gz"

    # Step 2: N4 bias field correction
    logger.info(f"Running N4 correction for {subject.id}")
    apply_n4_correction(subject.image_path, init_mask_path, n4_path)

    # Step 3: Brain extraction
    logger.info(f"Running brain extraction for {subject.id}")
    extract_brain(n4_path, brain_path, mask_path, device=device)

    # Clean up intermediate files
    n4_path.unlink(missing_ok=True)
    init_mask_path.unlink(missing_ok=True)

    return Subject(
        id=subject.id,
        site=subject.site,
        image_path=brain_path,
        mask_path=mask_path,
        age=subject.age,
        sex=subject.sex,
    )


def _create_initial_mask(image_path: Path, output_path: Path) -> None:
    """Create a rough initial mask for N4 using Otsu thresholding."""
    import SimpleITK as sitk

    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    sitk.WriteImage(mask, str(output_path))
