"""N4 bias field correction using SimpleITK."""

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def apply_n4_correction(
    input_path: Path,
    mask_path: Path,
    output_path: Path,
) -> None:
    """Apply N4 bias field correction to a NIfTI image.

    Args:
        input_path: Path to the input NIfTI image.
        mask_path: Path to the binary brain mask.
        output_path: Path for the corrected output image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read with SimpleITK for N4
    input_image = sitk.ReadImage(str(input_path), sitk.sitkFloat32)
    mask_image = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)

    # Configure N4 with conservative settings for robustness
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrector.SetConvergenceThreshold(1e-6)

    corrected = corrector.Execute(input_image, mask_image)

    sitk.WriteImage(corrected, str(output_path))
    logger.info(f"N4 correction complete: {output_path}")
