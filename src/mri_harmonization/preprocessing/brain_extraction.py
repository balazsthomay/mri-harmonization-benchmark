"""Brain extraction (skull stripping) using HD-BET v2."""

import logging
import shutil
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Module-level predictor cache to avoid reloading model weights per subject
_predictor = None


def _get_predictor(device: str = "cpu"):
    """Get or create the HD-BET predictor (cached)."""
    global _predictor
    if _predictor is None:
        from HD_BET.entry_point import get_hdbet_predictor

        torch_device = torch.device(device)
        _predictor = get_hdbet_predictor(use_tta=False, device=torch_device)
        logger.info(f"HD-BET predictor loaded on {device}")
    return _predictor


def _run_hd_bet(input_file: str, output_file: str, **kwargs) -> None:
    """Run HD-BET brain extraction. Separated for easy mocking in tests.

    HD-BET v2 API:
    - output_file must end with .nii.gz
    - Brain mask is saved as {output_file[:-7]}_bet.nii.gz
    - Brain-extracted image is saved as {output_file}
    """
    device = kwargs.get("device", "cpu")
    predictor = _get_predictor(device)

    from HD_BET.entry_point import hdbet_predict

    hdbet_predict(
        input_file,
        output_file,
        predictor,
        keep_brain_mask=True,
        compute_brain_extracted_image=True,
    )


def extract_brain(
    input_path: Path,
    brain_path: Path,
    mask_path: Path,
    device: str = "cpu",
) -> None:
    """Extract brain from a T1 MRI using HD-BET.

    Args:
        input_path: Path to the input NIfTI image.
        brain_path: Desired path for the brain-extracted image.
        mask_path: Desired path for the binary brain mask.
        device: Computation device ('cpu', 'cuda', 'mps').
    """
    brain_path.parent.mkdir(parents=True, exist_ok=True)

    # HD-BET v2: output_file must end with .nii.gz
    # It saves brain at output_file and mask at {output_file[:-7]}_bet.nii.gz
    _run_hd_bet(
        str(input_path),
        str(brain_path),
        device=device,
    )

    # HD-BET creates mask at {brain_path[:-7]}_bet.nii.gz
    hd_bet_mask = Path(str(brain_path)[:-7] + "_bet.nii.gz")

    # Move mask to desired location
    if hd_bet_mask.exists() and hd_bet_mask != mask_path:
        shutil.move(str(hd_bet_mask), str(mask_path))
    elif not hd_bet_mask.exists() and not mask_path.exists():
        logger.warning(f"HD-BET mask not found at expected path: {hd_bet_mask}")

    logger.info(f"Brain extraction complete: {brain_path}, {mask_path}")
