"""Shared test fixtures for synthetic medical imaging data."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def synthetic_nifti(tmp_path: Path) -> Path:
    """Create a small 3D NIfTI volume with non-uniform intensities."""
    rng = np.random.RandomState(42)
    data = rng.normal(100, 30, (32, 32, 32)).astype(np.float32)
    data = np.clip(data, 0, 255)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "test_image.nii.gz"
    nib.save(img, path)
    return path


@pytest.fixture
def synthetic_mask(tmp_path: Path) -> Path:
    """Create a binary brain mask (central sphere)."""
    mask = np.zeros((32, 32, 32), dtype=np.uint8)
    center = np.array([16, 16, 16])
    coords = np.mgrid[0:32, 0:32, 0:32].reshape(3, -1).T
    distances = np.linalg.norm(coords - center, axis=1)
    mask_flat = (distances < 12).astype(np.uint8)
    mask = mask_flat.reshape(32, 32, 32)
    img = nib.Nifti1Image(mask, np.eye(4))
    path = tmp_path / "test_mask.nii.gz"
    nib.save(img, path)
    return path


@pytest.fixture
def synthetic_nifti_with_bias(tmp_path: Path) -> Path:
    """Create a NIfTI volume with simulated bias field."""
    rng = np.random.RandomState(42)
    data = rng.normal(100, 20, (32, 32, 32)).astype(np.float32)
    data = np.clip(data, 10, 255)

    # Simulate a smooth bias field
    x = np.linspace(0.5, 1.5, 32)
    bias = x[:, None, None] * np.ones((32, 32, 32))
    data = (data * bias).astype(np.float32)

    img = nib.Nifti1Image(data, np.eye(4))
    path = tmp_path / "biased_image.nii.gz"
    nib.save(img, path)
    return path
