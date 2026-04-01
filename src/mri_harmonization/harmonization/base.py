"""Base protocol for image-level harmonization methods."""

from typing import Protocol, runtime_checkable

import nibabel as nib


@runtime_checkable
class ImageHarmonizer(Protocol):
    """Protocol for image-level intensity harmonization."""

    @property
    def name(self) -> str:
        """Short identifier for this harmonization method."""
        ...

    def normalize(
        self,
        image: nib.Nifti1Image,
        mask: nib.Nifti1Image | None = None,
    ) -> nib.Nifti1Image:
        """Normalize image intensities.

        Args:
            image: Input NIfTI image.
            mask: Optional binary brain mask.

        Returns:
            Normalized NIfTI image with same shape and affine.
        """
        ...


@runtime_checkable
class TrainableHarmonizer(ImageHarmonizer, Protocol):
    """Protocol for harmonizers that require fitting on a population."""

    def fit(
        self,
        images: list[nib.Nifti1Image],
        masks: list[nib.Nifti1Image] | None = None,
    ) -> None:
        """Fit the harmonizer on a population of images.

        Args:
            images: List of NIfTI images.
            masks: Optional list of binary brain masks.
        """
        ...
