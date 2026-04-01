"""Image-level harmonization method implementations.

These are standard MRI intensity normalization algorithms implemented
directly due to compatibility issues between intensity-normalization
and nibabel 5.x (ExpiredDeprecationError on get_data()).
"""

import logging

import nibabel as nib
import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class ZScoreHarmonizer:
    """Z-score intensity normalization.

    Standardizes intensities within the brain mask to zero mean and unit variance.
    """

    @property
    def name(self) -> str:
        return "zscore"

    def normalize(
        self,
        image: nib.Nifti1Image,
        mask: nib.Nifti1Image | None = None,
    ) -> nib.Nifti1Image:
        data = image.get_fdata().copy()

        if mask is not None:
            mask_data = mask.get_fdata().astype(bool)
        else:
            mask_data = data > 0

        brain_values = data[mask_data]
        mean = brain_values.mean()
        std = brain_values.std()

        if std > 0:
            data[mask_data] = (data[mask_data] - mean) / std

        return nib.Nifti1Image(data.astype(np.float32), image.affine, image.header)


class WhiteStripeHarmonizer:
    """WhiteStripe intensity normalization.

    Identifies the white matter peak in the intensity histogram and normalizes
    so that the white matter has zero mean and unit variance.
    """

    @property
    def name(self) -> str:
        return "whitestripe"

    def normalize(
        self,
        image: nib.Nifti1Image,
        mask: nib.Nifti1Image | None = None,
    ) -> nib.Nifti1Image:
        data = image.get_fdata().copy()

        if mask is not None:
            mask_data = mask.get_fdata().astype(bool)
        else:
            mask_data = data > 0

        brain_values = data[mask_data]

        # Find WM peak using KDE-based histogram analysis
        # The WM peak is typically the largest peak in T1 images
        hist, bin_edges = np.histogram(brain_values, bins=200)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Smooth histogram to find peaks
        from scipy.ndimage import gaussian_filter1d

        smoothed = gaussian_filter1d(hist.astype(float), sigma=5)

        # Find the peak (highest point in smoothed histogram)
        peak_idx = np.argmax(smoothed)
        wm_peak = bin_centers[peak_idx]

        # Define white stripe as region around WM peak (+/- 10% of peak intensity)
        stripe_width = 0.10 * wm_peak if wm_peak > 0 else 10.0
        lower = wm_peak - stripe_width
        upper = wm_peak + stripe_width

        ws_values = brain_values[(brain_values >= lower) & (brain_values <= upper)]

        if len(ws_values) > 0:
            ws_mean = ws_values.mean()
            ws_std = ws_values.std()
            if ws_std > 0:
                data[mask_data] = (data[mask_data] - ws_mean) / ws_std

        return nib.Nifti1Image(data.astype(np.float32), image.affine, image.header)


class NyulHarmonizer:
    """Nyul piecewise linear histogram matching.

    Fits a standard histogram from a population of images, then maps each
    image's intensity percentiles to the standard scale using piecewise
    linear interpolation.

    Reference: Nyul et al. (2000) "New variants of a method of MRI scale
    standardization" IEEE TMI.
    """

    def __init__(self) -> None:
        self._fitted = False
        self._standard_scale: np.ndarray | None = None
        self._percentiles = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99])

    @property
    def name(self) -> str:
        return "nyul"

    def fit(
        self,
        images: list[nib.Nifti1Image],
        masks: list[nib.Nifti1Image] | None = None,
    ) -> None:
        """Fit the standard histogram from a population of images."""
        all_landmarks = []

        for i, image in enumerate(images):
            data = image.get_fdata()
            if masks is not None and masks[i] is not None:
                mask_data = masks[i].get_fdata().astype(bool)
            else:
                mask_data = data > 0

            brain_values = data[mask_data]
            landmarks = np.percentile(brain_values, self._percentiles)
            all_landmarks.append(landmarks)

        # Standard scale is the mean of all landmarks
        self._standard_scale = np.mean(all_landmarks, axis=0)
        self._fitted = True
        logger.info("Nyul normalizer fitted on %d images", len(images))

    def normalize(
        self,
        image: nib.Nifti1Image,
        mask: nib.Nifti1Image | None = None,
    ) -> nib.Nifti1Image:
        if not self._fitted or self._standard_scale is None:
            raise RuntimeError(
                "NyulHarmonizer must be fit() on a population before normalize()"
            )

        data = image.get_fdata().copy()

        if mask is not None:
            mask_data = mask.get_fdata().astype(bool)
        else:
            mask_data = data > 0

        brain_values = data[mask_data]
        landmarks = np.percentile(brain_values, self._percentiles)

        # Piecewise linear mapping from image landmarks to standard scale
        mapping = interp1d(
            landmarks,
            self._standard_scale,
            kind="linear",
            fill_value="extrapolate",
        )

        data[mask_data] = mapping(brain_values).astype(np.float32)

        return nib.Nifti1Image(data.astype(np.float32), image.affine, image.header)
