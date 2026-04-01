"""Radiomics feature extraction using pyradiomics."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for radiomics feature extraction."""

    feature_classes: list[str] = field(
        default_factory=lambda: ["firstorder", "glcm", "glrlm", "glszm", "shape"]
    )
    bin_width: int = 25


class FeatureExtractor:
    """Extracts radiomics features from NIfTI images using pyradiomics.

    C extensions are disabled to avoid segfaults on Python 3.12.
    Pure-Python mode is functionally identical, just slower.
    """

    def __init__(self, config: ExtractionConfig) -> None:
        self._config = config
        self._setup_radiomics()

    def _setup_radiomics(self) -> None:
        """Configure pyradiomics."""
        import radiomics

        # Disable C extensions if available (removed in newer versions)
        if hasattr(radiomics, "enableCExtensions"):
            radiomics.enableCExtensions(False)
        radiomics.setVerbosity(logging.WARNING)

    def extract(self, image_path: Path, mask_path: Path) -> dict[str, float]:
        """Extract radiomics features from an image with a mask.

        Args:
            image_path: Path to the NIfTI image.
            mask_path: Path to the binary mask NIfTI.

        Returns:
            Dict mapping feature names to values.
        """
        from radiomics import featureextractor

        settings = {
            "binWidth": self._config.bin_width,
            "resampledPixelSpacing": None,
            "interpolator": "sitkBSpline",
            "force2D": False,
        }

        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

        # Disable all classes first, then enable requested ones
        extractor.disableAllFeatures()
        for feat_class in self._config.feature_classes:
            extractor.enableFeatureClassByName(feat_class)

        result = extractor.execute(str(image_path), str(mask_path))

        # Filter out diagnostic features (they start with "diagnostics_")
        features: dict[str, float] = {}
        for key, value in result.items():
            if key.startswith("diagnostics_"):
                continue
            features[key] = float(np.float64(value))

        return features
