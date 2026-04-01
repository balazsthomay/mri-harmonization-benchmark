"""Tests for radiomics feature extraction."""

from pathlib import Path

import nibabel as nib
import numpy as np

from mri_harmonization.features.extractor import ExtractionConfig, FeatureExtractor


class TestFeatureExtractor:
    def test_extract_returns_dict(
        self, synthetic_nifti: Path, synthetic_mask: Path
    ) -> None:
        config = ExtractionConfig(feature_classes=["firstorder", "shape"])
        extractor = FeatureExtractor(config)
        features = extractor.extract(synthetic_nifti, synthetic_mask)

        assert isinstance(features, dict)
        assert len(features) > 0

    def test_extract_contains_firstorder_features(
        self, synthetic_nifti: Path, synthetic_mask: Path
    ) -> None:
        config = ExtractionConfig(feature_classes=["firstorder"])
        extractor = FeatureExtractor(config)
        features = extractor.extract(synthetic_nifti, synthetic_mask)

        firstorder_keys = [k for k in features if "firstorder" in k]
        assert len(firstorder_keys) > 0

    def test_extract_contains_shape_features(
        self, synthetic_nifti: Path, synthetic_mask: Path
    ) -> None:
        config = ExtractionConfig(feature_classes=["shape"])
        extractor = FeatureExtractor(config)
        features = extractor.extract(synthetic_nifti, synthetic_mask)

        shape_keys = [k for k in features if "shape" in k]
        assert len(shape_keys) > 0

    def test_extract_glcm(
        self, synthetic_nifti: Path, synthetic_mask: Path
    ) -> None:
        config = ExtractionConfig(feature_classes=["glcm"])
        extractor = FeatureExtractor(config)
        features = extractor.extract(synthetic_nifti, synthetic_mask)

        glcm_keys = [k for k in features if "glcm" in k]
        assert len(glcm_keys) > 0

    def test_values_are_finite(
        self, synthetic_nifti: Path, synthetic_mask: Path
    ) -> None:
        config = ExtractionConfig(feature_classes=["firstorder", "shape"])
        extractor = FeatureExtractor(config)
        features = extractor.extract(synthetic_nifti, synthetic_mask)

        for name, value in features.items():
            assert np.isfinite(value), f"Non-finite value for {name}: {value}"

    def test_all_feature_classes(
        self, synthetic_nifti: Path, synthetic_mask: Path
    ) -> None:
        config = ExtractionConfig(
            feature_classes=["firstorder", "glcm", "glrlm", "glszm", "shape"]
        )
        extractor = FeatureExtractor(config)
        features = extractor.extract(synthetic_nifti, synthetic_mask)

        # Should have features from all requested classes
        assert any("firstorder" in k for k in features)
        assert any("glcm" in k for k in features)
        assert any("glrlm" in k for k in features)
        assert any("glszm" in k for k in features)
        assert any("shape" in k for k in features)
