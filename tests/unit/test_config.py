"""Tests for configuration module."""

from pathlib import Path

from mri_harmonization.config import PipelineConfig


class TestPipelineConfig:
    def test_default_paths(self, tmp_path: Path) -> None:
        config = PipelineConfig(base_dir=tmp_path)
        assert config.raw_dir == tmp_path / "data" / "raw"
        assert config.preprocessed_dir == tmp_path / "data" / "preprocessed"
        assert config.features_dir == tmp_path / "data" / "features"
        assert config.results_dir == tmp_path / "data" / "results"
        assert config.manifest_path == tmp_path / "data" / "manifest.csv"

    def test_subjects_per_site(self) -> None:
        config = PipelineConfig(base_dir=Path("/tmp"))
        assert config.subjects_per_site == 50

    def test_custom_subjects_per_site(self) -> None:
        config = PipelineConfig(base_dir=Path("/tmp"), subjects_per_site=10)
        assert config.subjects_per_site == 10

    def test_site_dirs(self, tmp_path: Path) -> None:
        config = PipelineConfig(base_dir=tmp_path)
        from mri_harmonization.types import Site

        for site in Site:
            site_dir = config.preprocessed_dir / site.value
            assert isinstance(site_dir, Path)

    def test_ixi_tar_url(self) -> None:
        config = PipelineConfig(base_dir=Path("/tmp"))
        assert "IXI-T1" in config.ixi_t1_url

    def test_feature_classes(self) -> None:
        config = PipelineConfig(base_dir=Path("/tmp"))
        assert "firstorder" in config.feature_classes
        assert "glcm" in config.feature_classes
        assert "glrlm" in config.feature_classes
        assert "glszm" in config.feature_classes
        assert "shape" in config.feature_classes

    def test_icc_threshold(self) -> None:
        config = PipelineConfig(base_dir=Path("/tmp"))
        assert config.icc_threshold == 0.75
