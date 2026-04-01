"""Pipeline configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for the MRI harmonization benchmark pipeline."""

    base_dir: Path
    subjects_per_site: int = 50
    icc_threshold: float = 0.75
    alpha: float = 0.05
    bin_width: int = 25
    feature_classes: list[str] = field(
        default_factory=lambda: ["firstorder", "glcm", "glrlm", "glszm", "shape"]
    )

    ixi_t1_url: str = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
    ixi_demographics_url: str = (
        "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls"
    )

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def preprocessed_dir(self) -> Path:
        return self.data_dir / "preprocessed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.csv"
