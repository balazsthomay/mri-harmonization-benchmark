"""Domain types for the MRI harmonization benchmark."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Site(Enum):
    """MRI scanner sites from the IXI dataset."""

    GUYS = "Guys"
    HH = "HH"
    IOP = "IOP"

    @classmethod
    def from_string(cls, value: str) -> "Site":
        """Parse a site string into a Site enum member."""
        for site in cls:
            if site.value == value:
                return site
        raise ValueError(f"Unknown site: {value!r}. Valid sites: {[s.value for s in cls]}")


class HarmonizationMethod(Enum):
    """Available harmonization methods."""

    NONE = "none"
    ZSCORE = "zscore"
    WHITESTRIPE = "whitestripe"
    NYUL = "nyul"

    @classmethod
    def image_level_methods(cls) -> list["HarmonizationMethod"]:
        """Return only the image-level harmonization methods."""
        return [m for m in cls if m != cls.NONE]


@dataclass(frozen=True)
class Subject:
    """A single MRI subject with associated metadata."""

    id: str
    site: Site
    image_path: Path
    mask_path: Path | None = None
    age: float | None = None
    sex: str | None = None
