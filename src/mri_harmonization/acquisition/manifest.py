"""Subject manifest management for the IXI dataset."""

import csv
import re
from pathlib import Path

from mri_harmonization.types import Site, Subject

_IXI_FILENAME_PATTERN = re.compile(
    r"^(IXI\d+)-(Guys|HH|IOP)-\d+-T1\.nii\.gz$"
)


def parse_ixi_filename(filename: str | Path) -> tuple[Site, str]:
    """Parse an IXI filename to extract site and subject ID.

    Args:
        filename: e.g. "IXI002-Guys-0828-T1.nii.gz"

    Returns:
        Tuple of (Site, subject_id)

    Raises:
        ValueError: If filename doesn't match expected IXI pattern.
    """
    name = Path(filename).name
    match = _IXI_FILENAME_PATTERN.match(name)
    if not match:
        raise ValueError(f"Cannot parse IXI filename: {name!r}")
    subject_id = match.group(1)
    site = Site.from_string(match.group(2))
    return site, subject_id


def build_manifest_from_directory(directory: Path) -> list[Subject]:
    """Build a subject list from a directory of IXI NIfTI files.

    Args:
        directory: Directory containing IXI T1 NIfTI files.

    Returns:
        List of Subject objects parsed from filenames.
    """
    subjects: list[Subject] = []
    for path in sorted(directory.glob("*.nii.gz")):
        try:
            site, subject_id = parse_ixi_filename(path)
        except ValueError:
            continue
        subjects.append(
            Subject(id=subject_id, site=site, image_path=path)
        )
    return subjects


def save_manifest(subjects: list[Subject], path: Path) -> None:
    """Save a subject manifest to CSV.

    Args:
        subjects: List of Subject objects to save.
        path: Output CSV file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "site", "image_path", "mask_path", "age", "sex"]
        )
        writer.writeheader()
        for s in subjects:
            writer.writerow(
                {
                    "id": s.id,
                    "site": s.site.value,
                    "image_path": str(s.image_path),
                    "mask_path": str(s.mask_path) if s.mask_path else "",
                    "age": s.age if s.age is not None else "",
                    "sex": s.sex if s.sex is not None else "",
                }
            )


def load_manifest(path: Path) -> list[Subject]:
    """Load a subject manifest from CSV.

    Args:
        path: Path to the manifest CSV file.

    Returns:
        List of Subject objects.
    """
    subjects: list[Subject] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subjects.append(
                Subject(
                    id=row["id"],
                    site=Site.from_string(row["site"]),
                    image_path=Path(row["image_path"]),
                    mask_path=Path(row["mask_path"]) if row["mask_path"] else None,
                    age=float(row["age"]) if row["age"] else None,
                    sex=row["sex"] if row["sex"] else None,
                )
            )
    return subjects
