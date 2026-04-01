"""Parse IXI demographic data."""

from pathlib import Path

import pandas as pd

from mri_harmonization.types import Subject


def load_demographics(path: Path) -> dict[str, dict[str, float | str]]:
    """Load demographics from the IXI spreadsheet.

    Args:
        path: Path to IXI.xls file.

    Returns:
        Dict mapping subject ID (e.g. "IXI002") to {"age": float, "sex": str}.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Demographics file not found: {path}")

    df = pd.read_excel(path)

    sex_col = [c for c in df.columns if "SEX" in c.upper()][0]
    age_col = [c for c in df.columns if "AGE" in c.upper()][0]
    id_col = [c for c in df.columns if "IXI_ID" in c.upper()][0]

    demographics: dict[str, dict[str, float | str]] = {}
    for _, row in df.iterrows():
        ixi_id = int(row[id_col])
        subject_id = f"IXI{ixi_id:03d}"
        sex_value = int(row[sex_col]) if pd.notna(row[sex_col]) else None
        age_value = float(row[age_col]) if pd.notna(row[age_col]) else None

        sex_str = "M" if sex_value == 1 else "F" if sex_value == 2 else None

        entry: dict[str, float | str] = {}
        if age_value is not None:
            entry["age"] = age_value
        if sex_str is not None:
            entry["sex"] = sex_str
        demographics[subject_id] = entry

    return demographics


def merge_demographics(
    subjects: list[Subject],
    demographics: dict[str, dict[str, float | str]],
) -> list[Subject]:
    """Merge demographic data into subject list.

    Args:
        subjects: List of Subject objects.
        demographics: Dict from load_demographics().

    Returns:
        New list of Subject objects with age/sex populated where available.
    """
    merged: list[Subject] = []
    for s in subjects:
        demo = demographics.get(s.id, {})
        merged.append(
            Subject(
                id=s.id,
                site=s.site,
                image_path=s.image_path,
                mask_path=s.mask_path,
                age=demo.get("age"),  # type: ignore[arg-type]
                sex=demo.get("sex"),  # type: ignore[arg-type]
            )
        )
    return merged
