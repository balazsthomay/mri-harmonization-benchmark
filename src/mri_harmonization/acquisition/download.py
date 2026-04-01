"""IXI dataset download and extraction."""

import logging
import tarfile
from collections import defaultdict
from pathlib import Path

import requests

from mri_harmonization.acquisition.manifest import parse_ixi_filename

logger = logging.getLogger(__name__)


def select_subjects_per_site(
    filenames: list[str], subjects_per_site: int
) -> list[str]:
    """Select a balanced subset of subjects across sites.

    Args:
        filenames: All available IXI T1 filenames.
        subjects_per_site: Number of subjects to select per site.

    Returns:
        Selected filenames, balanced across sites.
    """
    by_site: dict[str, list[str]] = defaultdict(list)
    for name in sorted(filenames):
        try:
            site, _ = parse_ixi_filename(name)
            by_site[site.value].append(name)
        except ValueError:
            continue

    selected: list[str] = []
    for site_name in sorted(by_site.keys()):
        site_files = by_site[site_name]
        n = min(len(site_files), subjects_per_site)
        selected.extend(site_files[:n])

    return selected


def list_tar_contents(tar_path: Path) -> list[str]:
    """List all filenames in a tar archive.

    Args:
        tar_path: Path to the tar file.

    Returns:
        List of member names in the archive.
    """
    with tarfile.open(tar_path, "r") as tar:
        return [m.name for m in tar.getmembers() if m.isfile()]


def extract_subjects_from_tar(
    tar_path: Path,
    output_dir: Path,
    filenames_to_extract: list[str],
) -> None:
    """Extract selected files from a tar archive.

    Args:
        tar_path: Path to the tar archive.
        output_dir: Directory to extract files into.
        filenames_to_extract: Set of filenames to extract.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extract_set = set(filenames_to_extract)

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name in extract_set and member.isfile():
                # Extract to flat directory (no subdirs from tar)
                member.name = Path(member.name).name
                tar.extract(member, output_dir, filter="data")


def download_file(url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a file with progress logging.

    Args:
        url: URL to download from.
        output_path: Where to save the file.
        chunk_size: Download chunk size in bytes (default 1MB).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    last_reported = 0

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                # Report every 5%
                if int(pct / 5) > int(last_reported / 5):
                    logger.info(f"Download progress: {pct:.0f}% ({downloaded / 1024 / 1024:.0f} MB)")
                    last_reported = pct

    logger.info(f"Downloaded {output_path.name} ({downloaded / 1024 / 1024:.0f} MB)")
