"""
Data downloader — automatically fetch ARC training data from GitHub.

Downloads:
  - ARC-AGI-1 training tasks (400 tasks)
  - ARC-AGI-1 evaluation tasks (400 tasks)
  - ARC-AGI-2 training tasks (if available)

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --output data/datasets
"""

from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path


# URLs for ARC datasets
SOURCES = {
    "arc1_training": {
        "url": "https://github.com/fchollet/ARC-AGI/raw/master/data/training/",
        "type": "dir",
        "out": "training",
    },
    "arc1_evaluation": {
        "url": "https://github.com/fchollet/ARC-AGI/raw/master/data/evaluation/",
        "type": "dir",
        "out": "evaluation",
    },
}

# Fallback: GitHub ZIP release
ARC_AGI1_ZIP_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
ARC_AGI2_ZIP_URL = "https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip"


def download_file(url: str, dest: Path, label: str = "") -> bool:
    """Download a file from *url* to *dest*.

    Returns:
        True on success, False on failure.
    """
    label = label or dest.name
    try:
        print(f"  Downloading {label}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        print("✓")
        return True
    except Exception as e:
        print(f"✗ ({e})")
        return False


def download_arc1(output_dir: Path) -> None:
    """Download ARC-AGI-1 dataset."""
    print("\n=== Downloading ARC-AGI-1 ===")
    zip_path = output_dir / "_arc1.zip"
    if not download_file(ARC_AGI1_ZIP_URL, zip_path, "ARC-AGI-1 archive"):
        print("Failed to download ARC-AGI-1.")
        return

    print("  Extracting...", end=" ", flush=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir / "_arc1_tmp")
    print("✓")

    # Move files to expected locations
    src_base = output_dir / "_arc1_tmp" / "ARC-AGI-master" / "data"
    for split in ("training", "evaluation"):
        src = src_base / split
        dst = output_dir / split
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for json_file in src.glob("*.json"):
                target = dst / json_file.name
                if not target.exists():
                    target.write_bytes(json_file.read_bytes())
            count = len(list(dst.glob("*.json")))
            print(f"  {split}: {count} tasks → {dst}")

    # Cleanup
    import shutil
    shutil.rmtree(output_dir / "_arc1_tmp", ignore_errors=True)
    zip_path.unlink(missing_ok=True)


def download_arc2(output_dir: Path) -> None:
    """Download ARC-AGI-2 dataset."""
    print("\n=== Downloading ARC-AGI-2 ===")
    zip_path = output_dir / "_arc2.zip"
    if not download_file(ARC_AGI2_ZIP_URL, zip_path, "ARC-AGI-2 archive"):
        print("Failed to download ARC-AGI-2.")
        return

    print("  Extracting...", end=" ", flush=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir / "_arc2_tmp")
        print("✓")

        # Move files
        for sub_dir in (output_dir / "_arc2_tmp").rglob("*.json"):
            parts = sub_dir.parts
            for split_name in ("training", "evaluation", "test"):
                if split_name in parts:
                    dst = output_dir / f"arc2_{split_name}"
                    dst.mkdir(parents=True, exist_ok=True)
                    target = dst / sub_dir.name
                    if not target.exists():
                        target.write_bytes(sub_dir.read_bytes())

        import shutil
        shutil.rmtree(output_dir / "_arc2_tmp", ignore_errors=True)
        zip_path.unlink(missing_ok=True)

        for split in ("arc2_training", "arc2_evaluation"):
            dst = output_dir / split
            if dst.exists():
                count = len(list(dst.glob("*.json")))
                print(f"  {split}: {count} tasks → {dst}")
    except Exception as e:
        print(f"✗ Error processing ARC-AGI-2: {e}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Download ARC datasets from GitHub")
    parser.add_argument(
        "--output", "-o",
        default="data/datasets",
        help="Output directory for datasets (default: data/datasets)",
    )
    parser.add_argument(
        "--arc2", action="store_true",
        help="Also download ARC-AGI-2 dataset",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    download_arc1(output_dir)
    if args.arc2:
        download_arc2(output_dir)

    # Summary
    print("\n=== Download Summary ===")
    for sub in sorted(output_dir.iterdir()):
        if sub.is_dir():
            n = len(list(sub.glob("*.json")))
            print(f"  {sub.name}/: {n} tasks")
    print("\nDone! Run: python scripts/solve.py --help")


if __name__ == "__main__":
    main()
