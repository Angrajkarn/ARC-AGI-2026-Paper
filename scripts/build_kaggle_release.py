"""
Kaggle Release Pipeline Builder CLI — automated test runner, ZIP packager, and notebook bundle exporter.

Usage:
    python scripts/build_kaggle_release.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

# Ensure project root is on path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from rich.console import Console

console = Console()


def build_release() -> bool:
    """Builds full Kaggle release bundle."""
    console.print("[bold green]Starting ARC-AGI-2026 Kaggle Release Build Pipeline...[/bold green]\n")

    # 1. Run unit test suite (ignore recursive self-test)
    console.print("[yellow]1. Running Pytest Suite...[/yellow]")
    ret = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--ignore=tests/unit/test_release_builder.py"],
        cwd=root_dir,
    )
    if ret.returncode != 0:
        console.print("[bold red]Pytest suite failed! Aborting release build.[/bold red]")
        return False
    console.print("[bold green]✔ All unit tests passed![/bold green]\n")

    # 2. Export standalone Kaggle notebook script
    console.print("[yellow]2. Exporting Kaggle Notebook Bundle...[/yellow]")
    from src.submission.kaggle_exporter import KaggleExporter
    exporter = KaggleExporter(root_dir=root_dir)
    notebook_path = root_dir / "results" / "kaggle_submission_notebook.py"
    exporter.export(notebook_path)
    console.print(f"[bold green]✔ Standalone notebook generated at {notebook_path}[/bold green]\n")

    # 3. Create zip archive
    console.print("[yellow]3. Building Kaggle Submission ZIP Archive...[/yellow]")
    zip_path = root_dir / "results" / "kaggle_submission.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(notebook_path, arcname="kaggle_submission_notebook.py")

    console.print(f"[bold green]✔ Release ZIP created at {zip_path}[/bold green]\n")
    console.print("[bold blue]✨ Release Pipeline Build Succeeded![/bold blue]")
    return True


if __name__ == "__main__":
    success = build_release()
    sys.exit(0 if success else 1)
