"""
Kaggle Bundle Exporter — package the complete ARC reasoning engine for Kaggle offline submissions.

Creates:
  1. `results/kaggle_bundle.zip`: Full source code zip for Kaggle Dataset attachment.
  2. `results/kaggle_submission_notebook.py`: Self-contained Python script to paste into Kaggle Notebook.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "results"


def export_bundle_zip(zip_path: Path) -> None:
    """Package src/, configs/, pyproject.toml into a zip archive."""
    print(f"Creating Kaggle bundle zip: {zip_path}...")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in ["src", "configs"]:
            folder_path = ROOT_DIR / folder
            for file in folder_path.rglob("*"):
                if file.is_file() and not file.name.endswith(".pyc") and "__pycache__" not in file.parts:
                    arcname = file.relative_to(ROOT_DIR)
                    zf.write(file, arcname)

        if (ROOT_DIR / "pyproject.toml").exists():
            zf.write(ROOT_DIR / "pyproject.toml", "pyproject.toml")

    print(f"[OK] Created {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")


def export_kaggle_notebook_script(script_path: Path) -> None:
    """Generate a single-file Python script to paste into a Kaggle Notebook."""
    print(f"Creating Kaggle notebook script: {script_path}...")
    script_path.parent.mkdir(parents=True, exist_ok=True)

    code_template = '''"""
Kaggle ARC-AGI-2 Offline Submission Notebook Entry Point.

Paste this into a Kaggle notebook and run to generate submission.json.
"""

import os
import sys
import json
from pathlib import Path

# Add bundle to Python path
sys.path.insert(0, ".")

from src.api.solver_api import ARCSolver
from src.submission.submission_generator import SubmissionGenerator

def run_kaggle_submission():
    print("=== ARC-AGI-2 Kaggle Submission Generator ===")
    
    # Path to Kaggle test dataset
    kaggle_test_dir = Path("/kaggle/input/arc-prize-2024/arc-agi_test-challenges.json")
    if not kaggle_test_dir.exists():
        # Fallback local path
        kaggle_test_dir = Path("data/datasets/evaluation")
        
    print(f"Reading test tasks from: {kaggle_test_dir}")
    solver = ARCSolver()
    submission_gen = SubmissionGenerator()
    
    if kaggle_test_dir.is_file():
        with open(kaggle_test_dir) as f:
            tasks = json.load(f)
        for task_id, task_dict in tasks.items():
            result = solver.solve(task_dict, task_id=task_id)
            submission_gen.add_from_solver_result(result)
    elif kaggle_test_dir.is_dir():
        for task_file in sorted(kaggle_test_dir.glob("*.json")):
            task_id = task_file.stem
            with open(task_file) as f:
                task_dict = json.load(f)
            result = solver.solve(task_dict, task_id=task_id)
            submission_gen.add_from_solver_result(result)
            
    # Save final submission.json
    output_path = Path("submission.json")
    submission_gen.save(output_path)
    print(f"[OK] Saved submission.json to {output_path.resolve()}")

if __name__ == "__main__":
    run_kaggle_submission()
'''

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code_template)

    print(f"[OK] Created {script_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_bundle_zip(OUTPUT_DIR / "kaggle_bundle.zip")
    export_kaggle_notebook_script(OUTPUT_DIR / "kaggle_submission_notebook.py")
    print("\nKaggle export complete! All files generated in results/")


if __name__ == "__main__":
    main()
