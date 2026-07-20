"""
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
