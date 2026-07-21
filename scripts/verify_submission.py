"""
Kaggle Offline Sandbox Verification Script — tests standalone execution of exported notebook submission.

Usage:
    python scripts/verify_submission.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from rich.console import Console

console = Console()


def verify_kaggle_submission() -> bool:
    """Executes generated submission notebook script in isolated subprocess."""
    console.print("[bold yellow]Verifying Kaggle Standalone Submission Bundle...[/bold yellow]\n")

    notebook_script = root_dir / "results" / "kaggle_submission_notebook.py"
    if not notebook_script.exists():
        # Export notebook first if not present
        from scripts.build_kaggle_release import build_release
        build_release()

    if not notebook_script.exists():
        console.print("[bold red]Submission notebook file not found![/bold red]")
        return False

    # Execute in clean subprocess with PYTHONPATH unset to verify self-containment
    env = os.environ.copy()
    env["PYTHONPATH"] = ""

    console.print(f"Executing {notebook_script.name} in offline sandbox environment...")
    res = subprocess.run([sys.executable, str(notebook_script)], env=env, cwd=root_dir)

    if res.returncode == 0:
        console.print("[bold green]✔ Offline submission notebook executed successfully![/bold green]")
        return True
    else:
        console.print(f"[bold red]Submission execution failed with code {res.returncode}![/bold red]")
        return False


if __name__ == "__main__":
    ok = verify_kaggle_submission()
    sys.exit(0 if ok else 1)
