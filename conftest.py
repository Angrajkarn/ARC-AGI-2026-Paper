"""pytest conftest — add project root to sys.path so 'src' is importable."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
