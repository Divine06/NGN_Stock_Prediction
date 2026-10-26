import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _setup  # noqa: E402,F401  (builds the synthetic dataset before tests import ml)
