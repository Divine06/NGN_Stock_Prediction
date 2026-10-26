"""Shared test setup: builds a synthetic dataset in a temp folder and points the
ml package at it through environment variables (before ml is imported)."""
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent          # folder that contains backend/
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))

try:
    import statsmodels  # noqa: F401
    USING_ARIMA_STUB = False
except ImportError:
    sys.path.insert(0, str(HERE / "stubs"))
    USING_ARIMA_STUB = True

ROOT = Path(tempfile.mkdtemp(prefix="ngx_ml_test_"))
os.environ["NGX_DATA_DIR"] = str(ROOT / "data")
os.environ["NGX_MODELS_DIR"] = str(ROOT / "models")
os.environ.pop("NGX_DB_PATH", None)
from synthetic import TICKERS as _SYN  # noqa: E402

TEST_TICKERS = list(_SYN) + ["SYNSHORT", "SYNGAP", "SYNLSE"]
os.environ["NGX_TICKERS"] = ",".join(TEST_TICKERS)

from synthetic import build_dataset  # noqa: E402

DATASET = build_dataset(ROOT)
