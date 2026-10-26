"""Run the tests without pytest:  python tests/run_tests.py
(With pytest installed, `pytest tests` works too.)"""
import importlib
import inspect
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _setup  # noqa: E402,F401


def main() -> int:
    passed, failed, skipped = 0, 0, 0
    for path in sorted(HERE.glob("test_*.py")):
        mod = importlib.import_module(path.stem)
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != mod.__name__:
                continue
            kwargs = {}
            if "tmp_path" in inspect.signature(fn).parameters:
                kwargs["tmp_path"] = Path(tempfile.mkdtemp())
            t0 = time.time()
            try:
                fn(**kwargs)
                passed += 1
                print(f"PASS {path.stem}.{name} ({time.time() - t0:.1f}s)")
            except unittest.SkipTest as exc:
                skipped += 1
                print(f"SKIP {path.stem}.{name}: {exc}")
            except Exception:
                failed += 1
                print(f"FAIL {path.stem}.{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
