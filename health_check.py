"""
health_check.py
───────────────
Minimal import validation script called by startup.sh before the main app
is launched. Antigravity (or any CI/CD pipeline) can run this independently:

    python health_check.py

Exit codes:
    0  — all imports and basic checks passed  → prints "OK"
    1  — at least one check failed            → prints the specific error
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _check(label: str, fn) -> bool:
    """
    Execute *fn* and print a pass/fail line.

    Parameters
    ----------
    label : str   Human-readable check name.
    fn    : callable  Zero-argument callable; raises on failure.

    Returns
    -------
    bool  True if *fn* succeeded, False otherwise.
    """
    try:
        fn()
        print(f"  [ OK ]  {label}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL]  {label}")
        print(f"          {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    """
    Run all health checks and return an exit code.

    Returns
    -------
    int   0 on full success, 1 if any check failed.
    """
    print("Health check - Zomato Restaurant Recommender")
    print("-" * 50)

    results: list[bool] = []

    # ── Standard-library imports ───────────────────────────────────────────
    results.append(_check("import os, sys, pathlib",
                           lambda: None))  # already imported above

    results.append(_check("import re, hashlib, logging",
                           lambda: __import__("re") and
                                   __import__("hashlib") and
                                   __import__("logging")))

    # ── Third-party imports ────────────────────────────────────────────────
    results.append(_check("import pandas",
                           lambda: __import__("pandas")))

    results.append(_check("import numpy",
                           lambda: __import__("numpy")))

    results.append(_check("import sklearn (TfidfVectorizer + linear_kernel)",
                           lambda: (
                               __import__(
                                   "sklearn.feature_extraction.text",
                                   fromlist=["TfidfVectorizer"]
                               ),
                               __import__(
                                   "sklearn.metrics.pairwise",
                                   fromlist=["linear_kernel"]
                               ),
                           )))

    results.append(_check("import joblib",
                           lambda: __import__("joblib")))

    results.append(_check("import pyarrow (parquet support)",
                           lambda: __import__("pyarrow")))

    results.append(_check("import streamlit",
                           lambda: __import__("streamlit")))

    # ── Project-module imports ─────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    results.append(_check("import preprocess",
                           lambda: __import__("preprocess")))

    results.append(_check("import recommender",
                           lambda: __import__("recommender")))

    # ── Data file presence ─────────────────────────────────────────────────
    data_path = Path(os.environ.get("DATA_PATH", "data/zomato.csv"))
    results.append(_check(
        f"dataset exists at '{data_path}'",
        lambda: (_ for _ in ()).throw(          # noqa: E731
            FileNotFoundError(f"Not found: {data_path}")
        ) if not data_path.exists() else None,
    ))

    # ── Cache directory writable ───────────────────────────────────────────
    def _check_cache_writable() -> None:
        cache = Path("cache")
        cache.mkdir(exist_ok=True)
        probe = cache / ".write_test"
        probe.touch()
        probe.unlink()

    results.append(_check("cache/ directory is writable",
                           _check_cache_writable))

    # ── Summary ────────────────────────────────────────────────────────────
    print("─" * 50)
    passed = sum(results)
    total  = len(results)

    if passed == total:
        print(f"OK  ({passed}/{total} checks passed)")
        return 0

    print(f"FAILED  ({passed}/{total} checks passed — see errors above)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
