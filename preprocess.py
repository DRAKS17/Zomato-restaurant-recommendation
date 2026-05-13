"""
preprocess.py
─────────────
Data cleaning and feature engineering pipeline for the Zomato Bangalore dataset.

Public API
----------
    load_and_clean(filepath: str) -> pd.DataFrame

The function executes 8 ordered steps:
    1. Load & drop exact duplicates
    2. Column selection & rename
    3. Type conversion
    4. Null handling
    5. Text normalisation
    6. Feature: metadata soup
    7. Deduplication by name (keep highest votes)
    8. Sampling guard (cap at 10 000 rows for performance)
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Column manifest ───────────────────────────────────────────────────────────
_REQUIRED_COLS: list[str] = [
    "name",
    "rate",
    "votes",
    "approx_cost(for two people)",
    "cuisines",
    "rest_type",
    "location",
    "listed_in(type)",
    "listed_in(city)",
]

_RENAME_MAP: dict[str, str] = {
    "approx_cost(for two people)": "cost",
    "listed_in(type)":             "category",
    "listed_in(city)":             "city",
}

_TEXT_COLS: list[str] = [
    "name", "cuisines", "rest_type", "location", "category", "city",
]

_MAX_ROWS: int = 10_000


# ── Private helpers ───────────────────────────────────────────────────────────

def _parse_rate(series: pd.Series) -> pd.Series:
    """
    Convert the raw 'rate' column to float.

    Handles values like '4.1/5', '4.1', 'NEW', '-', NaN.
    Non-numeric entries become NaN.

    Parameters
    ----------
    series : pd.Series
        Raw rate column (object dtype).

    Returns
    -------
    pd.Series
        Float series with non-parseable values as NaN.
    """
    cleaned = (
        series.astype(str)
              .str.strip()
              .str.replace(r"/5\s*$", "", regex=True)   # remove '/5' suffix
              .str.replace(r"(?i)^new$", "nan", regex=True)
              .str.replace(r"^-+$",     "nan", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_cost(series: pd.Series) -> pd.Series:
    """
    Convert the raw cost column to float.

    Strips thousands-separator commas before casting, e.g. '1,200' → 1200.0.

    Parameters
    ----------
    series : pd.Series
        Raw cost column (object dtype).

    Returns
    -------
    pd.Series
        Float series with non-parseable values as NaN.
    """
    cleaned = (
        series.astype(str)
              .str.strip()
              .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _normalise_text(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Apply text normalisation in-place on *cols*.

    Steps applied per column:
        - Cast to str, replace literal 'nan' strings with ''
        - Strip leading / trailing whitespace
        - Lowercase
        - Collapse runs of internal whitespace to a single space

    Parameters
    ----------
    df   : pd.DataFrame
        DataFrame to modify (already a copy).
    cols : list[str]
        Column names to normalise. Missing columns are silently skipped.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with normalised text columns.
    """
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = (
            df[col].astype(str)
                   .str.strip()
                   .replace("nan", "")               # literal 'nan' artefacts
                   .str.lower()
                   .str.replace(r"\s+", " ", regex=True)
                   .str.strip()
        )
    return df


def _build_soup(df: pd.DataFrame) -> pd.Series:
    """
    Construct a metadata 'soup' string for each restaurant row.

    Concatenates (space-separated): cuisines + rest_type + location + category.
    Removes commas and non-alphanumeric-or-space characters from the result.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with text-normalised columns.

    Returns
    -------
    pd.Series
        String series suitable for TF-IDF vectorisation.
    """
    parts = (
        df["cuisines"].fillna("") + " "
        + df["rest_type"].fillna("") + " "
        + df["location"].fillna("") + " "
        + df["category"].fillna("")
    )
    # Strip commas and extra punctuation; collapse whitespace
    soup = (
        parts.str.replace(",", " ", regex=False)
             .str.replace(r"[^a-z0-9 ]", " ", regex=True)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip()
    )
    return soup


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load the raw Zomato Bangalore CSV and return a fully cleaned DataFrame.

    The function does NOT mutate any intermediate object shared with the caller;
    it works on an explicit copy after loading.

    Parameters
    ----------
    filepath : str
        Path to ``zomato.csv`` (or any compatible CSV).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
        name, rate, votes, cost, cuisines, rest_type,
        location, category, city, soup

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If one or more required columns are absent from the CSV.

    Notes
    -----
    Step 8 caps the output at 10 000 rows (random_state=42) when the cleaned
    dataset exceeds that threshold, and prints a warning to stdout.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    # ── STEP 1 — LOAD ─────────────────────────────────────────────────────────
    raw: pd.DataFrame = pd.read_csv(
        path,
        encoding="utf-8",
        encoding_errors="replace",   # graceful handling of bad bytes
    )
    shape_before = raw.shape
    print(f"[load]  Raw shape: {shape_before}")

    # Work on an explicit copy so the original DataFrame is never mutated
    df: pd.DataFrame = raw.copy()
    df.drop_duplicates(inplace=True)
    print(f"[step1] After dropping exact duplicates: {df.shape}")

    # ── STEP 2 — COLUMN SELECTION & RENAME ────────────────────────────────────
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns not found in CSV: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[_REQUIRED_COLS].copy()
    df.rename(columns=_RENAME_MAP, inplace=True)
    print(f"[step2] Columns after selection & rename: {df.columns.tolist()}")

    # ── STEP 3 — TYPE CONVERSION ──────────────────────────────────────────────
    df["rate"]  = _parse_rate(df["rate"])
    df["cost"]  = _parse_cost(df["cost"])
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
    print(f"[step3] Types -> rate: {df['rate'].dtype}, "
          f"cost: {df['cost'].dtype}, votes: {df['votes'].dtype}")

    # ── STEP 4 — NULL HANDLING ────────────────────────────────────────────────
    rows_before_drop = len(df)
    df.dropna(subset=["name", "cuisines", "location"], inplace=True)
    print(f"[step4] Dropped {rows_before_drop - len(df)} rows with null "
          f"name / cuisines / location. Remaining: {len(df)}")

    rate_median = df["rate"].median()
    cost_median = df["cost"].median()
    df["rate"]     = df["rate"].fillna(rate_median)
    df["cost"]     = df["cost"].fillna(cost_median)
    df["rest_type"]= df["rest_type"].fillna("Unknown")
    print(f"[step4] rate median fill: {rate_median:.2f} | "
          f"cost median fill: {cost_median:.2f}")

    # ── STEP 5 — TEXT NORMALISATION ───────────────────────────────────────────
    df = _normalise_text(df, _TEXT_COLS)
    print("[step5] Text normalisation complete.")

    # ── STEP 6 — FEATURE: METADATA SOUP ──────────────────────────────────────
    df["soup"] = _build_soup(df)
    print("[step6] 'soup' column created.")

    # ── STEP 7 — DEDUPLICATION BY NAME ───────────────────────────────────────
    rows_before_dedup = len(df)
    df = (
        df.sort_values("votes", ascending=False)
          .drop_duplicates(subset=["name"], keep="first")
          .reset_index(drop=True)
    )
    print(f"[step7] Deduplicated by name (kept highest-votes row). "
          f"Removed {rows_before_dedup - len(df)} rows. Remaining: {len(df)}")

    # ── STEP 8 — SAMPLING GUARD ───────────────────────────────────────────────
    if len(df) > _MAX_ROWS:
        print(
            f"[step8] WARNING: Cleaned DataFrame has {len(df)} rows, "
            f"which exceeds the {_MAX_ROWS:,}-row performance cap. "
            f"Sampling {_MAX_ROWS:,} rows with random_state=42.",
            flush=True,
        )
        df = df.sample(n=_MAX_ROWS, random_state=42).reset_index(drop=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(
        f"\n{'-' * 50}\n"
        f"  Shape BEFORE cleaning : {shape_before}\n"
        f"  Shape AFTER  cleaning : {df.shape}\n"
        f"{'-' * 50}\n"
    )
    return df


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Zomato preprocessing pipeline."
    )
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "data" / "zomato.csv"),
        help="Path to raw zomato.csv (default: data/zomato.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the cleaned CSV.",
    )
    args = parser.parse_args()

    cleaned_df = load_and_clean(args.input)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(out_path, index=False)
        print(f"Cleaned data saved to: {out_path.resolve()}")
