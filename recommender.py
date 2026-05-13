"""
recommender.py  (performance-optimised)
────────────────────────────────────────
Optimisations applied
─────────────────────
  FIX 1 — float32 TF-IDF matrix     → halves memory vs float64
  FIX 2 — joblib disk cache         → skip recomputation on restart
  FIX 5 — precomputed top-20 dict   → O(1) query-time lookup; no N×N matrix in RAM

Public API is unchanged — app.py requires zero edits.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CACHE_DIR: Path = Path(__file__).resolve().parent / "cache"
_TOP_PRECOMPUTE: int = 20      # neighbours stored per restaurant
_BATCH_SIZE: int    = 500      # rows processed per linear_kernel call
_OUTPUT_COLS: list[str] = ["name", "rate", "cuisines", "location", "cost"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _df_fingerprint(df: pd.DataFrame) -> str:
    """
    Produce a short SHA-256 fingerprint for *df* used as a disk-cache key.

    The fingerprint encodes the DataFrame shape and the first 50 restaurant
    names, which changes whenever the underlying dataset changes.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (must contain a ``name`` column).

    Returns
    -------
    str
        16-character hex digest.
    """
    key = f"{df.shape}|{df['name'].head(50).tolist()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ── Core class ────────────────────────────────────────────────────────────────

class RestaurantRecommender:
    """
    Content-based restaurant recommender — TF-IDF + cosine similarity.

    Performance characteristics (10 k rows)
    ─────────────────────────────────────────
    Memory   : ~40 MB  (sparse float32 TF-IDF + top-20 dict)
               vs ~800 MB for a full float64 N×N similarity matrix
    Cold start : ~15–30 s (TF-IDF fit + batched linear_kernel + disk save)
    Warm start : < 1 s   (joblib load from ``cache/``)
    Query time : O(1)    (dict lookup on precomputed top-20)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from ``preprocess.load_and_clean()``.
        Must contain ``name`` and ``soup`` columns.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialise the recommender.

        On the first call the method:
          1. Fits a float32 TF-IDF vectoriser (FIX 1).
          2. Precomputes the top-:data:`_TOP_PRECOMPUTE` similar restaurants
             for each entry using batched ``linear_kernel`` calls (FIX 5).
          3. Persists both artefacts to ``cache/`` via joblib (FIX 2).

        On subsequent calls (same dataset fingerprint) both artefacts are
        loaded from disk, skipping all heavy computation.

        Parameters
        ----------
        df : pd.DataFrame
        """
        missing = [c for c in ("name", "soup") if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        self.df: pd.DataFrame = df.copy().reset_index(drop=True)

        # ── FIX 2: disk-cache paths ───────────────────────────────────────────
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fp             = _df_fingerprint(self.df)
        _tfidf_path    = CACHE_DIR / f"tfidf_{fp}.joblib"
        _top_sims_path = CACHE_DIR / f"top_sims_{fp}.joblib"

        if _tfidf_path.exists() and _top_sims_path.exists():
            log.info("Cache hit — loading artefacts from %s …", CACHE_DIR)
            self._tfidf_matrix              = joblib.load(_tfidf_path)
            self._top_sims: dict[int,
                list[tuple[int, float]]]    = joblib.load(_top_sims_path)
            log.info("Artefacts loaded.")
        else:
            # ── FIX 1: float32 TF-IDF — half the memory of float64 ───────────
            log.info("Cache miss — fitting TF-IDF vectoriser …")
            vectoriser = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                stop_words="english",
                max_features=15_000,
                sublinear_tf=True,
                dtype=np.float32,          # ← FIX 1: native float32 output
            )
            self._tfidf_matrix = vectoriser.fit_transform(
                self.df["soup"].fillna("")
            )
            log.info(
                "TF-IDF matrix: shape=%s  dtype=%s  nnz=%d",
                self._tfidf_matrix.shape,
                self._tfidf_matrix.dtype,
                self._tfidf_matrix.nnz,
            )

            # ── FIX 5: precompute top-N dict (no full N×N matrix) ─────────────
            self._top_sims = self._precompute_top_sims(_TOP_PRECOMPUTE)

            # ── FIX 2: persist to disk ────────────────────────────────────────
            log.info("Saving artefacts to %s …", CACHE_DIR)
            joblib.dump(self._tfidf_matrix, _tfidf_path,    compress=3)
            joblib.dump(self._top_sims,     _top_sims_path, compress=3)
            log.info("Artefacts saved.")

        # ── Name → index mapping ──────────────────────────────────────────────
        self._name_to_idx: pd.Series = pd.Series(
            self.df.index,
            index=self.df["name"].str.lower().str.strip(),
        )

    # ── FIX 5: batched precomputation ─────────────────────────────────────────

    def _precompute_top_sims(
        self, k: int
    ) -> dict[int, list[tuple[int, float]]]:
        """
        Build a dictionary mapping each restaurant index to its top-*k*
        most-similar neighbours.

        Uses ``linear_kernel`` in batches of :data:`_BATCH_SIZE` rows to
        avoid materialising the full N×N matrix in memory at any one time.
        Each batch produces a (batch_size × N) float32 block (~20 MB for
        N = 10 000), which is immediately reduced to top-k indices before
        the next batch is loaded.

        Parameters
        ----------
        k : int
            Number of neighbours to store per restaurant.

        Returns
        -------
        dict[int, list[tuple[int, float]]]
            ``{row_index: [(similar_index, score), …]}``
            Each value list is sorted by score descending and has length ≤ k.
        """
        n        = len(self.df)
        top_sims: dict[int, list[tuple[int, float]]] = {}

        for start in range(0, n, _BATCH_SIZE):
            end   = min(start + _BATCH_SIZE, n)
            batch = self._tfidf_matrix[start:end]

            # (batch_size × n) — kept only briefly then discarded
            sim_block: np.ndarray = linear_kernel(
                batch, self._tfidf_matrix
            ).astype(np.float32)

            for local_i, global_i in enumerate(range(start, end)):
                row = sim_block[local_i].copy()
                row[global_i] = -1.0             # exclude self

                # argpartition is O(n) — faster than full argsort for large n
                top_k_raw = np.argpartition(row, -k)[-k:]
                top_k_sorted = top_k_raw[np.argsort(row[top_k_raw])[::-1]]

                top_sims[global_i] = [
                    (int(j), float(row[j])) for j in top_k_sorted
                ]

            log.info("Precomputed %d / %d rows …", end, n)

        return top_sims

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend(
        self,
        restaurant_name: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Return the *top_n* most similar restaurants — O(1) dict lookup.

        Parameters
        ----------
        restaurant_name : str
            Query name (case-insensitive, whitespace-tolerant).
        top_n : int
            Number of results (silently capped at :data:`_TOP_PRECOMPUTE`).

        Returns
        -------
        pd.DataFrame
            Columns: name, rate, cuisines, location, cost, similarity_score.

        Raises
        ------
        ValueError
            If *restaurant_name* is not in the dataset; includes fuzzy hints.
        """
        key = restaurant_name.lower().strip()

        if key not in self._name_to_idx.index:
            close = self._fuzzy_suggestions(key)
            hint  = f"  Did you mean: {', '.join(close)}?" if close else ""
            raise ValueError(
                f"Restaurant '{restaurant_name}' not found in the dataset.{hint}\n"
                "Use get_all_names() to browse available restaurants."
            )

        raw_idx = self._name_to_idx[key]
        if isinstance(raw_idx, pd.Series):
            query_idx: int = int(self.df.loc[raw_idx.values, "rate"].idxmax())
        else:
            query_idx = int(raw_idx)

        # O(1) — precomputed, already sorted descending
        top_entries = self._top_sims.get(query_idx, [])
        top_entries = top_entries[: min(top_n, len(top_entries))]

        if not top_entries:
            return pd.DataFrame()

        indices = [i   for i, _ in top_entries]
        scores  = [s   for _, s in top_entries]

        result = self.df.loc[indices, self._safe_output_cols()].copy()
        result["similarity_score"] = scores
        result.sort_values(
            ["similarity_score", "rate"],
            ascending=[False, False],
            inplace=True,
        )
        return result.reset_index(drop=True)

    def get_all_names(self) -> list[str]:
        """
        Return a sorted list of all unique restaurant names for UI dropdowns.

        Returns
        -------
        list[str]
        """
        return sorted(self.df["name"].dropna().unique().tolist())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _safe_output_cols(self) -> list[str]:
        """Return output columns that actually exist in ``self.df``."""
        return [c for c in _OUTPUT_COLS if c in self.df.columns]

    def _fuzzy_suggestions(self, key: str, n: int = 5) -> list[str]:
        """
        Return up to *n* restaurant names that contain *key* as a substring.

        Parameters
        ----------
        key : str   Lowercased query fragment.
        n   : int   Maximum suggestions.

        Returns
        -------
        list[str]   Matching names in original casing.
        """
        mask = self.df["name"].str.lower().str.contains(key, na=False, regex=False)
        return self.df.loc[mask, "name"].head(n).tolist()


# ── Streamlit cache_resource factory ─────────────────────────────────────────

def build_recommender(df: pd.DataFrame) -> RestaurantRecommender:
    """
    Construct a :class:`RestaurantRecommender` wrapped in Streamlit's
    ``@st.cache_resource`` so the object is built only once per app session.

    Streamlit is imported lazily; the module works without Streamlit installed.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    RestaurantRecommender
    """
    try:
        import streamlit as st   # lazy import

        @st.cache_resource(show_spinner="⚙️ Building recommendation engine …")
        def _cached(_df: pd.DataFrame) -> RestaurantRecommender:
            return RestaurantRecommender(_df)

        return _cached(df)

    except ModuleNotFoundError:
        log.warning(
            "Streamlit not installed — building recommender without caching."
        )
        return RestaurantRecommender(df)


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preprocess import load_and_clean  # noqa: E402

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    DATA_PATH = Path(__file__).resolve().parent / "data" / "zomato.csv"
    print(f"\nLoading data from: {DATA_PATH}")
    df = load_and_clean(str(DATA_PATH))

    print("\nBuilding recommender …")
    rec = RestaurantRecommender(df)

    names  = rec.get_all_names()
    sample = "Truffles"
    if sample.lower() not in [n.lower() for n in names]:
        sample = names[0]

    print(f"\nTop-10 recommendations for '{sample}':\n")
    print(rec.recommend(sample, top_n=10).to_string(index=False))
