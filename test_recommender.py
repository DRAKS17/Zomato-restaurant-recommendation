"""
Unit tests for the Zomato Restaurant Recommender modules.

Validates the data-cleaning pipeline (preprocess.py) and the content-based
recommendation engine (recommender.py) using Python's built-in unittest.

Run with:
    python test_recommender.py
or:
    python -m unittest test_recommender -v
"""

from __future__ import annotations

import io
import sys
import unittest
import unittest.mock as mock
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal column set required by load_and_clean() to pass validation
_RAW_COLS = [
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


def _make_raw_df(n: int = 30) -> pd.DataFrame:
    """
    Build a synthetic 'raw' DataFrame that mimics the zomato.csv structure.

    All required columns are present; rate intentionally uses the '/5' suffix
    to exercise the _parse_rate() helper.
    """
    rng = np.random.default_rng(0)
    cuisines_pool = [
        "North Indian", "South Indian", "Chinese", "Italian",
        "Mexican", "Continental", "Fast Food",
    ]
    rest_types = ["Casual Dining", "Quick Bites", "Cafe", "Delivery"]
    locations  = ["Indiranagar", "Koramangala", "Whitefield", "HSR Layout"]
    types_     = ["Buffet", "Dine-out", "Delivery", "Cafes"]
    cities_    = ["Bangalore"]

    return pd.DataFrame({
        "name":                        [f"Restaurant_{i}" for i in range(n)],
        "rate":                        [f"{rng.uniform(1.5, 5.0):.1f}/5" for _ in range(n)],
        "votes":                       rng.integers(10, 5000, size=n).tolist(),
        "approx_cost(for two people)": [str(rng.integers(100, 2000)) for _ in range(n)],
        "cuisines":                    [rng.choice(cuisines_pool) for _ in range(n)],
        "rest_type":                   [rng.choice(rest_types)    for _ in range(n)],
        "location":                    [rng.choice(locations)     for _ in range(n)],
        "listed_in(type)":             [rng.choice(types_)        for _ in range(n)],
        "listed_in(city)":             [rng.choice(cities_)       for _ in range(n)],
    })


def _run_pipeline_on(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Invoke load_and_clean() with the CSV I/O replaced by *raw_df*.

    Patches both pathlib.Path.exists (returns True) and pandas.read_csv
    (returns a copy of *raw_df*) so no real file is needed on disk.
    """
    from preprocess import load_and_clean

    with (
        mock.patch("pathlib.Path.exists", return_value=True),
        mock.patch("pandas.read_csv",     return_value=raw_df.copy()),
    ):
        return load_and_clean("fake/path/zomato.csv")


# ---------------------------------------------------------------------------
# TestPreprocessor
# ---------------------------------------------------------------------------

class TestPreprocessor(unittest.TestCase):
    """Tests for preprocess.load_and_clean()."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the pipeline once; share the result across all test methods."""
        raw = _make_raw_df(n=30)
        cls.raw_df   = raw
        cls.clean_df = _run_pipeline_on(raw)

    # ── rate column ─────────────────────────────────────────────────────────

    def test_clean_rate_column(self) -> None:
        """
        Verify that the '/5' suffix is removed from the raw rate strings and
        the resulting column is cast to float (or float64 equivalent).

        The raw synthetic data uses values like '3.7/5'; after cleaning every
        non-null value must be a Python float within the range [0, 5].
        """
        rate_col = self.clean_df["rate"]
        self.assertTrue(
            pd.api.types.is_float_dtype(rate_col),
            msg=f"Expected float dtype, got {rate_col.dtype}",
        )
        non_null = rate_col.dropna()
        self.assertTrue(
            (non_null >= 0).all() and (non_null <= 5).all(),
            msg="rate values should be in [0, 5] after cleaning",
        )

    # ── null handling ────────────────────────────────────────────────────────

    def test_null_handling(self) -> None:
        """
        Verify that name, cuisines, and location contain no null values after
        the cleaning pipeline runs.

        Steps 4–5 of load_and_clean() drop rows with nulls in those critical
        columns and fill remaining numeric/text nulls.
        """
        for col in ("name", "cuisines", "location"):
            null_count = self.clean_df[col].isna().sum()
            self.assertEqual(
                null_count, 0,
                msg=f"Column '{col}' should have 0 nulls; found {null_count}",
            )

    # ── soup column ──────────────────────────────────────────────────────────

    def test_soup_column_exists(self) -> None:
        """
        Verify that Step 6 creates a 'soup' column containing non-empty strings
        suitable for TF-IDF vectorisation.

        Every row in the cleaned DataFrame must have a non-null, non-empty soup.
        """
        self.assertIn(
            "soup", self.clean_df.columns,
            msg="'soup' column was not created by the pipeline",
        )
        null_soup = self.clean_df["soup"].isna().sum()
        self.assertEqual(null_soup, 0, msg="soup column must not contain nulls")

        empty_soup = (self.clean_df["soup"].str.strip() == "").sum()
        self.assertEqual(
            empty_soup, 0,
            msg="soup column must not contain empty strings",
        )

    # ── deduplication ────────────────────────────────────────────────────────

    def test_deduplication(self) -> None:
        """
        Verify that Step 7 removes duplicate restaurant names, keeping the row
        with the highest vote count.

        After cleaning, each name in the output DataFrame must appear exactly once.
        """
        names = self.clean_df["name"]
        self.assertEqual(
            names.nunique(), len(names),
            msg="Duplicate restaurant names found after deduplication step",
        )

    def test_deduplication_keeps_highest_votes(self) -> None:
        """
        Verify that when duplicate names are present in the raw data, the row
        with the most votes is retained after Step 7.
        """
        # Build a raw DataFrame where 'Dup Restaurant' appears twice
        dup_raw = _make_raw_df(n=10)
        dup_raw = pd.concat([
            dup_raw,
            pd.DataFrame([{
                "name": "Dup Restaurant",
                "rate": "4.5/5",
                "votes": 999,
                "approx_cost(for two people)": "500",
                "cuisines": "North Indian",
                "rest_type": "Casual Dining",
                "location": "Koramangala",
                "listed_in(type)": "Dine-out",
                "listed_in(city)": "Bangalore",
            }, {
                "name": "Dup Restaurant",
                "rate": "3.0/5",
                "votes": 10,
                "approx_cost(for two people)": "500",
                "cuisines": "South Indian",
                "rest_type": "Cafe",
                "location": "HSR Layout",
                "listed_in(type)": "Cafes",
                "listed_in(city)": "Bangalore",
            }]),
        ], ignore_index=True)

        cleaned = _run_pipeline_on(dup_raw)

        dup_rows = cleaned[cleaned["name"] == "dup restaurant"]  # text-normalised
        self.assertEqual(len(dup_rows), 1, msg="Duplicate name not removed")
        # The retained row should have the higher votes value (999)
        self.assertEqual(
            int(dup_rows["votes"].iloc[0]), 999,
            msg="Deduplication should keep the highest-votes row",
        )

    # ── sampling guard ───────────────────────────────────────────────────────

    def test_sampling_guard(self) -> None:
        """
        Verify that when the cleaned DataFrame exceeds 10 000 rows, Step 8
        samples it down to exactly 10 000 rows (random_state=42).

        A synthetic DataFrame with 10 050 unique restaurant names is fed through
        the full pipeline; the output must have exactly 10 000 rows.
        """
        big_raw = _make_raw_df(n=10_050)
        # Ensure unique names so deduplication doesn't interfere
        big_raw["name"] = [f"UniqueRestaurant_{i}" for i in range(10_050)]

        cleaned = _run_pipeline_on(big_raw)

        self.assertEqual(
            len(cleaned), 10_000,
            msg=(
                f"Sampling guard should cap output at 10 000 rows; "
                f"got {len(cleaned)}"
            ),
        )


# ---------------------------------------------------------------------------
# Synthetic DataFrame for recommender tests
# ---------------------------------------------------------------------------

def _make_recommender_df(n: int = 20) -> pd.DataFrame:
    """
    Build a small but realistic DataFrame that RestaurantRecommender can consume.

    Columns required: name, rate, votes, cuisines, rest_type, location,
    category, city, cost, soup.
    Includes at least 5 distinct cuisine values.
    """
    rng = np.random.default_rng(1)
    cuisines_pool = [
        "North Indian", "South Indian", "Chinese", "Italian", "Mexican",
    ]
    rest_types = ["Casual Dining", "Quick Bites", "Cafe"]
    locations  = ["Koramangala", "Indiranagar", "Whitefield"]
    categories = ["Dine-out", "Delivery", "Cafes"]
    cities_    = ["Bangalore"]

    cuisine_col  = [cuisines_pool[i % len(cuisines_pool)] for i in range(n)]
    rest_type_col = [rng.choice(rest_types) for _ in range(n)]
    location_col  = [rng.choice(locations)  for _ in range(n)]
    category_col  = [rng.choice(categories) for _ in range(n)]

    # soup mirrors what _build_soup() would produce
    soup_col = [
        f"{c} {r} {loc} {cat}".lower().replace(",", " ")
        for c, r, loc, cat in zip(cuisine_col, rest_type_col, location_col, category_col)
    ]

    return pd.DataFrame({
        "name":     [f"Place_{i:02d}" for i in range(n)],
        "rate":     np.round(rng.uniform(2.0, 5.0, size=n), 1),
        "votes":    rng.integers(50, 3000, size=n).tolist(),
        "cuisines": cuisine_col,
        "rest_type":rest_type_col,
        "location": location_col,
        "category": category_col,
        "city":     [rng.choice(cities_) for _ in range(n)],
        "cost":     rng.integers(100, 2000, size=n).astype(float).tolist(),
        "soup":     soup_col,
    })


# ---------------------------------------------------------------------------
# TestRecommender
# ---------------------------------------------------------------------------

class TestRecommender(unittest.TestCase):
    """Tests for recommender.RestaurantRecommender."""

    def setUp(self) -> None:
        """
        Construct a RestaurantRecommender from 20 synthetic rows.

        The synthetic DataFrame has realistic rate (2.0–5.0), cost (100–2000),
        and at least 5 distinct cuisine values so the TF-IDF features are
        meaningful.  setUp() runs before every individual test method.
        """
        from recommender import RestaurantRecommender

        self.df  = _make_recommender_df(n=20)
        # Bypass disk cache for unit tests by pointing CACHE_DIR to a temp path
        import recommender as rec_module
        import tempfile, os
        self._tmp_cache = tempfile.mkdtemp()
        self._orig_cache_dir = rec_module.CACHE_DIR
        rec_module.CACHE_DIR = Path(self._tmp_cache)

        self.rec = RestaurantRecommender(self.df)

    def tearDown(self) -> None:
        """Restore the original CACHE_DIR and clean up temp files."""
        import recommender as rec_module
        import shutil
        rec_module.CACHE_DIR = self._orig_cache_dir
        shutil.rmtree(self._tmp_cache, ignore_errors=True)

    # ── return type ──────────────────────────────────────────────────────────

    def test_recommend_returns_dataframe(self) -> None:
        """
        Verify that recommend() always returns a pandas DataFrame, not a list,
        dict, or any other container type.
        """
        result = self.rec.recommend("Place_00", top_n=3)
        self.assertIsInstance(
            result, pd.DataFrame,
            msg=f"recommend() should return pd.DataFrame, got {type(result)}",
        )

    # ── row count ────────────────────────────────────────────────────────────

    def test_recommend_correct_count(self) -> None:
        """
        Verify that recommend() returns exactly top_n rows when at least top_n
        neighbours exist in the dataset.

        With 20 rows and top_n=5 there are always enough neighbours, so the
        result must contain exactly 5 rows.
        """
        top_n  = 5
        result = self.rec.recommend("Place_00", top_n=top_n)
        self.assertEqual(
            len(result), top_n,
            msg=f"Expected {top_n} recommendations, got {len(result)}",
        )

    # ── input exclusion ──────────────────────────────────────────────────────

    def test_recommend_excludes_input(self) -> None:
        """
        Verify that the queried restaurant does not appear among its own
        recommendations.

        The similarity matrix should exclude the self-similarity score so the
        input restaurant is never returned in the results.
        """
        query  = "Place_01"
        result = self.rec.recommend(query, top_n=5)
        returned_names = result["name"].str.lower().tolist()
        self.assertNotIn(
            query.lower(), returned_names,
            msg=f"Input restaurant '{query}' should not appear in its own results",
        )

    # ── invalid name ─────────────────────────────────────────────────────────

    def test_recommend_invalid_name_raises(self) -> None:
        """
        Verify that recommend() raises ValueError when the restaurant name is
        not found in the dataset.

        Passing a completely unknown name must trigger the lookup failure path
        in RestaurantRecommender.recommend(), not a KeyError or AttributeError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.rec.recommend("This Restaurant Does Not Exist XYZ")
        self.assertIn(
            "not found", str(ctx.exception).lower(),
            msg="ValueError message should mention 'not found'",
        )

    # ── get_all_names ────────────────────────────────────────────────────────

    def test_get_all_names_sorted(self) -> None:
        """
        Verify that get_all_names() returns a list of strings sorted in
        ascending alphabetical order.

        The UI selectbox depends on this sorted order for a good user experience.
        """
        names = self.rec.get_all_names()
        self.assertIsInstance(names, list, msg="get_all_names() should return a list")
        self.assertTrue(
            all(isinstance(n, str) for n in names),
            msg="get_all_names() should return strings only",
        )
        self.assertEqual(
            names, sorted(names),
            msg="get_all_names() should be sorted alphabetically",
        )

    # ── case insensitivity ───────────────────────────────────────────────────

    def test_recommend_case_insensitive(self) -> None:
        """
        Verify that recommend() returns the same results regardless of the
        capitalisation of the input restaurant name.

        Querying 'place_00', 'PLACE_00', and 'Place_00' must all return
        identical recommendation DataFrames.
        """
        result_lower = self.rec.recommend("place_00", top_n=3)
        result_upper = self.rec.recommend("PLACE_00", top_n=3)
        result_mixed = self.rec.recommend("Place_00", top_n=3)

        pd.testing.assert_frame_equal(
            result_lower.reset_index(drop=True),
            result_upper.reset_index(drop=True),
            check_like=False,
            obj="lower vs upper",
        )
        pd.testing.assert_frame_equal(
            result_lower.reset_index(drop=True),
            result_mixed.reset_index(drop=True),
            check_like=False,
            obj="lower vs mixed",
        )

    # ── similarity_score column ──────────────────────────────────────────────

    def test_recommend_has_similarity_score(self) -> None:
        """
        Verify that the result DataFrame contains a 'similarity_score' column
        with float values in the range [0, 1].

        The similarity scores drive the sort order in the UI.
        """
        result = self.rec.recommend("Place_02", top_n=4)
        self.assertIn(
            "similarity_score", result.columns,
            msg="Result DataFrame must contain a 'similarity_score' column",
        )
        scores = result["similarity_score"]
        self.assertTrue(
            pd.api.types.is_float_dtype(scores),
            msg=f"similarity_score should be float dtype, got {scores.dtype}",
        )
        self.assertTrue(
            (scores >= 0).all() and (scores <= 1.001).all(),
            msg="similarity_score values should be in [0, 1]",
        )

    # ── top_n cap ────────────────────────────────────────────────────────────

    def test_recommend_top_n_capped_at_precomputed(self) -> None:
        """
        Verify that requesting more recommendations than were precomputed
        (default _TOP_PRECOMPUTE = 20) does not raise an error; the result is
        silently capped at the number of available neighbours.
        """
        import recommender as rec_module
        precomputed = rec_module._TOP_PRECOMPUTE

        # Requesting 1000 should not crash; result size <= precomputed
        result = self.rec.recommend("Place_03", top_n=1000)
        self.assertLessEqual(
            len(result), precomputed,
            msg=(
                f"Requesting more than _TOP_PRECOMPUTE={precomputed} neighbours "
                f"should silently cap, not raise"
            ),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Suppress noisy stdout from preprocess.py print statements
    with mock.patch("sys.stdout", new_callable=io.StringIO):
        pass  # just ensuring the import works
    unittest.main(verbosity=2)
