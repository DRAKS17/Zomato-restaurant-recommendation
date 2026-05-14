"""app.py — Zomato Restaurant Recommender  (UI v4 – Professional, encoding-safe)"""
from __future__ import annotations
import os, random
from pathlib import Path
import pandas as pd
import streamlit as st
from preprocess import load_and_clean
from recommender import build_recommender

# ── constants ─────────────────────────────────────────────────────────────────
DATA_PATH = "data/zomato.csv"
_PARQUET  = Path("cache/zomato_clean.parquet")
_HIST_MAX = 6

CUISINE_EMOJI = {
    "north indian":"🍛","south indian":"🥘","chinese":"🥡","italian":"🍕",
    "continental":"🍽️","fast food":"🍔","bakery":"🥐","cafe":"☕",
    "desserts":"🍮","ice cream":"🍦","pizza":"🍕","biryani":"🍚",
    "seafood":"🦐","mughlai":"🍖","mexican":"🌮","thai":"🍜",
    "japanese":"🍣","american":"🌭","street food":"🥙","healthy food":"🥗",
    "sandwich":"🥪","burger":"🍔","momos":"🥟","sushi":"🍣",
}

# Accent colours cycling through cards
ACCENTS = ["#E53935","#8E24AA","#1E88E5","#00897B","#F4511E",
           "#6D4C41","#039BE5","#43A047","#FB8C00","#3949AB"]

import re as _re

def _sanitize(s: str) -> str:
    """Strip mojibake / encoding garbage – keep printable ASCII + common Unicode."""
    if not s or s == "—": return s
    # Remove sequences of non-latin replacement chars (Ã, Â, Å etc.)
    s = _re.sub(r'[\xc0-\xff]{2,}', '', s)   # multi-byte mojibake runs
    s = _re.sub(r'[^\x00-\x7F\u0900-\u097F\u0020-\u007E]+', ' ', s)  # keep ASCII + Devanagari
    s = _re.sub(r'\s{2,}', ' ', s).strip()
    return s or "—"

def c_emoji(s: str) -> str:
    s = (s or "").lower()
    for k, v in CUISINE_EMOJI.items():
        if k in s: return v
    return "🍴"

def fmt_stars(r) -> str:
    try:
        r = float(r); f = int(r); h = 1 if r - f >= 0.25 else 0
        return "★" * f + ("½" if h else "") + "☆" * (5 - f - h)
    except: return "—"

def match_color(pct: float) -> str:
    if pct >= 70: return "#4CAF50"
    if pct >= 45: return "#FF9800"
    return "#F44336"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zomato Restaurant Recommender",
    page_icon="🍽️", layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Scope Inter to Streamlit content containers only — never touch span/icon elements */
.stApp, .stMarkdown, .stText, .stAlert, .stMetric,
.stButton > button, .stSelectbox label, .stSlider label,
.stTabs [data-baseweb="tab"], .stDataFrame,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMetricValue"],
[data-testid="stSidebar"] [data-testid="stMetricLabel"],
.streamlit-expanderHeader p,
.block-container h1, .block-container h2, .block-container h3,
.block-container p, .block-container label {
    font-family: 'Inter', sans-serif !important;
}

/* ── Base ── */
.stApp { background: #111318 !important; }
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1280px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D0F14 !important;
    border-right: 1px solid #1F2330 !important;
}
[data-testid="stSidebar"] * { color: #C9CDD8 !important; }
[data-testid="stSidebar"] h2 { color: #FFFFFF !important; font-size: 1.1rem !important; font-weight: 700 !important; }
[data-testid="stSidebar"] h3 { color: #9CA3AF !important; font-size: 0.78rem !important;
    text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600 !important; }
[data-testid="stSidebar"] hr { border-color: #1F2330 !important; }
[data-testid="stSidebar"] .stMetric { background: #161A24 !important; border-radius: 10px !important; padding: 0.6rem !important; }
[data-testid="stSidebar"] [data-testid="metric-container"] { background: #161A24; border-radius: 10px; padding: 0.7rem 0.8rem; border: 1px solid #1F2330; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700 !important; color: #FFFFFF !important; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #6B7280 !important; text-transform: uppercase; }

/* ── Buttons ── */
.stButton > button {
    background: #E53935 !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.9rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: background 0.2s, transform 0.15s !important;
    box-shadow: 0 2px 8px rgba(229,57,53,0.3) !important;
}
.stButton > button:hover { background: #C62828 !important; transform: translateY(-1px) !important; }

/* ── Selectbox ── */
div[data-baseweb="select"] > div {
    background: #1A1D27 !important;
    border: 1px solid #2A2F3E !important;
    border-radius: 8px !important; color: #E5E7EB !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #E53935 !important;
    box-shadow: 0 0 0 2px rgba(229,57,53,0.15) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161A24 !important; border-radius: 10px !important;
    padding: 4px !important; gap: 6px !important; border: 1px solid #1F2330 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important; font-weight: 500 !important;
    color: #6B7280 !important; font-size: 0.88rem !important;
    padding: 0.4rem 1.1rem !important; min-width: 90px !important;
    text-align: center !important;
}
.stTabs [aria-selected="true"] {
    background: #E53935 !important; color: #fff !important; font-weight: 700 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #161A24 !important; border-radius: 8px !important;
    font-weight: 600 !important; color: #9CA3AF !important; font-size: 0.88rem !important;
    border: 1px solid #1F2330 !important;
}
details[open] .streamlit-expanderHeader { border-radius: 8px 8px 0 0 !important; }
.streamlit-expanderContent {
    background: #161A24 !important; border: 1px solid #1F2330 !important;
    border-top: none !important; border-radius: 0 0 8px 8px !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }

/* ── Divider ── */
hr { border-color: #1F2330 !important; }

/* ── Input label ── */
label { font-size: 0.8rem !important; color: #6B7280 !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)

# ── data & model ──────────────────────────────────────────────────────────────
def _parquet_fresh(csv: str, pq: Path) -> bool:
    if not pq.exists(): return False
    try: return pq.stat().st_mtime >= os.path.getmtime(csv)
    except: return False

@st.cache_data(ttl=3600, show_spinner="Loading & cleaning dataset…")
def get_data(path: str) -> pd.DataFrame:
    _PARQUET.parent.mkdir(parents=True, exist_ok=True)
    if _parquet_fresh(path, _PARQUET):
        return pd.read_parquet(_PARQUET)
    df = load_and_clean(path)
    df.to_parquet(_PARQUET, index=False)
    return df

df = get_data(DATA_PATH)
if "rec" not in st.session_state:
    with st.spinner("Building recommendation engine…"):
        st.session_state["rec"] = build_recommender(df)
rec       = st.session_state["rec"]
all_names = rec.get_all_names()

# session defaults
_defaults = {"toasted": False, "history": [], "selected": all_names[0],
             "results": None, "last_q": ""}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state["toasted"]:
    st.toast(f"Ready — {len(all_names):,} restaurants loaded", icon="✅")
    st.session_state["toasted"] = True

# precompute sidebar stats once
n_restaurants = len(df)
n_cuisines    = int(df["cuisines"].dropna().str.split(",").explode().str.strip().nunique())
n_locations   = int(df["location"].nunique())

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:0.2rem 0 0.8rem">
        <div style="background:#E53935;border-radius:8px;width:32px;height:32px;
            display:flex;align-items:center;justify-content:center;font-size:1.1rem">🍽️</div>
        <div>
            <div style="font-size:1rem;font-weight:700;color:#fff">Zomato Recommender</div>
            <div style="font-size:0.72rem;color:#6B7280">Bangalore · AI-powered</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Dataset stats — plain text labels (emojis in labels cause Streamlit 1.57 icon parsing bugs)
    st.markdown("### Dataset")
    st.metric("Restaurants", f"{n_restaurants:,}")
    st.metric("Cuisines",    str(n_cuisines))
    st.metric("Areas",       str(n_locations))
    st.divider()

    # Filters
    st.markdown("### Filters")
    top_n      = st.slider("Results count", 1, 10, 5)
    min_rating = st.slider("Min rating (stars)", 0.0, 5.0, 0.0, 0.5,
                           format="%.1f",
                           help="Only show results at or above this rating")
    st.divider()

    # Recent searches
    if st.session_state["history"]:
        st.markdown("### Recent Searches")
        for name in reversed(st.session_state["history"]):
            label = name.title()
            if len(label) > 22: label = label[:21] + "…"
            if st.button(label, key=f"hist_{name}", use_container_width=True):
                st.session_state["selected"] = name
                st.rerun()
        if st.button("Clear history", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()
        st.divider()

    # How it works
    with st.expander("How it works"):
        st.markdown("""
**Content-Based Filtering**

1. Each restaurant's cuisine, category & location are combined into a *text soup*
2. **TF-IDF** converts the soup into weighted numeric vectors
3. **Cosine similarity** ranks every restaurant by closeness to your pick
4. Top-20 neighbours are precomputed at startup → **< 50 ms per query**
        """)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.3rem">
        <span style="font-size:1.8rem">🍽️</span>
        <h1 style="font-size:1.9rem;font-weight:800;color:#FFFFFF;margin:0;line-height:1">
            Restaurant Recommender</h1>
    </div>
    <p style="color:#6B7280;font-size:0.95rem;margin:0;padding-left:52px">
        Discover Bangalore restaurants similar to one you already love</p>
</div>
""", unsafe_allow_html=True)

# divider line accent
st.markdown('<div style="height:3px;background:linear-gradient(90deg,#E53935,#FF7043,transparent);'
            'border-radius:2px;margin-bottom:1.5rem"></div>', unsafe_allow_html=True)

# ── search panel ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#161A24;border:1px solid #1F2330;border-radius:12px;
    padding:1.2rem 1.4rem;margin-bottom:1.4rem">
    <div style="font-size:0.78rem;font-weight:600;color:#6B7280;
        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem">
        Choose a restaurant</div>
""", unsafe_allow_html=True)

sc1, sc2, sc3 = st.columns([6, 2, 1], gap="small")
with sc1:
    try:    idx = all_names.index(st.session_state["selected"])
    except: idx = 0
    sel = st.selectbox("restaurant", all_names, index=idx,
                       label_visibility="collapsed")
    st.session_state["selected"] = sel

with sc2:
    go = st.button("Find Similar", use_container_width=True)

with sc3:
    if st.button("🎲", help="Pick a random restaurant", use_container_width=True):
        st.session_state["selected"] = random.choice(all_names)
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ── run recommendation ────────────────────────────────────────────────────────
if go:
    try:
        res = rec.recommend(sel, top_n=top_n)
        if min_rating > 0 and "rate" in res.columns:
            res = res[res["rate"] >= min_rating]
        st.session_state["results"] = res
        st.session_state["last_q"]  = sel
        hist = st.session_state["history"]
        if sel in hist: hist.remove(sel)
        hist.append(sel)
        if len(hist) > _HIST_MAX: hist.pop(0)
    except ValueError as e:
        st.error(str(e)); st.session_state["results"] = None

results = st.session_state["results"]
last_q  = st.session_state["last_q"]

# ── results ───────────────────────────────────────────────────────────────────
if results is not None and not results.empty:

    avg_score = float(results["similarity_score"].mean() * 100) \
                if "similarity_score" in results else 0

    # summary bar
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:1.2rem">
        <div style="background:#E53935;color:#fff;font-size:0.82rem;font-weight:700;
            border-radius:6px;padding:0.3rem 0.9rem">
            {len(results)} results</div>
        <div style="color:#E5E7EB;font-size:0.9rem;font-weight:600">
            Similar to &ldquo;{last_q.title()}&rdquo;</div>
        <div style="margin-left:auto;color:#6B7280;font-size:0.82rem">
            Avg match score: <span style="color:#fff;font-weight:600">{avg_score:.0f}%</span>
            &nbsp;·&nbsp; Min rating filter:
            <span style="color:#fff;font-weight:600">{min_rating:.1f}★</span></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["  Cards  ", "  Table  ", "  Insights  "])

    # ── TAB 1 — Cards ──────────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2, gap="medium")
        for i, (_, row) in enumerate(results.iterrows()):
            name    = _sanitize(str(row.get("name",     "")).title()) or "—"
            cuisine = _sanitize(str(row.get("cuisines", "")).title()) or "—"
            loc     = _sanitize(str(row.get("location", "")).title()) or "—"
            score   = float(row.get("similarity_score", 0))
            pct     = score * 100
            bc      = match_color(pct)
            emoji   = c_emoji(str(row.get("cuisines", "")))
            accent  = ACCENTS[i % len(ACCENTS)]
            try:    rate_s = f"{float(row['rate']):.1f}"
            except: rate_s = "—"
            try:    cost_s = f"₹{int(float(row['cost'])):,}"
            except: cost_s = "—"

            card = f"""
<div style="background:#161A24;border:1px solid #1F2330;border-radius:12px;
    padding:1.1rem 1.2rem;margin-bottom:0.9rem;
    border-left:3px solid {accent};position:relative">

  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px">
    <div style="display:flex;align-items:center;gap:10px;min-width:0;flex:1">
      <div style="background:#1F2330;border-radius:8px;width:38px;height:38px;flex-shrink:0;
          display:flex;align-items:center;justify-content:center;font-size:1.2rem">{emoji}</div>
      <div style="min-width:0">
        <div style="font-size:0.97rem;font-weight:700;color:#F9FAFB;
            overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
          <span style="color:{accent};font-size:0.75rem;font-weight:600;margin-right:5px">#{i+1}</span>
          {name}</div>
        <div style="font-size:0.77rem;color:#6B7280;margin-top:2px">📍 {loc}</div>
      </div>
    </div>
    <div style="background:#1F2330;border-radius:6px;padding:0.2rem 0.6rem;
        font-size:0.75rem;font-weight:600;color:{bc};white-space:nowrap;flex-shrink:0;
        border:1px solid {bc}33">{pct:.0f}% match</div>
  </div>

  <div style="display:flex;flex-wrap:wrap;gap:5px;margin:0.8rem 0 0.7rem">
    <span style="background:#1F2330;border:1px solid #2A2F3E;border-radius:5px;
        padding:0.15rem 0.6rem;font-size:0.73rem;color:#D1D5DB">{cuisine[:35]}</span>
    <span style="background:#1F2330;border:1px solid #2A2F3E;border-radius:5px;
        padding:0.15rem 0.6rem;font-size:0.73rem;color:#FCD34D">⭐ {rate_s} / 5</span>
    <span style="background:#1F2330;border:1px solid #2A2F3E;border-radius:5px;
        padding:0.15rem 0.6rem;font-size:0.73rem;color:#6EE7B7">💰 {cost_s}</span>
  </div>

  <div style="display:flex;justify-content:space-between;font-size:0.72rem;
      color:#4B5563;margin-bottom:5px">
    <span>Similarity score</span>
    <span style="color:{bc};font-weight:600">{pct:.1f}%</span>
  </div>
  <div style="height:4px;background:#1F2330;border-radius:2px">
    <div style="width:{pct:.1f}%;height:100%;background:{bc};border-radius:2px"></div>
  </div>
</div>"""
            with (col_a if i % 2 == 0 else col_b):
                st.markdown(card, unsafe_allow_html=True)

    # ── TAB 2 — Table ──────────────────────────────────────────────────────
    with tab2:
        disp = pd.DataFrame({
            "Rank":        range(1, len(results) + 1),
            "Restaurant":  results["name"].str.title().map(_sanitize)
                           if "name" in results.columns else "—",
            "Cuisine":     results["cuisines"].str.title().map(_sanitize)
                           if "cuisines" in results.columns else "—",
            "Location":    results["location"].str.title().map(_sanitize)
                           if "location" in results.columns else "—",
            "Rating":      results["rate"].map(lambda x: round(float(x), 1))
                           if "rate" in results.columns else 0,
            "Avg Cost":    results["cost"].map(
                               lambda x: f"₹{int(x):,}" if pd.notna(x) else "—")
                           if "cost" in results.columns else "—",
            "Match %":     (results["similarity_score"] * 100).map(lambda x: round(x, 1))
                           if "similarity_score" in results.columns else 0,
        })
        st.dataframe(
            disp, use_container_width=True, hide_index=True,
            column_config={
                "Rank":    st.column_config.NumberColumn(width="small"),
                "Rating":  st.column_config.ProgressColumn(
                               "Rating", min_value=0, max_value=5, format="%.1f ⭐", width="medium"),
                "Match %": st.column_config.ProgressColumn(
                               "Match %", min_value=0, max_value=100, format="%.1f%%", width="medium"),
            },
        )
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        csv = disp.to_csv(index=False)
        st.download_button(
            "📥 Download as CSV", csv,
            file_name=f"recommendations_for_{sel[:25].replace(' ','_')}.csv",
            mime="text/csv", use_container_width=False,
        )

    # ── TAB 3 — Insights ───────────────────────────────────────────────────
    with tab3:
        avg_r = results["rate"].mean()   if "rate" in results.columns   else None
        avg_c = results["cost"].mean()   if "cost" in results.columns   else None
        top_c = "—"
        if "cuisines" in results.columns and len(results) > 0:
            top_c = (results["cuisines"].str.split(",").explode()
                     .str.strip().value_counts().index[0].title())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Results",    len(results))
        m2.metric("Avg Rating", f"{avg_r:.2f} ⭐" if avg_r else "—")
        m3.metric("Avg Cost",   f"₹{int(avg_c):,}" if avg_c else "—")
        m4.metric("Top Cuisine", top_c)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**📍 Location breakdown**")
            if "location" in results.columns:
                st.bar_chart(results["location"].str.title().value_counts(),
                             color="#E53935")
        with ch2:
            st.markdown("**🍴 Cuisine breakdown**")
            if "cuisines" in results.columns:
                cuis = (results["cuisines"].str.split(",").explode()
                        .str.strip().str.title().value_counts().head(8))
                st.bar_chart(cuis, color="#FF7043")

    # Why these?
    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
    with st.expander("Why these restaurants?"):
        st.markdown(f"""
Each restaurant is represented by a *feature soup* — a text string combining its
**cuisine type**, **restaurant category**, and **neighbourhood**.

**TF-IDF** (Term Frequency–Inverse Document Frequency) converts those texts into
weighted numeric vectors, giving higher scores to distinctive terms.

**Cosine similarity** then measures the angle between two vectors: a score of 1.0
means identical profiles; 0.0 means completely different.

The {len(results)} restaurants shown above had the **highest cosine similarity**
to **"{last_q.title()}"**. Queries resolve in **< 50 ms** because the top-20
neighbours are precomputed for every restaurant at startup.
        """)

# ── idle state ────────────────────────────────────────────────────────────────
else:
    st.markdown("""
<div style="text-align:center;padding:5rem 2rem;background:#161A24;
    border:1px solid #1F2330;border-radius:12px;margin-top:0.5rem">
  <div style="font-size:3rem;margin-bottom:1rem">🔍</div>
  <h3 style="color:#F9FAFB;font-weight:700;margin:0 0 0.4rem">
      Search for a restaurant to begin</h3>
  <p style="color:#6B7280;font-size:0.9rem;max-width:420px;margin:0 auto 1.5rem">
      Select any restaurant from the dropdown above and click
      <strong style="color:#E53935">Find Similar</strong>,
      or press <strong style="color:#E53935">🎲</strong> for a random pick.</p>
  <div style="display:flex;justify-content:center;flex-wrap:wrap;gap:8px">
""" + "".join(
        f'<span style="background:#1F2330;border:1px solid #2A2F3E;border-radius:6px;'
        f'padding:0.25rem 0.8rem;font-size:0.78rem;color:#9CA3AF">{v} {k.title()}</span>'
        for k, v in list(CUISINE_EMOJI.items())[:10]
    ) + """
  </div>
</div>
""", unsafe_allow_html=True)
