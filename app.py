"""app.py — Zomato Restaurant Recommender  (UI v3)"""
from __future__ import annotations
import os, random
from pathlib import Path
import pandas as pd
import streamlit as st
from preprocess import load_and_clean
from recommender import build_recommender

DATA_PATH     = "data/zomato.csv"
_PARQUET      = Path("cache/zomato_clean.parquet")
_HIST_MAX     = 6

CUISINE_EMOJI = {
    "north indian":"🍛","south indian":"🥘","chinese":"🥡","italian":"🍕",
    "continental":"🍽️","fast food":"🍔","bakery":"🥐","cafe":"☕",
    "desserts":"🍮","ice cream":"🍦","pizza":"🍕","biryani":"🍚",
    "seafood":"🦐","mughlai":"🍖","mexican":"🌮","thai":"🍜",
    "japanese":"🍣","american":"🌭","street food":"🥙","healthy food":"🥗",
    "sandwich":"🥪","burger":"🍔","momos":"🥟","sushi":"🍣",
}

GRAD = [
    "linear-gradient(135deg,#f72585,#b5179e)",
    "linear-gradient(135deg,#7209b7,#3a0ca3)",
    "linear-gradient(135deg,#4361ee,#4cc9f0)",
    "linear-gradient(135deg,#f77f00,#fcbf49)",
    "linear-gradient(135deg,#2dc653,#008000)",
    "linear-gradient(135deg,#e63946,#f4a261)",
    "linear-gradient(135deg,#06d6a0,#118ab2)",
    "linear-gradient(135deg,#ff6b6b,#feca57)",
    "linear-gradient(135deg,#a8edea,#fed6e3)",
    "linear-gradient(135deg,#5f27cd,#48dbfb)",
]

def c_emoji(s):
    if not s or s=="—": return "🍴"
    s=s.lower()
    for k,v in CUISINE_EMOJI.items():
        if k in s: return v
    return "🍴"

def stars(r):
    try:
        r=float(r); f=int(r); h=1 if r-f>=0.25 else 0
        return "★"*f+("½" if h else "")+"☆"*(5-f-h)
    except: return "—"

def bar_color(pct):
    if pct>=70: return "#06d6a0"
    if pct>=45: return "#ffd166"
    return "#ef233c"

# ─── page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="🍽️ Zomato Finder",page_icon="🍽️",layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800;900&display=swap');
*,html,body,[class*="css"]{font-family:'Poppins',sans-serif!important}
.stApp{background:#0a0a0f!important}
.block-container{padding-top:1rem!important;max-width:1300px!important}

/* sidebar */
[data-testid="stSidebar"]{
  background:linear-gradient(160deg,#0d0015 0%,#0a001f 50%,#000d1a 100%)!important;
  border-right:2px solid rgba(255,107,107,0.2)!important}
[data-testid="stSidebar"] *{color:#dde!important}
[data-testid="stSidebar"] h2{
  background:linear-gradient(90deg,#ff6b6b,#feca57);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  font-size:1.4rem!important;font-weight:900!important}
[data-testid="stSidebar"] h3{color:#feca57!important;font-weight:700!important}
[data-testid="stSidebar"] hr{border-color:rgba(255,107,107,0.2)!important}

/* buttons */
.stButton>button{
  background:linear-gradient(135deg,#ff6b6b,#feca57)!important;
  color:#000!important;border:none!important;border-radius:12px!important;
  font-weight:800!important;font-size:0.95rem!important;
  transition:all .2s!important;box-shadow:0 4px 20px rgba(255,107,107,0.4)!important}
.stButton>button:hover{transform:translateY(-3px) scale(1.02)!important;
  box-shadow:0 8px 30px rgba(255,107,107,0.6)!important}

/* selectbox */
div[data-baseweb="select"]>div{
  background:rgba(255,255,255,0.05)!important;
  border:1.5px solid rgba(255,107,107,0.4)!important;
  border-radius:12px!important;color:#fff!important;font-size:1rem!important}
div[data-baseweb="select"]>div:focus-within{
  border-color:#feca57!important;box-shadow:0 0 0 3px rgba(254,202,87,0.2)!important}

/* tabs */
.stTabs [data-baseweb="tab-list"]{
  background:rgba(255,255,255,0.04)!important;border-radius:14px!important;padding:4px!important}
.stTabs [data-baseweb="tab"]{
  border-radius:10px!important;font-weight:600!important;color:#aaa!important}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,#ff6b6b,#feca57)!important;
  color:#000!important;font-weight:800!important}

/* expander */
.streamlit-expanderHeader{background:rgba(255,255,255,0.04)!important;
  border-radius:10px!important;font-weight:600!important}
</style>
""", unsafe_allow_html=True)

# ─── data & model ─────────────────────────────────────────────────────────────
def _fresh(csv, pq):
    if not pq.exists(): return False
    try: return pq.stat().st_mtime >= os.path.getmtime(csv)
    except: return False

@st.cache_data(ttl=3600, show_spinner="📂 Loading dataset…")
def get_data(p):
    _PARQUET.parent.mkdir(parents=True, exist_ok=True)
    if _fresh(p, _PARQUET): return pd.read_parquet(_PARQUET)
    df = load_and_clean(p)
    df.to_parquet(_PARQUET, index=False)
    return df

df = get_data(DATA_PATH)
if "rec" not in st.session_state:
    with st.spinner("⚙️ Warming up the engine…"):
        st.session_state["rec"] = build_recommender(df)
rec = st.session_state["rec"]
all_names = rec.get_all_names()

for k,v in [("toasted",False),("history",[]),("selected",all_names[0]),
            ("results",None),("last_q",""),("compare",[])]:
    if k not in st.session_state: st.session_state[k]=v

if not st.session_state["toasted"]:
    st.toast("🎉 Ready! Explore 8,700+ Bangalore restaurants",icon="🍽️")
    st.session_state["toasted"]=True

# ─── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ Zomato Finder")
    st.caption("Content-based restaurant discovery · Bangalore")
    st.divider()

    st.markdown("### ⚙️ Settings")
    top_n = st.slider("Recommendations",1,10,5)
    min_rating = st.slider("Min Rating ⭐",0.0,5.0,0.0,0.1,
                           help="Filter results by minimum rating")
    st.divider()

    # Stats
    st.markdown("### 📊 Dataset")
    n_r = len(df)
    n_c = df["cuisines"].dropna().str.split(",").explode().str.strip().nunique()
    n_l = df["location"].nunique()
    cols = st.columns(3)
    for col,val,lbl in zip(cols,[f"{n_r:,}",str(n_c),str(n_l)],["🏪","🍴","📍"]):
        col.metric(lbl,val)
    st.caption(f"🏪 {n_r:,} restaurants · 🍴 {n_c} cuisines · 📍 {n_l} areas")
    st.divider()

    # History
    if st.session_state["history"]:
        st.markdown("### 🕑 Recent")
        for h in reversed(st.session_state["history"]):
            if st.button(f"↩ {h[:24]}",key=f"h_{h}",use_container_width=True):
                st.session_state["selected"]=h
                st.rerun()
        if st.button("🗑 Clear",use_container_width=True):
            st.session_state["history"]=[]; st.rerun()
        st.divider()

    with st.expander("💡 How it works"):
        st.markdown("""
**Content-Based Filtering**
1. Each restaurant → *feature soup* (cuisine + category + area)
2. **TF-IDF** converts text → weighted numeric vector
3. **Cosine similarity** finds closest neighbours
4. Top-20 precomputed → **O(1) queries** (<50 ms)
        """)

# ─── hero banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#ff6b6b 0%,#feca57 40%,#48dbfb 70%,#ff9ff3 100%);
border-radius:24px;padding:2.5rem 2.8rem;margin-bottom:1.6rem;position:relative;overflow:hidden">
<div style="position:absolute;top:-30px;right:-30px;width:200px;height:200px;
  background:rgba(255,255,255,0.08);border-radius:50%"></div>
<div style="position:absolute;bottom:-40px;left:20%;width:150px;height:150px;
  background:rgba(255,255,255,0.06);border-radius:50%"></div>
<h1 style="font-size:2.6rem;font-weight:900;color:#fff;margin:0;
  text-shadow:0 2px 20px rgba(0,0,0,0.3)">🍽️ Zomato Restaurant Finder</h1>
<p style="color:rgba(255,255,255,0.9);font-size:1.1rem;margin:0.4rem 0 0;font-weight:500">
Discover your next favourite Bangalore restaurant — <b>8,700+ choices</b> powered by AI similarity</p>
</div>
""", unsafe_allow_html=True)

# ─── search bar ──────────────────────────────────────────────────────────────
c1,c2,c3 = st.columns([5,2,1],gap="small")
with c1:
    try: idx=all_names.index(st.session_state["selected"])
    except: idx=0
    sel = st.selectbox("🔍 Restaurant",all_names,index=idx,
                       label_visibility="collapsed",
                       placeholder="Type to search…")
    st.session_state["selected"]=sel
with c2:
    go = st.button("🚀 Find Similar",use_container_width=True)
with c3:
    if st.button("🎲",help="Random pick",use_container_width=True):
        st.session_state["selected"]=random.choice(all_names); st.rerun()

# ─── run recommendation ───────────────────────────────────────────────────────
if go:
    try:
        res = rec.recommend(sel, top_n=top_n)
        if min_rating>0 and "rate" in res.columns:
            res = res[res["rate"]>=min_rating]
        st.session_state["results"]=res
        st.session_state["last_q"]=sel
        hist=st.session_state["history"]
        if sel in hist: hist.remove(sel)
        hist.append(sel)
        if len(hist)>_HIST_MAX: hist.pop(0)
    except ValueError as e:
        st.error(str(e),icon="🚫"); st.session_state["results"]=None

results = st.session_state["results"]
last_q  = st.session_state["last_q"]

# ─── results ─────────────────────────────────────────────────────────────────
if results is not None and not results.empty:
    # summary strip
    avg_score = results["similarity_score"].mean()*100 if "similarity_score" in results else 0
    st.markdown(f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:1rem">
      <div style="background:linear-gradient(135deg,#ff6b6b,#feca57);border-radius:12px;
        padding:.5rem 1.2rem;color:#000;font-weight:800;font-size:0.9rem">
        🎯 {len(results)} matches for "{last_q.title()}"</div>
      <div style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);
        border-radius:12px;padding:.5rem 1.2rem;color:#fff;font-size:0.85rem">
        ✨ Avg match: <b>{avg_score:.0f}%</b></div>
      <div style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);
        border-radius:12px;padding:.5rem 1.2rem;color:#fff;font-size:0.85rem">
        ⭐ Filter: ≥ {min_rating}</div>
    </div>
    """, unsafe_allow_html=True)

    # tabs
    tab1, tab2, tab3 = st.tabs(["🃏 Restaurant Cards","📊 Comparison Table","📈 Insights"])

    # ── TAB 1 : Cards ────────────────────────────────────────────────────────
    with tab1:
        cols2 = st.columns(2, gap="medium")
        for i,(_, row) in enumerate(results.iterrows()):
            name    = str(row.get("name","")).title() or "—"
            cuisine = str(row.get("cuisines","")).title() or "—"
            loc     = str(row.get("location","")).title() or "—"
            score   = float(row.get("similarity_score",0))
            pct     = score*100
            bc      = bar_color(pct)
            emoji   = c_emoji(str(row.get("cuisines","")))
            try: rate_s = f"{float(row['rate']):.1f}"
            except: rate_s = "—"
            try: cost_s = f"₹{int(float(row['cost'])):,}"
            except: cost_s = "—"
            st_txt  = stars(row.get("rate"))
            grad    = GRAD[i % len(GRAD)]

            with cols2[i%2]:
                st.markdown(f"""
<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
  border-radius:18px;padding:1.3rem 1.4rem;margin-bottom:1rem;
  transition:all .25s;position:relative;overflow:hidden">
  <div style="position:absolute;top:0;left:0;right:0;height:4px;background:{grad}"></div>

  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px">
    <div style="display:flex;align-items:center;gap:10px;flex:1;min-width:0">
      <div style="background:{grad};width:36px;height:36px;border-radius:10px;
        display:flex;align-items:center;justify-content:center;
        font-size:1.2rem;flex-shrink:0">{emoji}</div>
      <div style="min-width:0">
        <div style="font-size:1rem;font-weight:800;color:#fff;
          white-space:nowrap;overflow:hidden;text-overflow:ellipsis">#{i+1} {name}</div>
        <div style="font-size:0.78rem;color:#888;margin-top:1px">📍 {loc}</div>
      </div>
    </div>
    <div style="background:{grad};border-radius:20px;padding:.25rem .8rem;
      font-size:0.78rem;font-weight:800;color:#000;white-space:nowrap;flex-shrink:0">
      {pct:.0f}% match</div>
  </div>

  <div style="display:flex;flex-wrap:wrap;gap:6px;margin:0.85rem 0 0.7rem">
    <span style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);
      border-radius:20px;padding:.18rem .65rem;font-size:.73rem;color:#ddd">{cuisine[:30]}</span>
    <span style="background:rgba(254,202,87,0.12);border:1px solid rgba(254,202,87,0.25);
      border-radius:20px;padding:.18rem .65rem;font-size:.73rem;color:#feca57">💰 {cost_s}</span>
    <span style="background:rgba(6,214,160,0.1);border:1px solid rgba(6,214,160,0.25);
      border-radius:20px;padding:.18rem .65rem;font-size:.73rem;color:#06d6a0">⭐ {rate_s}/5</span>
  </div>

  <div style="font-size:.78rem;color:#666;margin-bottom:4px;
    display:flex;justify-content:space-between">
    <span>Similarity match</span>
    <span style="color:{bc};font-weight:700">{pct:.1f}%</span>
  </div>
  <div style="height:6px;background:rgba(255,255,255,0.07);border-radius:3px;overflow:hidden">
    <div style="width:{pct:.1f}%;height:100%;background:{bc};border-radius:3px"></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── TAB 2 : Table ────────────────────────────────────────────────────────
    with tab2:
        disp = pd.DataFrame({
            "Rank":    range(1, len(results)+1),
            "Restaurant": results["name"].str.title() if "name" in results else "—",
            "Cuisine":    results["cuisines"].str.title() if "cuisines" in results else "—",
            "Location":   results["location"].str.title() if "location" in results else "—",
            "Rating ⭐":  results["rate"].map(lambda x: f"{x:.1f}") if "rate" in results else "—",
            "Cost 💰":    results["cost"].map(lambda x: f"₹{int(x):,}" if pd.notna(x) else "—") if "cost" in results else "—",
            "Match %":    (results["similarity_score"]*100).map(lambda x: f"{x:.1f}%") if "similarity_score" in results else "—",
        })
        st.dataframe(disp, use_container_width=True, hide_index=True,
                     column_config={
                         "Rank": st.column_config.NumberColumn(width="small"),
                         "Match %": st.column_config.TextColumn(width="small"),
                         "Rating ⭐": st.column_config.TextColumn(width="small"),
                     })
        csv = disp.to_csv(index=False)
        st.download_button("📥 Download CSV", csv,
                           file_name=f"similar_to_{last_q[:20]}.csv",
                           mime="text/csv", use_container_width=True)

    # ── TAB 3 : Insights ─────────────────────────────────────────────────────
    with tab3:
        ic1, ic2, ic3 = st.columns(3)
        avg_r = results["rate"].mean() if "rate" in results else 0
        avg_c = results["cost"].mean() if "cost" in results else 0
        top_c = (results["cuisines"].str.split(",").explode().str.strip()
                 .value_counts().index[0].title()
                 if "cuisines" in results and len(results)>0 else "—")
        ic1.metric("⭐ Avg Rating", f"{avg_r:.2f}" if avg_r else "—")
        ic2.metric("💰 Avg Cost",   f"₹{int(avg_c):,}" if avg_c else "—")
        ic3.metric("🍴 Top Cuisine", top_c)

        st.markdown("#### 📍 Location Distribution")
        if "location" in results.columns:
            loc_counts = results["location"].str.title().value_counts()
            st.bar_chart(loc_counts)

        st.markdown("#### 🍴 Cuisine Spread")
        if "cuisines" in results.columns:
            cuis = (results["cuisines"].str.split(",").explode()
                    .str.strip().str.title().value_counts().head(10))
            st.bar_chart(cuis)

    # explainer
    with st.expander("💡 Why these restaurants?"):
        st.markdown(f"""
Each restaurant is described by a **feature soup** — cuisine + category + neighbourhood.

**TF-IDF** converts these texts into numeric vectors; **cosine similarity** measures
how "close" two restaurants are. The restaurants above are the closest to
**"{last_q.title()}"** in this multi-dimensional feature space.

*Queries run in < 50 ms* — top-20 neighbours are precomputed at startup.
        """)

else:
    # idle
    st.markdown("""
<div style="text-align:center;padding:4rem 1rem;
  background:linear-gradient(135deg,rgba(255,107,107,0.05),rgba(254,202,87,0.05));
  border:2px dashed rgba(255,107,107,0.2);border-radius:24px;margin-top:0.5rem">
  <div style="font-size:4rem">🍽️</div>
  <h2 style="color:#fff;font-weight:800;margin:.5rem 0 .3rem">What are you craving?</h2>
  <p style="color:#888;font-size:1rem">
    Pick a restaurant you love above and hit <b style="color:#feca57">🚀 Find Similar</b><br>
    or let the <b style="color:#ff6b6b">🎲 dice</b> decide for you!
  </p>
  <div style="margin-top:1.5rem;display:flex;justify-content:center;gap:1rem;flex-wrap:wrap">
""" + "".join([
    f'<span style="background:linear-gradient(135deg,{g});border-radius:20px;'
    f'padding:.3rem .9rem;color:#000;font-weight:700;font-size:.8rem">'
    f'{e} {c.title()}</span>'
    for (c,e),g in zip(list(CUISINE_EMOJI.items())[:8], GRAD)
]) + """
  </div>
</div>
""", unsafe_allow_html=True)
