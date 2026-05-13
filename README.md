# 🍽️ Zomato Bangalore Restaurant Recommender

A **content-based recommendation system** that suggests Bangalore restaurants similar to a user-chosen restaurant. Built with Python 3.10+, scikit-learn TF-IDF cosine similarity, and Streamlit.

---

## 📐 Project Structure

```
zomato-recommender/
├── app.py              ← Streamlit UI entry-point
├── recommender.py      ← Cosine-similarity recommendation engine
├── preprocess.py       ← Data cleaning, feature engineering & TF-IDF pipeline
├── data/
│   └── zomato.csv      ← Raw dataset (you must place this here)
├── models/             ← Auto-generated preprocessing artefacts (git-ignored)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚡ Quickstart

### 1 — Get the dataset

Download the **Zomato Bangalore Restaurants** dataset from Kaggle and place it at:

```
data/zomato.csv
```

> **Kaggle link:** https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants

### 2 — Create & activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — (Optional) Pre-build artefacts

The first launch builds TF-IDF artefacts automatically. To pre-build manually:

```bash
python preprocess.py
```

Artefacts are saved in `models/` and reused on subsequent runs.

### 5 — Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🧠 How It Works

| Stage | Module | What happens |
|-------|--------|-------------|
| **Ingest & clean** | `preprocess.py` | Normalise columns, parse ratings/costs, drop duplicates |
| **Feature engineering** | `preprocess.py` | Build a weighted text "soup" (cuisines × 3, rest_type × 2, location, dishes, cost/rating bins) |
| **Vectorise** | `preprocess.py` | Fit a TF-IDF matrix (up to 10 000 uni- and bi-gram features) |
| **Recommend** | `recommender.py` | Cosine similarity between the query restaurant and all others; rank & filter |
| **UI** | `app.py` | Streamlit interface — search, filters, card results, similarity score bars |

### Filtering options (sidebar)

- **Location** — restrict to a specific Bangalore neighbourhood  
- **Cuisine type** — show only matching cuisines  
- **Max cost for two** — upper price bound (₹)  
- **Minimum rating** — lower star bound  
- **Online order / Table booking** — boolean flags  

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.10 | Language |
| pandas | 2.2.2 | Data wrangling |
| numpy | 1.26.4 | Numerical ops |
| scikit-learn | 1.4.2 | TF-IDF, cosine similarity |
| streamlit | 1.35.0 | Web UI |
| joblib | 1.4.2 | Artefact serialisation |
| nltk | 3.8.1 | NLP utilities |

---

## 🗂 Data Notes

- **Source:** Zomato Bangalore dataset (~51 000 rows, 17 columns)  
- The `rate` column is parsed from strings like `"4.1/5"` or `"NEW"`.  
- `approx_cost(for two people)` is cleaned and cast to float.  
- Exact duplicates (same name + location) are dropped before vectorisation.  

---

## 📦 Deployment (Antigravity / Streamlit Cloud)

1. Push this repository (without `data/` and `models/`) to GitHub.  
2. Add `zomato.csv` via the platform's secret-storage or a cloud bucket, then adjust `DATA_PATH` in `preprocess.py`.  
3. Set the **main file** to `app.py` in the deployment dashboard.  
4. Streamlit `>= 1.30` is required (pinned at `1.35.0` here).  

---

## 📄 License

MIT — feel free to use, modify, and distribute.
