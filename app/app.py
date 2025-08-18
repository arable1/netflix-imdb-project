# app/app.py
import os
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import pandas as pd
import joblib

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

# ---- Paths ----
REG_PIPE_PATH   = ROOT / "models" / "tuned_randomforest_pipeline.joblib"
CLF_PIPE_PATH   = ROOT / "models" / "clf_randomforest_tuned.joblib"
STRICT_TABLE    = ROOT / "data" / "processed" / "model_strict.csv"
COMBINED_CLEAN  = ROOT / "data" / "processed" / "netflix_combined_clean_synced.csv"
SIM_ARTIFACTS   = ROOT / "artifacts" / "tfidf_similarity.joblib"

# ---- App ----
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "devkey")

# ---- Load data used by pages ----
df_features = pd.read_csv(STRICT_TABLE)            # model-ready (for predictor)
df_content  = pd.read_csv(COMBINED_CLEAN)          # human-facing (for recommend/similar)

def norm(s):
    return s.fillna("Unknown").astype(str)

for col in ["type", "age_certification", "country", "listed_in", "description"]:
    if col in df_content.columns:
        df_content[col] = norm(df_content[col])

# Ranges for UI
YEAR_MIN = int(df_content["release_year"].min())
YEAR_MAX = int(df_content["release_year"].max())
DUR_MIN  = int(np.nan_to_num(df_content["duration_minutes"]).min())
DUR_MAX  = int(np.nan_to_num(df_content["duration_minutes"]).max())
DESC_MIN = int(df_content["description"].fillna("").str.len().min())
DESC_MAX = int(df_content["description"].fillna("").str.len().max())
VOTE_MIN = int(df_features["imdb_votes"].min())
VOTE_MAX = int(df_features["imdb_votes"].max())
SCORE_MIN = float(df_content["imdb_score"].min())
SCORE_MAX = float(df_content["imdb_score"].max())

TYPE_OPTS = ["ALL", "MOVIE", "SHOW"]
AGE_OPTS  = ["ALL","G","PG","PG-13","R","TV-14","TV-MA","TV-PG","TV-Y","TV-Y7","Unknown"]
COUNTRY_OPTS = [
    "ALL","United States","United Kingdom","India","Japan",
    "South Korea","Canada","Spain","France","Philippines",
    "Nigeria","Other","Unknown"
]

def expand_genres(df):
    out = set()
    for s in df["listed_in"].dropna().tolist():
        for g in [x.strip() for x in s.split(",") if x.strip()]:
            out.add(g)
    return sorted(out)

GENRE_OPTS = expand_genres(df_content)

# ---- Load pipelines ----
reg_pipe = joblib.load(REG_PIPE_PATH) if REG_PIPE_PATH.exists() else None
clf_pipe = joblib.load(CLF_PIPE_PATH) if CLF_PIPE_PATH.exists() else None

# ---- Similarity artifacts (optional) ----
SIM_ON = False
vectorizer = None
tfidf = None
df_sim = None

def _get(pack, *keys):
    for k in keys:
        if k in pack:
            return pack[k]
    return None

if SIM_ARTIFACTS.exists():
    try:
        pack = joblib.load(SIM_ARTIFACTS)
        vectorizer = _get(pack, "vectorizer", "tfidf_vectorizer")
        tfidf      = _get(pack, "tfidf", "X", "matrix")
        df_sim     = _get(pack, "df", "corpus_df", "dataframe")

        if vectorizer is None or tfidf is None or df_sim is None:
            missing = []
            if vectorizer is None: missing.append("vectorizer")
            if tfidf is None:      missing.append("tfidf")
            if df_sim is None:     missing.append("df")
            raise KeyError(f"Artifact missing required keys: {missing}")

        for col in ["type", "listed_in", "country", "age_certification"]:
            if col in df_sim.columns:
                df_sim[col] = df_sim[col].fillna("Unknown").astype(str)

        SIM_ON = True
        print("[ok] Similarity artifacts loaded:",
              f"matrix={getattr(tfidf, 'shape', None)} rows={len(df_sim)}")

    except Exception as e:
        print("[warn] Could not load similarity artifacts:", repr(e))
else:
    print("[warn] Similarity artifacts not found at", SIM_ARTIFACTS)

# ---- Feature scaffold (column names) for building one prediction row ----
DROP_NON_FEATURES = {"imdb_score","title","imdb_id","id","description","is_high"}
X_COLS = [c for c in df_features.columns if c not in DROP_NON_FEATURES]

def classic_flag(year: int) -> int:
    try:
        y = int(year)
    except Exception:
        return 0
    return 1 if y < 2000 else 0

def type_to_is_show(value: str):
    if value == "SHOW":
        return 1
    if value == "MOVIE":
        return 0
    return None  # for ALL

def build_one_row_from_form(form) -> pd.DataFrame:
    """Build a single-row frame matching the model input columns."""
    row = pd.DataFrame({c: [0] for c in X_COLS})

    # Required numerics (with sensible fallbacks)
    release_year     = int(form.get("release_year") or YEAR_MAX)
    duration_minutes = int(form.get("duration_minutes") or int(np.nanmedian(df_content["duration_minutes"])))
    desc_len         = int(form.get("desc_len") or int(df_content["description"].fillna("").str.len().median()))
    num_genres       = int(form.get("num_genres") or 2)

    votes_str = form.get("imdb_votes") or ""
    if votes_str.strip():
        imdb_votes = int(votes_str)
    else:
        imdb_votes = int(df_features["imdb_votes"].median())
    imdb_votes_log1p = float(np.log1p(imdb_votes))

    row.loc[0, "release_year"] = release_year
    row.loc[0, "duration_minutes"] = duration_minutes
    row.loc[0, "desc_len"] = desc_len
    row.loc[0, "num_genres"] = num_genres
    row.loc[0, "imdb_votes"] = imdb_votes
    row.loc[0, "imdb_votes_log1p"] = imdb_votes_log1p

    # is_show / is_classic
    type_choice = form.get("type") or "ALL"
    is_show_val = type_to_is_show(type_choice)
    if is_show_val is not None and "is_show" in row.columns:
        row.loc[0, "is_show"] = is_show_val
    if "is_classic" in row.columns:
        row.loc[0, "is_classic"] = classic_flag(release_year)

    # 'type' categorical (if present among features)
    if "type" in row.columns:
        row["type"] = "MOVIE" if type_choice == "ALL" else type_choice

    # Age one-hots
    age_choice = form.get("age_cert") or "ALL"
    age_cols = [c for c in X_COLS if c.startswith("age_")]
    for c in age_cols:
        row.loc[0, c] = 0
    if age_choice != "ALL":
        colname = f"age_{age_choice}"
        if colname in row.columns:
            row.loc[0, colname] = 1

    # Country one-hots
    ctry_choice = form.get("country") or "ALL"
    ctry_cols = [c for c in X_COLS if c.startswith("ctry_")]
    for c in ctry_cols:
        row.loc[0, c] = 0
    if ctry_choice != "ALL":
        colname = f"ctry_{ctry_choice}"
        if colname in row.columns:
            row.loc[0, colname] = 1

    # Ensure numeric dtypes for vote fields (quiet a Pandas warning)
    for col in ("imdb_votes", "imdb_votes_log1p"):
        if col in row.columns:
            row[col] = row[col].astype("float64")

    return row

# ----------- Routes -----------
@app.get("/debug/sim")
def debug_sim():
    return {
        "SIM_ON": SIM_ON,
        "tfidf_shape": getattr(tfidf, "shape", None),
        "rows_in_df_sim": None if df_sim is None else int(len(df_sim)),
        "df_sim_cols": None if df_sim is None else list(df_sim.columns)[:12],
    }

@app.get("/")
def home():
    return render_template(
        "index.html",
        type_opts=TYPE_OPTS,
        age_opts=AGE_OPTS,
        country_opts=COUNTRY_OPTS,
        year_min=YEAR_MIN, year_max=YEAR_MAX,
        dur_min=DUR_MIN, dur_max=DUR_MAX,
        desc_min=DESC_MIN, desc_max=DESC_MAX,
        votes_min=VOTE_MIN, votes_max=VOTE_MAX,
        score_min=SCORE_MIN, score_max=SCORE_MAX
    )

@app.post("/predict")
def predict():
    if reg_pipe is None or clf_pipe is None:
        flash("Models not found. Please train and save the pipelines.", "error")
        return redirect(url_for("home"))

    row = build_one_row_from_form(request.form)

    # Regression
    pred_score = float(reg_pipe.predict(row)[0])
    pred_score = max(min(pred_score, SCORE_MAX), SCORE_MIN)

    # Classification
    proba = float(clf_pipe.predict_proba(row)[0][1])
    classification = "High (≥ 7.0)" if proba >= 0.5 else "Not High (< 7.0)"

    return render_template(
        "result.html",
        pred_score=f"{pred_score:.2f}",
        high_prob=f"{proba:.2%}",
        classification=classification,
        score_min=SCORE_MIN, score_max=SCORE_MAX
    )

@app.route("/recommend", methods=["GET","POST"])
def recommend():
    results = None
    return render_template(
        "recommend.html",
        type_opts=TYPE_OPTS,
        genre_opts=GENRE_OPTS,
        country_opts=COUNTRY_OPTS,
        year_min=YEAR_MIN, year_max=YEAR_MAX,
        score_min=SCORE_MIN, score_max=SCORE_MAX,
        results=results
    )

@app.post("/recommend/search")
def recommend_search():
    type_choice = request.form.get("type") or "ALL"
    genres      = request.form.getlist("genres")
    country     = request.form.get("country") or "ALL"
    min_year    = int(request.form.get("min_year") or YEAR_MIN)
    max_year    = int(request.form.get("max_year") or YEAR_MAX)
    min_score   = float(request.form.get("min_score") or SCORE_MIN)
    top_n       = int(request.form.get("top_n") or 20)

    df = df_content.copy()

    if type_choice != "ALL":
        df = df[df["type"] == type_choice]
    if country != "ALL":
        df = df[df["country"].str.contains(country, na=False)]

    df = df[(df["release_year"] >= min_year) & (df["release_year"] <= max_year)]
    df = df[df["imdb_score"] >= min_score]

    for g in genres:
        df = df[df["listed_in"].str.contains(g, na=False)]

    if "imdb_votes" in df.columns:
        df = df.sort_values(["imdb_score","imdb_votes"], ascending=[False, False])
    else:
        df = df.sort_values("imdb_score", ascending=False)

    results = df[["title","type","release_year","age_certification","country","listed_in","imdb_score"]].head(top_n)

    return render_template(
        "recommend.html",
        type_opts=TYPE_OPTS,
        genre_opts=GENRE_OPTS,
        country_opts=COUNTRY_OPTS,
        year_min=YEAR_MIN, year_max=YEAR_MAX,
        score_min=SCORE_MIN, score_max=SCORE_MAX,
        results=results.to_dict(orient="records")
    )

@app.route("/similar", methods=["GET","POST"])
def similar():
    if request.method == "GET":
        return render_template(
            "similar.html",
            type_opts=TYPE_OPTS,
            genre_opts=GENRE_OPTS,
            country_opts=COUNTRY_OPTS,
            age_opts=AGE_OPTS,
            year_min=YEAR_MIN, year_max=YEAR_MAX,
            enabled=SIM_ON,
            results=None
        )

    if not SIM_ON:
        flash("Similarity artifacts not available. Run scripts/build_similarity.py first.", "error")
        return redirect(url_for("similar"))

    type_choice = request.form.get("type") or "ALL"
    query_text  = (request.form.get("query") or "").strip()
    genres      = request.form.getlist("genres")
    country     = request.form.get("country") or "ALL"
    age_cert    = request.form.get("age_cert") or "ALL"
    year_val    = request.form.get("year") or ""
    top_n       = int(request.form.get("top_n") or 20)

    # Base similarity
    if query_text:
        q_vec = vectorizer.transform([query_text])
        sims = (tfidf @ q_vec.T).toarray().ravel()
        base = df_sim.copy()
        base["sim"] = sims
    else:
        base = df_sim.copy()
        base["sim"] = 1.0

    # Filters
    if type_choice != "ALL":
        base = base[base["type"] == type_choice]
    if genres:
        for g in genres:
            base = base[base["listed_in"].str.contains(g, na=False)]
    if country != "ALL":
        base = base[base["country"].str.contains(country, na=False)]
    if age_cert != "ALL":
        base = base[base["age_certification"] == age_cert]
    if year_val.strip():
        try:
            y = int(year_val)
            base = base[base["release_year"] == y]
        except Exception:
            pass

    base = base.sort_values(["sim","imdb_score"], ascending=[False, False]).head(top_n)
    results = base[["title","type","release_year","age_certification","country","listed_in","imdb_score","sim"]].to_dict(orient="records")

    return render_template(
        "similar.html",
        type_opts=TYPE_OPTS,
        genre_opts=GENRE_OPTS,
        country_opts=COUNTRY_OPTS,
        age_opts=AGE_OPTS,
        year_min=YEAR_MIN, year_max=YEAR_MAX,
        enabled=SIM_ON,
        results=results
    )

if __name__ == "__main__":
    app.run(debug=True)