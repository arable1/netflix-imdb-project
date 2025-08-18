import pandas as pd

# Column names your model expects (saved in artifacts/features_used.json in nb4)
# We’ll fetch this from artifacts at runtime in modeling.py, but keep helpers here.

SAFE_NUMS = [
    "imdb_votes","imdb_votes_log1p","duration_minutes","release_year",
    "desc_len","num_genres","is_show","is_classic"
]

SAFE_CATS = [
    "type","age_G","age_Other","age_PG","age_PG-13","age_R","age_TV-14",
    "age_TV-G","age_TV-MA","age_TV-PG","age_TV-Y","age_TV-Y7","age_Unknown",
    "ctry_Canada","ctry_France","ctry_India","ctry_Japan","ctry_Nigeria","ctry_Other",
    "ctry_Philippines","ctry_South Korea","ctry_Spain","ctry_United Kingdom",
    "ctry_United States","ctry_Unknown"
]

def form_to_frame(form: dict) -> pd.DataFrame:
    """
    Convert a dict (from Flask form/JSON) into a single-row DataFrame with
    the raw feature columns your pipeline expects BEFORE ColumnTransformer.
    Minimal validation; do UI validation for better UX.
    """
    # numeric base
    row = {
        "imdb_votes": float(form.get("imdb_votes", 0) or 0),
        "imdb_votes_log1p": float(form.get("imdb_votes_log1p", 0) or 0),
        "duration_minutes": float(form.get("duration_minutes", 90) or 90),
        "release_year": int(form.get("release_year", 2018) or 2018),
        "desc_len": int(form.get("desc_len", 200) or 200),
        "num_genres": float(form.get("num_genres", 2) or 2),
        "is_show": 1 if form.get("type") == "SHOW" else 0,
        "is_classic": 1 if int(form.get("release_year", 2018)) < 2000 else 0,
        "type": (form.get("type") or "MOVIE"),
    }

    # one-hot toggles from dropdowns (age, country). We keep the same column names as training
    age_key = form.get("age_cert", "Unknown")
    country_key = form.get("country", "Unknown")

    # Set all to 0 then flip the chosen ones to 1
    for c in [c for c in SAFE_CATS if c.startswith("age_")]:
        row[c] = 0
    for c in [c for c in SAFE_CATS if c.startswith("ctry_")]:
        row[c] = 0

    row[f"age_{age_key}"] = 1 if f"age_{age_key}" in SAFE_CATS else 0
    row[f"ctry_{country_key}"] = 1 if f"ctry_{country_key}" in SAFE_CATS else 0

    return pd.DataFrame([row])