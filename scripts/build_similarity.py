# scripts/build_similarity.py
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "netflix_combined_clean_synced.csv"
OUT  = ROOT / "artifacts" / "tfidf_similarity.joblib"

def norm(s):
    return s.fillna("").astype(str)

def main():
    df = pd.read_csv(DATA)

    # Keep only the columns we’ll need at inference time
    cols_needed = [
        "title","description","type","listed_in","country",
        "age_certification","release_year","imdb_score"
    ]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {DATA}: {missing}")

    df = df[cols_needed].copy()
    df["description"] = norm(df["description"])

    # Build TF-IDF on description (you can concatenate title if you want)
    corpus = df["description"].tolist()
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1,2),
        min_df=2,
        stop_words="english"
    )
    tfidf = vectorizer.fit_transform(corpus)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "tfidf": tfidf, "df": df}, OUT)
    print(f"✓ Saved {OUT}")

if __name__ == "__main__":
    main()