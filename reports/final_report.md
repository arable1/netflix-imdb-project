# Netflix Ratings: EDA → Modeling → App (Executive Summary)

## Objective
Predict IMDb score and classify “high-rated” titles (≥7.0), and ship a simple web app for recommendations.

## Data
- IMDb + Netflix metadata (strict join on title+year, plus a cautious loose pass).
- Final modeling table: 3,067 rows, 37 features (engineered: log votes, duration mins, is_show, etc.).
- License: CC0.

## Key EDA Findings
- Ratings are centered ~6.5; shows rate slightly higher than movies.
- Strongest predictors: **duration_minutes**, **log(votes)**, **release_year**.
- Country/age flags contribute modestly compared to the top numeric drivers.

## Models & Performance (Test)
**Regression (RandomForest):**
- MAE: ~0.75, RMSE: ~0.98, R²: ~0.26

**Classification (RandomForest, threshold 7.0):**
- Accuracy: ~0.77, Precision: ~0.85, Recall: ~0.51, ROC AUC: ~0.90, PR AUC: ~0.85

## Interpretability
- Permutation importance + PDP/ALE confirm:
  - Longer runtimes and more votes → higher predicted scores (diminishing returns).
  - Shows trend higher than movies.

## App
- Flask app with:
  - Score prediction UI.
  - **Content-based recommender** (by genres/description/country/age).
- Ready for Docker.

## Limitations & Next Steps
- No user personalization; purely content-based.
- Extend features (crew networks, release recency).
- Try gradient boosting (LightGBM/XGBoost) + calibrated probabilities.