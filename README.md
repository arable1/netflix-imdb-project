# Netflix IMDb Project

**Goal:** Practice EDA, build ML models to predict IMDb score, and ship a small app (recommendations + rating prediction).

## Data
- Source: Kaggle – “Netflix Movies and TV Shows” (CC0)
- File: `data/raw/netflix_titles.csv`
- To re-download (requires Kaggle CLI + API key):
  ```bash
  kaggle datasets download -d shivamb/netflix-shows -p data/raw
  unzip data/raw/*.zip -d data/raw



## Project Structure

netflix-imdb-project/
├── .gitignore
├── README.md
├── requirements.txt
├── environment.yml
├── .env
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 1_EDA.ipynb
│   ├── 2_Preprocessing.ipynb
│   ├── 3_Modeling_Regression.ipynb
│   └── 4_Modeling_Classification.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── modeling.py
│   └── utils.py
├── reports/
│   ├── figures/
│   └── final_report.md
└── app/
    ├── app.py
    ├── templates/
    └── static/