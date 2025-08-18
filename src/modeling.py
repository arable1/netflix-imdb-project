import pandas as pd
from .utils import load_joblib, load_json

# filenames produced in your notebooks
REG_PATH = "tuned_randomforest_pipeline.joblib"
CLF_PATH = "clf_randomforest_tuned.joblib"
FEATURES_JSON = "features_used.json"   # saved in artifacts/ during nb4

class ModelBundle:
    def __init__(self):
        feats = load_json(FEATURES_JSON)
        self.numeric_cols = feats.get("numeric_cols", [])
        self.categorical_cols = feats.get("categorical_cols", [])
        self.reg = load_joblib(REG_PATH)
        self.clf = load_joblib(CLF_PATH)

    def predict_score(self, df_row: pd.DataFrame) -> float:
        # Pipeline handles preprocessing
        return float(self.reg.predict(df_row)[0])

    def predict_high(self, df_row: pd.DataFrame) -> dict:
        proba = self.clf.predict_proba(df_row)[0, 1]
        label = int(proba >= 0.5)
        return {"proba_high": float(proba), "label": label}