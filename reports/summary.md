# Netflix IMDb Project — Reporting Summary

## Datasets

- **Strict table**: `model_strict.csv` — 3067 rows, 37 cols.
- **Combined table**: `model_combined.csv` — 5267 rows, 38 cols.

## Regression (IMDb score prediction)

- **Pipeline**: `tuned_randomforest_pipeline.joblib`  
- **Test MAE**: 0.750  
- **Test RMSE**: 0.975  
- **Test R²**: 0.260

**Figures**  
- Pred vs Actual: `reports/figures/reg_pred_vs_actual.png`  
- Residuals: `reports/figures/reg_residuals_hist.png`

## Classification (High vs Not-High)

- **Pipeline**: `clf_randomforest_tuned.joblib`  
- **Test Accuracy**: 0.772  
- **Precision**: 0.845  
- **Recall**: 0.514  
- **F1**: 0.639  
- **ROC-AUC**: 0.897  
- **PR-AUC**: 0.855

**Figures**  
- Confusion Matrix: `reports/figures/cls_confusion_matrix.png`  
- ROC: `reports/figures/cls_roc.png`  
- PR: `reports/figures/cls_pr.png`

## Notes & Caveats

- Votes (log and raw), duration, and release year are consistently among the strongest predictors.
- Classification favors precision at the default 0.5 threshold. Tune the threshold for higher recall if needed.
- Country/age categories have small standalone effects; use with caution to avoid spurious conclusions.
