# System Architecture

```text
                     ┌──────────────────────────┐
                     │      Customer / User      │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │   Streamlit Dashboard    │
                     └────────────┬─────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
                 ▼                                 ▼
        ┌─────────────────┐               ┌─────────────────┐
        │ Dataset Explorer│               │ Prediction Form │
        └────────┬────────┘               └────────┬────────┘
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Data Validation /        │
                     │ Cleaning / Transformation│
                     └────────────┬─────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Feature Engineering      │
                     │ + Feature Selection      │
                     └────────────┬─────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Classification Models     │
                     │ LR / DT / KNN / NB / RF  │
                     │ SVM / Gradient Boosting  │
                     └────────────┬─────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Evaluation + Tuning      │
                     │ Accuracy / Recall / F1   │
                     │ ROC-AUC / Confusion      │
                     └────────────┬─────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Best Model + Probability │
                     └────────────┬─────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Risk Classification      │
                     │ Low / Medium / High      │
                     └────────────┬─────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Explainable AI           │
                     │ SHAP + Permutation       │
                     └────────────┬─────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Insights / Visualization │
                     └──────────────────────────┘
```

## Data-mining mapping

- Cleaning → data preprocessing
- Encoding and scaling → data transformation
- Engineered variables → feature construction
- SelectPercentile → feature selection/reduction
- Seven classifiers → classification
- Cross-validation/GridSearchCV → model optimization
- Metrics → model evaluation
- Permutation importance/SHAP → explainability
- Probability thresholds → customer risk segmentation
- Streamlit → applied analytics interface
