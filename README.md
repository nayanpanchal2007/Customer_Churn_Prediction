# Customer Churn Prediction and Analysis System Using Data Mining

**Subject:** Data Mining Techniques  
**Subject Code:** BE05000181  
**Academic Level:** B.E. Computer Engineering — Semester 5  
**Project Type:** GTU Problem Based Learning (PBL)

## Overview

This project is an end-to-end customer churn data-mining system. It covers data cleaning, transformation, EDA, feature engineering, feature selection through model-oriented analysis, classification, hyperparameter tuning, model evaluation, explainability, risk analysis, and an interactive Streamlit dashboard.

No experimental result is hardcoded. Metrics and feature importance are generated from the dataset supplied by the user.

## Dataset

Recommended dataset: IBM Telco Customer Churn / standard public Telco Customer Churn CSV.

Place the file at:

`data/raw/customer_churn.csv`

The standard public version contains 7,043 customer records and 21 columns, with `Churn` as the target.

## Project Structure

```text
customer-churn-prediction/
├── data/
│   ├── raw/
│   │   └── customer_churn.csv
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── hyperparameter_tuning.py
│   ├── explainability.py
│   └── prediction.py
├── app/
│   └── app.py
├── reports/
│   ├── figures/
│   └── results/
├── requirements.txt
└── run.py
```

## Installation

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Add Dataset

Copy the real dataset to:

```text
data/raw/customer_churn.csv
```

Do not fabricate rows or target labels.

## Train and Generate Analysis

```powershell
python run.py
```

This creates model comparison results, evaluation charts, confusion matrix, ROC curve, and permutation feature importance.

## Optional Hyperparameter Tuning

```powershell
python -m src.hyperparameter_tuning
```

## Start Dashboard

```powershell
streamlit run app/app.py
```

## Evaluation Strategy

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The automatic selection score emphasizes F1, Recall, and ROC-AUC rather than accuracy alone because missing likely churners can be costly in a retention setting.

## Explainability

Global explanations use permutation importance. Local explanations attempt SHAP and gracefully fall back when the runtime/model combination does not support a fast SHAP explanation.

Explainability indicates model behavior and associations; it must not be interpreted as proof of causation.

## Risk Thresholds

- Low: probability < 0.30
- Medium: 0.30–0.60
- High: > 0.60

These are configurable project thresholds, not universal industry standards.

## Academic Data-Mining Mapping

| Project Component | Data Mining Concept |
|---|---|
| Data cleaning | Data preprocessing |
| Missing-value handling | Data cleaning |
| Encoding/scaling | Data transformation |
| Feature engineering | Data transformation |
| Permutation importance | Feature analysis/reduction |
| Classification models | Classification |
| Probability prediction | Prediction |
| Cross-validation/tuning | Model optimization |
| Accuracy/Recall/F1/ROC-AUC | Model evaluation |
| EDA/interactive charts | Data visualization |
| SHAP/permutation importance | Explainable AI |
| Risk segmentation | Applied analytics |

## Important

The project intentionally does not include fabricated accuracy, churn rate, feature rankings, or business conclusions. Run the pipeline against the actual dataset before placing numerical results in the academic report.
