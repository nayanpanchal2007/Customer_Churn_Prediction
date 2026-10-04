from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the project root importable when Streamlit launches this file directly.
# This supports: streamlit run app/app.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import MODEL_DIR, RISK_THRESHOLDS, TARGET
from src.data_preprocessing import clean_dataframe, load_raw_data
from src.feature_engineering import engineer_features
from src.prediction import load_model, predict_customer
from src.explainability import local_explanation

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.metric-card { padding: 1rem; border-radius: 12px; background: #f6f8fb; }
.small-note { color: #667085; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return clean_dataframe(load_raw_data())


@st.cache_resource
def get_model():
    return load_model()


try:
    df = get_data()
    model = get_model()
except Exception as exc:
    st.error(str(exc))
    st.info("Train the model first with: python -m src.train_models")
    st.stop()

st.sidebar.title("Customer Churn")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Data Explorer", "EDA", "Model Performance",
     "Predict Churn", "Customer Risk", "Explainability"],
)

if page == "Dashboard":
    st.title("📊 Customer Churn Intelligence")
    st.caption("Data Mining Techniques — Customer Churn Prediction and Analysis System")

    total = len(df)
    churned = int(df[TARGET].sum())
    churn_rate = churned / total if total else 0
    avg_monthly = float(df["MonthlyCharges"].mean()) if "MonthlyCharges" in df else 0

    engineered = engineer_features(df.drop(columns=[TARGET]))
    try:
        probs = model.predict_proba(engineered)[:, 1]
        high_risk = int((probs > RISK_THRESHOLDS["medium_max"]).sum())
    except Exception:
        high_risk = 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Churn Rate", f"{churn_rate:.1%}")
    c4.metric("Avg Monthly Charges", f"{avg_monthly:,.2f}")
    c5.metric("High-Risk Customers", f"{high_risk:,}")

    left, right = st.columns(2)
    with left:
        fig = px.pie(df, names=TARGET, title="Churn Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if "Contract" in df:
            contract = pd.crosstab(df["Contract"], df[TARGET], normalize="index") * 100
            contract = contract.reset_index()
            contract.columns = ["Contract", "Stayed", "Churned"]
            fig = px.bar(
                contract, x="Contract", y=["Stayed", "Churned"],
                barmode="group", title="Churn Rate by Contract"
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Data Explorer":
    st.title("🔎 Data Explorer")
    columns = st.multiselect("Select columns", df.columns.tolist(), default=df.columns.tolist())
    filtered = df.copy()
    st.dataframe(filtered[columns], use_container_width=True, height=500)
    st.write("Summary statistics")
    st.dataframe(df[columns].describe(include="all").T, use_container_width=True)

elif page == "EDA":
    st.title("📈 Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=TARGET, color=TARGET, title="Churn Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "MonthlyCharges" in df.columns:
            fig = px.box(df, x=TARGET, y="MonthlyCharges", color=TARGET,
                         title="Monthly Charges vs Churn")
            st.plotly_chart(fig, use_container_width=True)

    for col in ["Contract", "PaymentMethod", "InternetService"]:
        if col in df.columns:
            grouped = pd.crosstab(df[col], df[TARGET], normalize="index") * 100
            grouped = grouped.reset_index()
            grouped.columns = [col, "Stayed", "Churned"]
            fig = px.bar(grouped, x=col, y=["Stayed", "Churned"],
                         barmode="group", title=f"Churn by {col}")
            st.plotly_chart(fig, use_container_width=True)

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        fig = px.imshow(numeric.corr(), text_auto=".2f", aspect="auto",
                        title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Performance":
    st.title("🤖 Model Performance")
    path = PROJECT_ROOT / "reports" / "results" / "model_comparison.csv"
    if path.exists():
        comparison = pd.read_csv(path, index_col=0)
        st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)
        metric = st.selectbox("Metric", ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
        chart = comparison[metric].sort_values().reset_index()
        chart.columns = ["Model", metric]
        fig = px.bar(chart, x=metric, y="Model", orientation="h", title=f"{metric} Comparison")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No evaluation results found. Run the training pipeline first.")

elif page == "Predict Churn":
    st.title("🎯 Predict Customer Churn")
    st.caption("Enter customer information. The model returns probability and a project-defined risk category.")

    # Inputs are aligned to the standard IBM Telco dataset.
    customer = {}
    c1, c2 = st.columns(2)
    with c1:
        customer["gender"] = st.selectbox("Gender", ["Male", "Female"])
        customer["SeniorCitizen"] = st.selectbox("Senior Citizen", [0, 1])
        customer["Partner"] = st.selectbox("Partner", ["Yes", "No"])
        customer["Dependents"] = st.selectbox("Dependents", ["Yes", "No"])
        customer["tenure"] = st.number_input("Tenure (months)", 0, 100, 12)
        customer["PhoneService"] = st.selectbox("Phone Service", ["Yes", "No"])
        customer["MultipleLines"] = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        customer["InternetService"] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        customer["OnlineSecurity"] = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        customer["OnlineBackup"] = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    with c2:
        customer["DeviceProtection"] = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        customer["TechSupport"] = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        customer["StreamingTV"] = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        customer["StreamingMovies"] = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        customer["Contract"] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        customer["PaperlessBilling"] = st.selectbox("Paperless Billing", ["Yes", "No"])
        customer["PaymentMethod"] = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        customer["MonthlyCharges"] = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        customer["TotalCharges"] = st.number_input("Total Charges", min_value=0.0, value=840.0)

    if st.button("PREDICT CHURN", type="primary", use_container_width=True):
        try:
            result = predict_customer(customer, model)
            probability = result["probability"]

            st.subheader("Prediction Result")
            a, b, c = st.columns(3)
            a.metric("Prediction", "Likely to Churn" if result["prediction"] else "Likely to Stay")
            b.metric("Churn Probability", f"{probability:.1%}")
            c.metric("Risk", result["risk"])

            st.progress(min(max(probability, 0.0), 1.0))

            local = local_explanation(model, engineer_features(pd.DataFrame([customer])))
            st.subheader("Why this prediction?")
            st.dataframe(local.head(8), use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

elif page == "Customer Risk":
    st.title("⚠️ Customer Risk Analysis")
    engineered = engineer_features(df.drop(columns=[TARGET]))
    probabilities = model.predict_proba(engineered)[:, 1]
    risk_df = df.copy()
    risk_df["Churn Probability"] = probabilities
    risk_df["Risk Level"] = pd.cut(
        risk_df["Churn Probability"],
        bins=[-0.001, 0.30, 0.60, 1.0],
        labels=["Low", "Medium", "High"],
    )
    level = st.multiselect("Risk filter", ["Low", "Medium", "High"],
                           default=["High", "Medium", "Low"])
    shown = risk_df[risk_df["Risk Level"].isin(level)].sort_values(
        "Churn Probability", ascending=False
    )
    cols = [c for c in ["customerID", "Churn Probability", "Risk Level",
                        "Contract", "tenure", "MonthlyCharges"] if c in shown.columns]
    st.dataframe(shown[cols], use_container_width=True, height=600)

elif page == "Explainability":
    st.title("🧠 Explainable AI")
    path = PROJECT_ROOT / "reports" / "results" / "permutation_feature_importance.csv"
    if path.exists():
        imp = pd.read_csv(path).head(15)
        fig = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature",
                     orientation="h", title="Global Permutation Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Permutation importance measures how much model performance changes when a feature is shuffled; it is not a causal effect.")
    else:
        st.warning("Run `python run.py` first to generate global feature importance.")

st.sidebar.divider()
st.sidebar.caption("Risk thresholds are project-configurable: Low < 0.30, Medium 0.30–0.60, High > 0.60.")
