from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIGURE_DIR
from .data_preprocessing import clean_dataframe, load_raw_data


def run_eda():
    df = clean_dataframe(load_raw_data())
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "churn_rate": float(df["Churn"].mean()),
    }

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn")
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "churn_distribution.png", dpi=160)
    plt.close()

    for col in ["Contract", "InternetService", "PaymentMethod"]:
        if col in df.columns:
            plt.figure(figsize=(9, 5))
            sns.countplot(data=df, x=col, hue="Churn")
            plt.title(f"Churn by {col}")
            plt.xlabel(col)
            plt.ylabel("Number of Customers")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / f"churn_by_{col.lower()}.png", dpi=160)
            plt.close()

    if "tenure" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="Churn", y="tenure")
        plt.title("Tenure by Churn Status")
        plt.xlabel("Churn")
        plt.ylabel("Tenure (months)")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "tenure_vs_churn.png", dpi=160)
        plt.close()

    if "MonthlyCharges" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
        plt.title("Monthly Charges by Churn Status")
        plt.xlabel("Churn")
        plt.ylabel("Monthly Charges")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "monthly_charges_vs_churn.png", dpi=160)
        plt.close()

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        plt.figure(figsize=(11, 8))
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0)
        plt.title("Numerical Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "correlation_matrix.png", dpi=160)
        plt.close()

    return summary


if __name__ == "__main__":
    print(run_eda())
