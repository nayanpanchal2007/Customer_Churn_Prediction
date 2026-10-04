from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve

from .config import FIGURE_DIR, RESULT_DIR


def save_model_comparison_charts(comparison: pd.DataFrame):
    for metric in ["Accuracy", "F1", "ROC-AUC"]:
        ax = comparison[metric].sort_values().plot(
            kind="barh", figsize=(9, 5), title=f"{metric} Comparison"
        )
        ax.set_xlabel(metric)
        ax.set_ylabel("Model")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{metric.lower().replace('-', '_')}_comparison.png", dpi=160)
        plt.close()


def save_confusion_matrix(model, X_test, y_test, name="best_model"):
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{name}_confusion_matrix.png", dpi=160)
    plt.close()


def save_roc_curve(model, X_test, y_test, name="best_model"):
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label="Best Model")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{name}_roc_curve.png", dpi=160)
    plt.close()
