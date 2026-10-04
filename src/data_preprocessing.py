from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_CANDIDATES, RANDOM_STATE, TARGET, TEST_SIZE


def locate_dataset() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Customer churn dataset not found. Place the CSV at "
        "data/raw/customer_churn.csv."
    )


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path else locate_dataset()
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("The dataset is empty.")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Normalize column names without changing their semantic names unnecessarily.
    data.columns = [str(c).strip() for c in data.columns]

    # Remove exact duplicates.
    data = data.drop_duplicates().reset_index(drop=True)

    if TARGET not in data.columns:
        raise ValueError(f"Required target column '{TARGET}' was not found.")

    # Common IBM Telco issue: TotalCharges can contain blank strings.
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # Convert target to a clean binary representation.
    # Use string conversion rather than checking only for dtype == "object":
    # newer pandas versions may use StringDtype / Arrow-backed strings.
    target_text = data[TARGET].astype("string").str.strip().str.lower()

    target_map = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "y": 1,
        "n": 0,
    }

    # Preserve already-numeric 0/1 targets.
    numeric_target = pd.to_numeric(data[TARGET], errors="coerce")
    mapped_target = target_text.map(target_map)

    data[TARGET] = mapped_target.where(
        mapped_target.notna(),
        numeric_target
    )

    # Any remaining value is invalid and must not silently become a class.
    invalid_target = data[TARGET].isna()
    if invalid_target.any():
        invalid_values = (
            df.loc[invalid_target, TARGET]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"Invalid or missing values found in target column '{TARGET}': "
            f"{invalid_values}. Expected Yes/No or 0/1."
        )

    data[TARGET] = data[TARGET].astype(int)

    # Strip whitespace from object columns and convert blank strings to NaN.
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype(str).str.strip()
        data[col] = data[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    return data


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Customer ID is an identifier, not a predictive feature.
    id_cols = [c for c in X.columns if c in {"customerID", "CustomerID", "customer_id"}]
    X = X.drop(columns=id_cols, errors="ignore")
    return X, y


def train_test_data(df: pd.DataFrame):
    X, y = split_features_target(df)
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
