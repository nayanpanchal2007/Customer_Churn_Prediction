from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [
    ROOT_DIR / "data" / "raw" / "customer_churn.csv",
    ROOT_DIR / "data" / "customer_churn.csv",
]
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
RESULT_DIR = REPORT_DIR / "results"

TARGET = "Churn"
ID_COLUMNS = ["customerID", "CustomerID", "customer_id"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

RISK_THRESHOLDS = {
    "low_max": 0.30,
    "medium_max": 0.60,
}

for directory in [MODEL_DIR, FIGURE_DIR, RESULT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
