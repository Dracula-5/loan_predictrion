import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "archive"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
PREDICTIONS_DIR = BASE_DIR / "predictions"
LOGS_DIR = BASE_DIR / "logs"

for _d in [MODEL_DIR, REPORTS_DIR, PREDICTIONS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train_u6lujuX_CVtuZ9i.csv"

MODEL_PATH = MODEL_DIR / "best_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
MODEL_COMPARISON_PATH = MODEL_DIR / "model_comparison.json"
SHAP_BACKGROUND_PATH = MODEL_DIR / "shap_background.pkl"
HISTORY_PATH = PREDICTIONS_DIR / "history.json"

TARGET_COLUMN = "Loan_Status"
LOAN_ID_COLUMN = "Loan_ID"

NUMERICAL_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "TotalIncome",
    "LoanAmountLog",
    "TotalIncomeLog",
    "EMI",
    "Income_to_Loan_Ratio",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5
SHAP_BACKGROUND_SIZE = 100

SECRET_KEY = os.environ.get("SECRET_KEY", "ldp-flask-secret-2024")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
PORT = int(os.environ.get("PORT", 5000))
HOST = os.environ.get("HOST", "127.0.0.1")
MAX_CONTENT_LENGTH = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"csv"}
