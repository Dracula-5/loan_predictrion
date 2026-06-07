import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.features.engineering import engineer_features
from src.models.trainer import build_preprocessor


@pytest.fixture(scope="session", autouse=True)
def trained_test_model():
    """
    Builds and persists a minimal RandomForest pipeline so API tests
    can run without needing a full train.py run.
    """
    np.random.seed(42)
    n = 80

    df = pd.DataFrame({
        "Gender": np.random.choice(["Male", "Female"], n),
        "Married": np.random.choice(["Yes", "No"], n),
        "Dependents": np.random.choice(["0", "1", "2", "3"], n),
        "Education": np.random.choice(["Graduate", "Not Graduate"], n),
        "Self_Employed": np.random.choice(["Yes", "No"], n),
        "ApplicantIncome": np.random.randint(2000, 12000, n).astype(float),
        "CoapplicantIncome": np.random.randint(0, 5000, n).astype(float),
        "LoanAmount": np.random.randint(50, 400, n).astype(float),
        "Loan_Amount_Term": np.random.choice([120.0, 180.0, 240.0, 360.0], n),
        "Credit_History": np.random.choice([0.0, 1.0], n),
        "Property_Area": np.random.choice(["Urban", "Semiurban", "Rural"], n),
    })
    df = engineer_features(df)

    num_feats = [f for f in config.NUMERICAL_FEATURES if f in df.columns]
    cat_feats = [f for f in config.CATEGORICAL_FEATURES if f in df.columns]
    feature_names = num_feats + cat_feats

    X = df[feature_names]
    y = np.random.choice([0, 1], n, p=[0.35, 0.65])

    preprocessor = build_preprocessor(num_feats, cat_feats)
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", clf)])
    pipeline.fit(X, y)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    joblib.dump(feature_names, config.FEATURE_NAMES_PATH)

    metrics = {
        "model_name": "RandomForest",
        "accuracy": 0.75,
        "roc_auc": 0.79,
        "precision": 0.73,
        "recall": 0.70,
        "f1": 0.71,
        "cv_auc_mean": 0.77,
        "cv_auc_std": 0.04,
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f)

    comparison = [{
        "model": "RandomForest",
        "accuracy": 0.75,
        "roc_auc": 0.79,
        "precision": 0.73,
        "recall": 0.70,
        "f1": 0.71,
        "cv_auc_mean": 0.77,
        "cv_auc_std": 0.04,
        "is_best": True,
    }]
    with open(config.MODEL_COMPARISON_PATH, "w") as f:
        json.dump(comparison, f)

    yield pipeline


@pytest.fixture(scope="session")
def flask_client(trained_test_model):
    import app as flask_app
    flask_app.reload_artifacts()
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as client:
        yield client


VALID_PAYLOAD = {
    "gender": "Male",
    "married": "Yes",
    "dependents": "1",
    "education": "Graduate",
    "self_employed": "No",
    "applicant_income": 6000.0,
    "coapplicant_income": 2000.0,
    "loan_amount": 150.0,
    "loan_amount_term": 360.0,
    "credit_history": 1.0,
    "property_area": "Semiurban",
}
