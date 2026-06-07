# Loan Default Prediction System

A production-style machine learning web application that predicts loan eligibility using an ensemble of ML models with SHAP-based explainability, a Flask REST API, and a responsive frontend dashboard.

## Tech Stack

| Layer | Technologies |
|---|---|
| ML | scikit-learn, XGBoost, LightGBM, CatBoost, SHAP |
| Backend | Flask, Joblib, fpdf2 |
| Frontend | Bootstrap 5, Chart.js |
| Data | pandas, NumPy, seaborn, matplotlib |

## Project Structure

```
loan-default-prediction/
├── src/
│   ├── data/preprocessing.py       # Data loading and cleaning
│   ├── features/engineering.py     # Feature engineering
│   ├── models/
│   │   ├── trainer.py              # Model configs + hyperparameter tuning
│   │   └── evaluator.py            # Metrics, plots (ROC, PR, CM, LC)
│   └── utils/logger.py             # Logging setup
├── templates/                       # Jinja2 HTML templates
├── static/                          # CSS and JS
├── models/                          # Saved model artifacts (auto-created)
├── reports/                         # EDA and evaluation plots (auto-created)
├── predictions/                     # Prediction history JSON (auto-created)
├── archive/                         # Dataset CSVs
├── config.py                        # Centralized configuration
├── train.py                         # Training pipeline
├── app.py                           # Flask application
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Step 1 — Train the model:**
```bash
python train.py
```

This runs the full pipeline: EDA, feature engineering, trains 8 models (LR, DT, RF, KNN, SVM, XGBoost, LightGBM, CatBoost) with GridSearchCV/RandomizedSearchCV, evaluates all, selects the best by ROC-AUC, generates all report plots, and saves the model to `models/`.

**Step 2 — Start the web app:**
```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Features

- Single applicant prediction with SHAP-based explanation
- Batch prediction via CSV upload
- Model performance dashboard (ROC, PR curve, confusion matrix, learning curves, SHAP plots)
- Prediction history with PDF report download
- REST API (`/api/metrics`, `/api/history`, `/api/health`)

## Engineered Features

| Feature | Formula |
|---|---|
| TotalIncome | ApplicantIncome + CoapplicantIncome |
| LoanAmountLog | log1p(LoanAmount) |
| TotalIncomeLog | log1p(TotalIncome) |
| EMI | LoanAmount / Loan_Amount_Term |
| Income_to_Loan_Ratio | TotalIncome / LoanAmount |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check, model status |
| GET | `/api/metrics` | Best model metrics as JSON |
| GET | `/api/history` | Prediction history (paginated) |
| DELETE | `/api/history/<id>` | Delete a prediction record |
