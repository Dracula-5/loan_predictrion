import json
import os
import uuid
import sys
import threading
from io import BytesIO
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).parent))

import config
from src.features.engineering import engineer_features
from src.utils.logger import get_logger

try:
    import shap
    _SHAP = True
except ImportError:
    _SHAP = False

try:
    from fpdf import FPDF
    _FPDF = True
except ImportError:
    _FPDF = False

app = Flask(__name__)
_raw_origins = os.environ.get("CORS_ORIGINS", "*")
_cors_origins = [o.strip() for o in _raw_origins.split(",")] if "," in _raw_origins else _raw_origins
CORS(app, origins=_cors_origins)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

logger = get_logger("app", config.LOGS_DIR)

_model = None
_feature_names = None
_metrics = None
_model_comparison = None
_shap_explainer = None
_shap_model_type = None
_initialized = False
_init_lock = threading.Lock()
_history_lock = threading.Lock()


def _load_artifacts() -> bool:
    global _model, _feature_names, _metrics, _model_comparison

    if not config.MODEL_PATH.exists():
        logger.warning("No trained model found. Run train.py first.")
        return False

    _model = joblib.load(config.MODEL_PATH)
    _feature_names = joblib.load(config.FEATURE_NAMES_PATH)

    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH) as f:
            _metrics = json.load(f)

    if config.MODEL_COMPARISON_PATH.exists():
        with open(config.MODEL_COMPARISON_PATH) as f:
            _model_comparison = json.load(f)

    logger.info(f"Model loaded: {_metrics.get('model_name', 'unknown') if _metrics else 'unknown'}")
    return True


def _setup_shap():
    global _shap_explainer, _shap_model_type

    if not _SHAP or not config.SHAP_BACKGROUND_PATH.exists():
        return

    try:
        X_bg = joblib.load(config.SHAP_BACKGROUND_PATH)
        clf = _model.named_steps["model"]
        model_class = type(clf).__name__

        tree_models = {
            "RandomForestClassifier", "DecisionTreeClassifier",
            "XGBClassifier", "LGBMClassifier", "CatBoostClassifier",
        }

        if model_class in tree_models:
            _shap_explainer = shap.TreeExplainer(clf)
            _shap_model_type = "tree"
        elif model_class == "LogisticRegression":
            _shap_explainer = shap.LinearExplainer(clf, X_bg)
            _shap_model_type = "linear"
        else:
            _shap_explainer = None
            _shap_model_type = None

        logger.info(f"SHAP explainer ready ({_shap_model_type or 'unavailable'})")

    except Exception as exc:
        logger.warning(f"SHAP setup failed: {exc}")


def reload_artifacts():
    """Reload model artifacts. Used by tests after creating a test model."""
    global _initialized
    _load_artifacts()
    if _model is not None:
        _setup_shap()
    _initialized = True


@app.before_request
def _ensure_initialized():
    global _initialized
    if not _initialized:
        with _init_lock:
            if not _initialized:
                _load_artifacts()
                if _model is not None:
                    _setup_shap()
                _initialized = True


def _get_shap_contributions(X_preprocessed: np.ndarray) -> list:
    if _shap_explainer is None or _feature_names is None:
        return []
    try:
        sv = _shap_explainer.shap_values(X_preprocessed)
        if isinstance(sv, list):
            sv = sv[1]
        values = np.array(sv)
        if len(values.shape) == 2:
            values = values[0]

        contribs = [
            {"feature": _feature_names[i], "value": float(values[i])}
            for i in range(len(_feature_names))
        ]
        contribs.sort(key=lambda x: abs(x["value"]), reverse=True)
        return contribs[:8]

    except Exception as exc:
        logger.debug(f"SHAP inference failed: {exc}")
        return []


FIELD_TYPES = {
    "gender": str,
    "married": str,
    "dependents": str,
    "education": str,
    "self_employed": str,
    "applicant_income": float,
    "coapplicant_income": float,
    "loan_amount": float,
    "loan_amount_term": float,
    "credit_history": float,
    "property_area": str,
}

VALID_VALUES = {
    "gender": {"Male", "Female"},
    "married": {"Yes", "No"},
    "dependents": {"0", "1", "2", "3"},
    "education": {"Graduate", "Not Graduate"},
    "self_employed": {"Yes", "No"},
    "credit_history": {0.0, 1.0},
    "property_area": {"Urban", "Semiurban", "Rural"},
}


def _validate_input(data: dict) -> list:
    errors = []
    for field, ftype in FIELD_TYPES.items():
        if field not in data:
            errors.append(f"Missing field: {field}")
            continue
        try:
            converted = ftype(data[field])
        except (ValueError, TypeError):
            errors.append(f"Invalid type for '{field}': expected {ftype.__name__}")
            continue
        if field in VALID_VALUES and converted not in VALID_VALUES[field]:
            errors.append(f"Invalid value for '{field}': {data[field]}. Allowed: {sorted(str(v) for v in VALID_VALUES[field])}")
    if "applicant_income" in data:
        try:
            if float(data["applicant_income"]) < 0:
                errors.append("applicant_income must be non-negative")
        except (ValueError, TypeError):
            pass
    if "loan_amount" in data:
        try:
            if float(data["loan_amount"]) <= 0:
                errors.append("loan_amount must be positive")
        except (ValueError, TypeError):
            pass
    if "loan_amount_term" in data:
        try:
            if float(data["loan_amount_term"]) <= 0:
                errors.append("loan_amount_term must be positive")
        except (ValueError, TypeError):
            pass
    if "coapplicant_income" in data:
        try:
            if float(data["coapplicant_income"]) < 0:
                errors.append("coapplicant_income must be non-negative")
        except (ValueError, TypeError):
            pass
    return errors


def _parse_json_input(data: dict) -> pd.DataFrame:
    raw = {
        "Gender": str(data["gender"]),
        "Married": str(data["married"]),
        "Dependents": str(data["dependents"]).replace("+", ""),
        "Education": str(data["education"]),
        "Self_Employed": str(data["self_employed"]),
        "ApplicantIncome": float(data["applicant_income"]),
        "CoapplicantIncome": float(data["coapplicant_income"]),
        "LoanAmount": float(data["loan_amount"]),
        "Loan_Amount_Term": float(data["loan_amount_term"]),
        "Credit_History": float(data["credit_history"]),
        "Property_Area": str(data["property_area"]),
    }
    return engineer_features(pd.DataFrame([raw]))


def _load_history() -> list:
    if config.HISTORY_PATH.exists():
        with open(config.HISTORY_PATH) as f:
            return json.load(f)
    return []


def _save_to_history(record: dict):
    with _history_lock:
        history = _load_history()
        history.insert(0, record)
        config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)


def _generate_pdf(record: dict) -> BytesIO:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_fill_color(26, 47, 75)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 14, "Loan Eligibility Report", new_x="LMARGIN", new_y="NEXT", align="C", fill=True)
    pdf.ln(4)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Application ID : {record['id']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Generated      : {record['timestamp'][:19].replace('T', ' ')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Model          : {record.get('model', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    result = record.get("result", "Unknown")
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(22, 163, 74) if result == "Approved" else pdf.set_text_color(220, 38, 38)
    pdf.cell(0, 10, f"Decision: {result}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(3)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Risk Assessment", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Approval Probability : {record['approval_probability'] * 100:.1f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Risk Score           : {record['risk_score']} / 100", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    inp = record.get("input", {})
    field_map = {
        "gender": "Gender", "married": "Married", "dependents": "Dependents",
        "education": "Education", "self_employed": "Self Employed",
        "applicant_income": "Applicant Income", "coapplicant_income": "Coapplicant Income",
        "loan_amount": "Loan Amount (K)", "loan_amount_term": "Loan Term (months)",
        "credit_history": "Credit History", "property_area": "Property Area",
    }
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Applicant Details", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for key, label in field_map.items():
        val = inp.get(key, "-")
        pdf.cell(95, 6, f"  {label}", border="B")
        pdf.cell(95, 6, str(val), border="B", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, (
        "Disclaimer: This report is generated by an automated machine learning model "
        "for informational purposes only and does not constitute a formal credit decision."
    ))

    buf = BytesIO()
    buf.write(pdf.output())
    buf.seek(0)
    return buf


# ──────────────────────────── API Routes ────────────────────────────

@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
        "model": _metrics.get("model_name") if _metrics else None,
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/metrics")
def api_metrics():
    if _metrics is None:
        return jsonify({"error": "Model not trained. Run train.py first."}), 404
    return jsonify(_metrics)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if _model is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    errors = _validate_input(data)
    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    try:
        input_df = _parse_json_input(data)
        pred = int(_model.predict(input_df)[0])
        proba = _model.predict_proba(input_df)[0]
        approval_prob = float(proba[1])
        risk_score = int((1 - approval_prob) * 100)

        shap_contributions = []
        if _shap_explainer is not None:
            preprocessor = _model.named_steps["preprocessor"]
            X_proc = preprocessor.transform(input_df)
            shap_contributions = _get_shap_contributions(X_proc)

        prediction_id = str(uuid.uuid4())[:8].upper()
        record = {
            "id": prediction_id,
            "timestamp": datetime.now().isoformat(),
            "input": data,
            "result": "Approved" if pred == 1 else "Rejected",
            "approval_probability": round(approval_prob, 4),
            "risk_score": risk_score,
            "model": _metrics.get("model_name", "Unknown") if _metrics else "Unknown",
            "shap_contributions": shap_contributions,
        }
        _save_to_history(record)

        logger.info(f"Prediction [{prediction_id}]: {record['result']} (prob={approval_prob:.3f})")
        return jsonify(record), 201

    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        return jsonify({"error": "Internal server error during prediction."}), 500


@app.route("/api/result/<pid>")
def api_result(pid):
    history = _load_history()
    record = next((r for r in history if r["id"] == pid), None)
    if record is None:
        return jsonify({"error": "Record not found"}), 404
    return jsonify(record)


@app.route("/api/dashboard")
def api_dashboard():
    if _model is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    report_files = {
        name: (config.REPORTS_DIR / f"{name}.png").exists()
        for name in [
            "roc_curve", "pr_curve", "confusion_matrix", "learning_curve",
            "feature_importance", "model_comparison", "shap_summary", "shap_importance",
        ]
    }
    return jsonify({
        "metrics": _metrics,
        "model_comparison": _model_comparison,
        "report_files": report_files,
    })


@app.route("/api/history")
def api_history():
    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(200, max(1, int(request.args.get("per_page", 20))))
    except (ValueError, TypeError):
        return jsonify({"error": "page and per_page must be integers"}), 400
    records = _load_history()
    start = (page - 1) * per_page
    return jsonify({
        "records": records[start: start + per_page],
        "total": len(records),
        "page": page,
        "per_page": per_page,
    })


@app.route("/api/history/<pid>", methods=["DELETE"])
def api_delete_prediction(pid):
    history = _load_history()
    updated = [r for r in history if r["id"] != pid]
    if len(updated) == len(history):
        return jsonify({"error": "Record not found"}), 404
    with open(config.HISTORY_PATH, "w") as f:
        json.dump(updated, f, indent=2)
    return jsonify({"message": "Deleted"})


@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    with open(config.HISTORY_PATH, "w") as f:
        json.dump([], f)
    return jsonify({"message": "History cleared"})


@app.route("/api/batch", methods=["POST"])
def api_batch():
    if _model is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted"}), 400

    try:
        df = pd.read_csv(file)

        required_cols = {
            "Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History", "Property_Area",
        }
        missing = required_cols - set(df.columns)
        if missing:
            return jsonify({"error": f"Missing columns: {', '.join(sorted(missing))}"}), 400

        df["Dependents"] = df["Dependents"].astype(str).str.replace("+", "", regex=False)
        df = engineer_features(df)

        input_df = df[_feature_names]
        preds = _model.predict(input_df)
        probas = _model.predict_proba(input_df)[:, 1]

        df["Prediction"] = ["Approved" if p == 1 else "Rejected" for p in preds]
        df["Approval_Probability"] = probas.round(3)
        df["Risk_Score"] = ((1 - probas) * 100).astype(int)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"batch_{ts}.csv"
        df.to_csv(config.PREDICTIONS_DIR / out_name, index=False)

        display_cols = [c for c in [
            "Gender", "Education", "LoanAmount", "Credit_History",
            "Prediction", "Approval_Probability", "Risk_Score",
        ] if c in df.columns]

        summary = {
            "total": len(df),
            "approved": int((df["Prediction"] == "Approved").sum()),
            "rejected": int((df["Prediction"] == "Rejected").sum()),
        }

        logger.info(f"Batch prediction: {len(df)} records, {summary['approved']} approved")

        return jsonify({
            "summary": summary,
            "results": df[display_cols].to_dict("records"),
            "download_url": f"/api/download_batch/{out_name}",
        })

    except Exception as exc:
        logger.error(f"Batch error: {exc}")
        return jsonify({"error": f"Error processing file: {str(exc)}"}), 500


@app.route("/api/download/<pid>")
def api_download_report(pid):
    if not _FPDF:
        return jsonify({"error": "fpdf2 not installed"}), 501

    history = _load_history()
    record = next((r for r in history if r["id"] == pid), None)
    if record is None:
        return jsonify({"error": "Record not found"}), 404

    buf = _generate_pdf(record)
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"loan_report_{pid}.pdf",
    )


@app.route("/api/download_batch/<filename>")
def api_download_batch(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        return jsonify({"error": "Invalid filename"}), 400
    if not (config.PREDICTIONS_DIR / safe_name).exists():
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(config.PREDICTIONS_DIR, safe_name, as_attachment=True)


@app.route("/api/reports/<path:filename>")
def api_serve_report(filename):
    return send_from_directory(config.REPORTS_DIR, filename)


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum upload size is 10 MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
