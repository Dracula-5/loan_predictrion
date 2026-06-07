import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import VALID_PAYLOAD


class TestHealthEndpoint:
    def test_returns_200(self, flask_client):
        r = flask_client.get("/api/health")
        assert r.status_code == 200

    def test_has_status_ok(self, flask_client):
        data = json.loads(flask_client.get("/api/health").data)
        assert data["status"] == "ok"

    def test_model_loaded_true(self, flask_client):
        data = json.loads(flask_client.get("/api/health").data)
        assert data["model_loaded"] is True

    def test_has_timestamp(self, flask_client):
        data = json.loads(flask_client.get("/api/health").data)
        assert "timestamp" in data


class TestMetricsEndpoint:
    def test_returns_200(self, flask_client):
        r = flask_client.get("/api/metrics")
        assert r.status_code == 200

    def test_has_required_keys(self, flask_client):
        data = json.loads(flask_client.get("/api/metrics").data)
        required = {"model_name", "accuracy", "roc_auc", "precision", "recall", "f1"}
        assert required.issubset(set(data.keys()))

    def test_scores_in_range(self, flask_client):
        data = json.loads(flask_client.get("/api/metrics").data)
        for key in ["accuracy", "roc_auc", "precision", "recall", "f1"]:
            assert 0.0 <= data[key] <= 1.0


class TestPredictEndpoint:
    def test_valid_request_returns_201(self, flask_client):
        r = flask_client.post(
            "/api/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        assert r.status_code == 201

    def test_response_has_result(self, flask_client):
        r = flask_client.post(
            "/api/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        data = json.loads(r.data)
        assert data["result"] in ("Approved", "Rejected")

    def test_response_has_probability(self, flask_client):
        r = flask_client.post(
            "/api/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        data = json.loads(r.data)
        assert 0.0 <= data["approval_probability"] <= 1.0

    def test_response_has_risk_score(self, flask_client):
        r = flask_client.post(
            "/api/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        data = json.loads(r.data)
        assert 0 <= data["risk_score"] <= 100

    def test_response_has_prediction_id(self, flask_client):
        r = flask_client.post(
            "/api/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        data = json.loads(r.data)
        assert "id" in data and len(data["id"]) > 0

    def test_missing_field_returns_400(self, flask_client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "gender"}
        r = flask_client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_invalid_gender_returns_400(self, flask_client):
        payload = {**VALID_PAYLOAD, "gender": "Unknown"}
        r = flask_client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_invalid_credit_history_returns_400(self, flask_client):
        payload = {**VALID_PAYLOAD, "credit_history": 5.0}
        r = flask_client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_non_json_body_returns_400(self, flask_client):
        r = flask_client.post("/api/predict", data="not json", content_type="text/plain")
        assert r.status_code == 400

    def test_prediction_is_stored_in_history(self, flask_client):
        r = flask_client.post(
            "/api/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        pid = json.loads(r.data)["id"]
        history_r = flask_client.get("/api/history")
        history = json.loads(history_r.data)["records"]
        ids = [rec["id"] for rec in history]
        assert pid in ids


class TestResultEndpoint:
    def test_returns_record_after_prediction(self, flask_client):
        r = flask_client.post(
            "/api/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        pid = json.loads(r.data)["id"]
        result_r = flask_client.get(f"/api/result/{pid}")
        assert result_r.status_code == 200
        data = json.loads(result_r.data)
        assert data["id"] == pid

    def test_unknown_id_returns_404(self, flask_client):
        r = flask_client.get("/api/result/NOTEXIST")
        assert r.status_code == 404


class TestHistoryEndpoints:
    def test_get_history_returns_200(self, flask_client):
        r = flask_client.get("/api/history")
        assert r.status_code == 200

    def test_response_has_records_and_total(self, flask_client):
        data = json.loads(flask_client.get("/api/history").data)
        assert "records" in data
        assert "total" in data
        assert isinstance(data["records"], list)

    def test_delete_nonexistent_returns_404(self, flask_client):
        r = flask_client.delete("/api/history/BADID")
        assert r.status_code == 404

    def test_delete_existing_record(self, flask_client):
        pred_r = flask_client.post(
            "/api/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        pid = json.loads(pred_r.data)["id"]
        del_r = flask_client.delete(f"/api/history/{pid}")
        assert del_r.status_code == 200
        result_r = flask_client.get(f"/api/result/{pid}")
        assert result_r.status_code == 404


class TestDashboardEndpoint:
    def test_returns_200(self, flask_client):
        r = flask_client.get("/api/dashboard")
        assert r.status_code == 200

    def test_has_metrics_and_comparison(self, flask_client):
        data = json.loads(flask_client.get("/api/dashboard").data)
        assert "metrics" in data
        assert "model_comparison" in data
        assert "report_files" in data
