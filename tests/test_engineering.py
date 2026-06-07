import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineering import engineer_features


def _base_df(n=10):
    return pd.DataFrame({
        "ApplicantIncome": np.full(n, 5000.0),
        "CoapplicantIncome": np.full(n, 2000.0),
        "LoanAmount": np.full(n, 150.0),
        "Loan_Amount_Term": np.full(n, 360.0),
    })


class TestEngineerFeatures:
    def test_creates_total_income(self):
        df = _base_df()
        result = engineer_features(df)
        assert "TotalIncome" in result.columns
        assert result["TotalIncome"].iloc[0] == pytest.approx(7000.0)

    def test_total_income_formula(self):
        df = pd.DataFrame({
            "ApplicantIncome": [4000.0, 8000.0],
            "CoapplicantIncome": [1000.0, 3000.0],
            "LoanAmount": [100.0, 200.0],
            "Loan_Amount_Term": [360.0, 180.0],
        })
        result = engineer_features(df)
        assert result["TotalIncome"].tolist() == pytest.approx([5000.0, 11000.0])

    def test_creates_log_features(self):
        df = _base_df()
        result = engineer_features(df)
        assert "LoanAmountLog" in result.columns
        assert "TotalIncomeLog" in result.columns

    def test_log_features_non_negative(self):
        df = _base_df()
        result = engineer_features(df)
        assert (result["LoanAmountLog"] >= 0).all()
        assert (result["TotalIncomeLog"] >= 0).all()

    def test_creates_emi(self):
        df = _base_df()
        result = engineer_features(df)
        assert "EMI" in result.columns
        expected = 150.0 / 360.0
        assert result["EMI"].iloc[0] == pytest.approx(expected)

    def test_zero_term_no_division_error(self):
        df = _base_df()
        df["Loan_Amount_Term"] = 0.0
        result = engineer_features(df)
        assert not result["EMI"].isnull().any()
        assert not np.isinf(result["EMI"]).any()

    def test_creates_income_to_loan_ratio(self):
        df = _base_df()
        result = engineer_features(df)
        assert "Income_to_Loan_Ratio" in result.columns
        expected = 7000.0 / 150.0
        assert result["Income_to_Loan_Ratio"].iloc[0] == pytest.approx(expected, rel=1e-3)

    def test_zero_loan_no_division_error(self):
        df = _base_df()
        df["LoanAmount"] = 0.0
        result = engineer_features(df)
        assert not result["Income_to_Loan_Ratio"].isnull().any()
        assert not np.isinf(result["Income_to_Loan_Ratio"]).any()

    def test_does_not_mutate_input(self):
        df = _base_df()
        cols_before = set(df.columns)
        engineer_features(df)
        assert set(df.columns) == cols_before

    def test_original_columns_preserved(self):
        df = _base_df()
        result = engineer_features(df)
        original_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
        for col in original_cols:
            assert col in result.columns

    def test_output_has_five_new_columns(self):
        df = _base_df()
        result = engineer_features(df)
        new_cols = {"TotalIncome", "LoanAmountLog", "TotalIncomeLog", "EMI", "Income_to_Loan_Ratio"}
        assert new_cols.issubset(set(result.columns))
