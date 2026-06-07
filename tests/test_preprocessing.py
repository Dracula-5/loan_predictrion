import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import clean_data, encode_target, load_raw_data


def _make_raw_df(n=20):
    return pd.DataFrame({
        "Loan_ID": [f"LP{i:03d}" for i in range(n)],
        "Gender": ["Male"] * n,
        "Married": ["Yes"] * n,
        "Dependents": ["0"] * (n - 2) + ["3+", None],
        "Education": ["Graduate"] * n,
        "Self_Employed": ["No"] * (n - 1) + [None],
        "ApplicantIncome": np.random.randint(3000, 10000, n).astype(float),
        "CoapplicantIncome": np.random.randint(0, 5000, n).astype(float),
        "LoanAmount": np.random.randint(50, 300, n).astype(float),
        "Loan_Amount_Term": [360.0] * (n - 1) + [None],
        "Credit_History": [1.0] * (n - 1) + [None],
        "Property_Area": ["Urban"] * n,
        "Loan_Status": ["Y"] * (n // 2) + ["N"] * (n - n // 2),
    })


class TestCleanData:
    def test_drops_loan_id(self):
        df = _make_raw_df()
        result = clean_data(df)
        assert "Loan_ID" not in result.columns

    def test_no_missing_values_after_clean(self):
        df = _make_raw_df()
        result = clean_data(df)
        assert result.isnull().sum().sum() == 0

    def test_does_not_mutate_input(self):
        df = _make_raw_df()
        original_nulls = df.isnull().sum().sum()
        clean_data(df)
        assert df.isnull().sum().sum() == original_nulls

    def test_dependents_3plus_converted(self):
        df = _make_raw_df()
        result = clean_data(df)
        assert "3+" not in result["Dependents"].values
        assert "3" in result["Dependents"].values

    def test_loan_amount_filled_with_median(self):
        df = _make_raw_df()
        df.loc[0, "LoanAmount"] = np.nan
        expected_median = df["LoanAmount"].median()  # median of remaining 19 values, same as clean_data computes
        result = clean_data(df)
        assert result.loc[0, "LoanAmount"] == pytest.approx(expected_median, rel=1e-3)

    def test_preserves_row_count(self):
        df = _make_raw_df(30)
        result = clean_data(df)
        assert len(result) == 30

    def test_loan_status_retained(self):
        df = _make_raw_df()
        result = clean_data(df)
        assert "Loan_Status" in result.columns


class TestEncodeTarget:
    def test_y_maps_to_1(self):
        df = pd.DataFrame({"Loan_Status": ["Y", "N", "Y"]})
        result = encode_target(df)
        assert list(result["Loan_Status"]) == [1, 0, 1]

    def test_n_maps_to_0(self):
        df = pd.DataFrame({"Loan_Status": ["N", "N"]})
        result = encode_target(df)
        assert list(result["Loan_Status"]) == [0, 0]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"Loan_Status": ["Y", "N"]})
        encode_target(df)
        assert df["Loan_Status"].tolist() == ["Y", "N"]

    def test_noop_without_loan_status_column(self):
        df = pd.DataFrame({"Income": [5000, 6000]})
        result = encode_target(df)
        assert "Loan_Status" not in result.columns

    def test_result_is_integer_dtype(self):
        df = pd.DataFrame({"Loan_Status": ["Y", "N", "Y", "N"]})
        result = encode_target(df)
        assert result["Loan_Status"].dtype in [int, "int64", "int32"]
