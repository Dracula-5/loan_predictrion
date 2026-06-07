import pandas as pd
import numpy as np


def load_raw_data(filepath) -> pd.DataFrame:
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Loan_ID" in df.columns:
        df.drop("Loan_ID", axis=1, inplace=True)

    mode_cols = [
        "Gender", "Married", "Dependents",
        "Self_Employed", "Loan_Amount_Term", "Credit_History",
    ]
    for col in mode_cols:
        if col in df.columns and df[col].isnull().any():
            mode = df[col].mode()
            if len(mode) > 0:
                df[col] = df[col].fillna(mode[0])

    if "LoanAmount" in df.columns and df["LoanAmount"].isnull().any():
        df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].astype(str).str.replace("3+", "3", regex=False)

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Loan_Status" in df.columns:
        df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
    return df
