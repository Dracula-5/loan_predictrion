import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanAmountLog"] = np.log1p(df["LoanAmount"])
    df["TotalIncomeLog"] = np.log1p(df["TotalIncome"])

    term_median = df["Loan_Amount_Term"].median()
    term_fallback = term_median if term_median > 0 else 360.0
    safe_term = df["Loan_Amount_Term"].replace(0, np.nan).fillna(term_fallback)
    df["EMI"] = df["LoanAmount"] / safe_term

    loan_median = df["LoanAmount"].median()
    loan_fallback = loan_median if loan_median > 0 else 1.0
    safe_loan = df["LoanAmount"].replace(0, np.nan).fillna(loan_fallback)
    df["Income_to_Loan_Ratio"] = df["TotalIncome"] / safe_loan

    return df
