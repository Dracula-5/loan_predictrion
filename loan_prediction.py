# ============================================================
# Loan Default Prediction System
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("LOAN DEFAULT PREDICTION SYSTEM")
print("=" * 60)

# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading Dataset...")

df = pd.read_csv(
    "archive/train_u6lujuX_CVtuZ9i.csv"
)

print("\nDataset Loaded Successfully")
print("Shape :", df.shape)

# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nFirst Five Records")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing Values")
print(df.isnull().sum())

# ============================================================
# DATA CLEANING
# ============================================================

print("\nPerforming Data Cleaning...")

df.drop("Loan_ID", axis=1, inplace=True)

df["Gender"].fillna(
    df["Gender"].mode()[0],
    inplace=True
)

df["Married"].fillna(
    df["Married"].mode()[0],
    inplace=True
)

df["Dependents"].fillna(
    df["Dependents"].mode()[0],
    inplace=True
)

df["Self_Employed"].fillna(
    df["Self_Employed"].mode()[0],
    inplace=True
)

df["LoanAmount"].fillna(
    df["LoanAmount"].median(),
    inplace=True
)

df["Loan_Amount_Term"].fillna(
    df["Loan_Amount_Term"].mode()[0],
    inplace=True
)

df["Credit_History"].fillna(
    df["Credit_History"].mode()[0],
    inplace=True
)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating New Feature : Total Income")

df["TotalIncome"] = (
    df["ApplicantIncome"]
    + df["CoapplicantIncome"]
)

# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

print("\nGenerating EDA Graphs...")

plt.figure(figsize=(6,4))
sns.countplot(
    x="Loan_Status",
    data=df
)
plt.title("Loan Status Distribution")
plt.savefig("loan_status_distribution.png")
plt.close()

plt.figure(figsize=(6,4))
sns.countplot(
    x="Education",
    hue="Loan_Status",
    data=df
)
plt.title("Education vs Loan Status")
plt.savefig("education_vs_status.png")
plt.close()

plt.figure(figsize=(7,4))
sns.countplot(
    x="Property_Area",
    hue="Loan_Status",
    data=df
)
plt.title("Property Area vs Loan Status")
plt.savefig("property_area_vs_status.png")
plt.close()

plt.figure(figsize=(8,5))
sns.histplot(
    df["ApplicantIncome"],
    bins=30,
    kde=True
)
plt.title("Applicant Income Distribution")
plt.savefig("income_distribution.png")
plt.close()

plt.figure(figsize=(8,5))
sns.histplot(
    df["LoanAmount"],
    bins=30,
    kde=True
)
plt.title("Loan Amount Distribution")
plt.savefig("loan_amount_distribution.png")
plt.close()

# ============================================================
# LABEL ENCODING
# ============================================================

print("\nEncoding Categorical Variables...")

le = LabelEncoder()

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig("correlation_heatmap.png")

plt.close()

# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    "Loan_Status",
    axis=1
)

y = df["Loan_Status"]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\n" + "="*60)
print("LOGISTIC REGRESSION")
print("="*60)

lr = LogisticRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_acc = accuracy_score(
    y_test,
    lr_pred
)

print("Accuracy :", lr_acc)

print("\nConfusion Matrix")
print(confusion_matrix(
    y_test,
    lr_pred
))

print("\nClassification Report")
print(classification_report(
    y_test,
    lr_pred
))

# ============================================================
# DECISION TREE
# ============================================================

print("\n" + "="*60)
print("DECISION TREE")
print("="*60)

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(
    y_test,
    dt_pred
)

print("Accuracy :", dt_acc)

print(confusion_matrix(
    y_test,
    dt_pred
))

# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "="*60)
print("RANDOM FOREST")
print("="*60)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(
    y_test,
    rf_pred
)

print("Accuracy :", rf_acc)

print(confusion_matrix(
    y_test,
    rf_pred
))

# ============================================================
# KNN
# ============================================================

print("\n" + "="*60)
print("KNN")
print("="*60)

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(
    X_train,
    y_train
)

knn_pred = knn.predict(
    X_test
)

knn_acc = accuracy_score(
    y_test,
    knn_pred
)

print("Accuracy :", knn_acc)

# ============================================================
# SVM
# ============================================================

print("\n" + "="*60)
print("SUPPORT VECTOR MACHINE")
print("="*60)

svm = SVC()

svm.fit(
    X_train,
    y_train
)

svm_pred = svm.predict(
    X_test
)

svm_acc = accuracy_score(
    y_test,
    svm_pred
)

print("Accuracy :", svm_acc)

# ============================================================
# MODEL COMPARISON
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "SVM"
    ],
    "Accuracy": [
        lr_acc,
        dt_acc,
        rf_acc,
        knn_acc,
        svm_acc
    ]
})

print("\n")
print("="*60)
print("MODEL COMPARISON")
print("="*60)

print(results)

# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8,5))

sns.barplot(
    x="Model",
    y="Accuracy",
    data=results
)

plt.xticks(rotation=20)

plt.title(
    "Machine Learning Model Comparison"
)

plt.savefig(
    "model_comparison.png"
)

plt.close()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance":
        rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("="*60)
print("FEATURE IMPORTANCE")
print("="*60)

print(importance)

plt.figure(figsize=(10,6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title(
    "Feature Importance"
)

plt.savefig(
    "feature_importance.png"
)

plt.close()

# ============================================================
# BEST MODEL
# ============================================================

best_accuracy = results[
    "Accuracy"
].max()

best_model = results.loc[
    results["Accuracy"].idxmax(),
    "Model"
]

print("\n")
print("="*60)
print("FINAL RESULT")
print("="*60)

print(
    f"Best Model : {best_model}"
)

print(
    f"Best Accuracy : {best_accuracy:.4f}"
)

print("\nProject Execution Completed Successfully")

# ============================================================