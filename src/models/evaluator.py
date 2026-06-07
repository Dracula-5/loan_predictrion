import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import learning_curve

plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False})


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"],
        ax=ax, linewidths=0.5,
    )
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_roc_curve(model, X_test, y_test, model_name, save_path):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="lightgray", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_pr_curve(model, X_test, y_test, model_name, save_path):
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_learning_curve(model, X, y, save_path, cv=5):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=cv,
        scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
    )

    tr_mean = train_scores.mean(axis=1)
    tr_std = train_scores.std(axis=1)
    va_mean = val_scores.mean(axis=1)
    va_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, tr_mean, "o-", color="steelblue", label="Training Score")
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="steelblue")
    ax.plot(train_sizes, va_mean, "o-", color="darkorange", label="Validation Score")
    ax.fill_between(train_sizes, va_mean - va_std, va_mean + va_std, alpha=0.15, color="darkorange")
    ax.set_xlabel("Training Examples", fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("Learning Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_feature_importance(feature_names, importances, save_path, top_n=15):
    paired = sorted(zip(importances, feature_names), reverse=True)[:top_n]
    vals, names = zip(*paired)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(names)))
    bars = ax.barh(range(len(names)), vals, color=colors[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_model_comparison(comparison: list, save_path):
    names = [r["model"] for r in comparison]
    auc_scores = [r["roc_auc"] for r in comparison]
    acc_scores = [r["accuracy"] for r in comparison]
    f1_scores = [r["f1"] for r in comparison]

    x = np.arange(len(names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, auc_scores, width, label="ROC-AUC", color="steelblue")
    ax.bar(x, acc_scores, width, label="Accuracy", color="darkorange")
    ax.bar(x + width, f1_scores, width, label="F1 Score", color="mediumseagreen")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
