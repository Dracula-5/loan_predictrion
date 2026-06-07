import json
import warnings
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.data.preprocessing import load_raw_data, clean_data, encode_target
from src.features.engineering import engineer_features
from src.models.trainer import build_preprocessor, train_all_models
from src.models.evaluator import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_learning_curve,
    plot_feature_importance,
    plot_model_comparison,
)
from src.utils.logger import get_logger

try:
    import shap
    _SHAP = True
except ImportError:
    _SHAP = False


def _run_eda(df: pd.DataFrame, reports_dir: Path, logger):
    logger.info("Generating EDA plots...")

    plt.rcParams.update({"figure.dpi": 100})
    palette = {"Y": "#2563eb", "N": "#ef4444"}

    fig, ax = plt.subplots(figsize=(6, 4))
    df_eda = df.copy()
    if "Loan_Status" in df_eda.columns and df_eda["Loan_Status"].dtype in [int, float]:
        df_eda["Loan_Status"] = df_eda["Loan_Status"].map({1: "Y", 0: "N"})
    counts = df_eda["Loan_Status"].value_counts()
    ax.bar(["Rejected", "Approved"], [counts.get("N", 0), counts.get("Y", 0)],
           color=["#ef4444", "#2563eb"])
    ax.set_title("Loan Status Distribution", fontweight="bold")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig(reports_dir / "eda_loan_status.png", bbox_inches="tight")
    plt.close()

    cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Credit_History"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        if col in df_eda.columns:
            temp = df_eda.groupby([col, "Loan_Status"]).size().unstack(fill_value=0)
            temp.plot(kind="bar", ax=axes[i], color=["#ef4444", "#2563eb"], rot=30)
            axes[i].set_title(f"{col} vs Loan Status", fontweight="bold", fontsize=10)
            axes[i].set_xlabel("")
            axes[i].legend(["Rejected", "Approved"], fontsize=8)
            axes[i].tick_params(labelsize=8)
    plt.suptitle("Categorical Features vs Loan Status", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(reports_dir / "eda_categorical.png", bbox_inches="tight")
    plt.close()

    num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        if col in df.columns:
            axes[i].hist(df[col].dropna(), bins=35, color="steelblue", edgecolor="white", alpha=0.8)
            axes[i].set_title(f"{col} Distribution", fontweight="bold", fontsize=10)
            axes[i].set_xlabel(col, fontsize=9)
            axes[i].set_ylabel("Count", fontsize=9)
    plt.suptitle("Numerical Feature Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(reports_dir / "eda_distributions.png", bbox_inches="tight")
    plt.close()

    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(12, 9))
        mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
        sns.heatmap(
            numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, ax=ax, linewidths=0.3, annot_kws={"size": 8},
        )
        ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(reports_dir / "eda_correlation.png", bbox_inches="tight")
        plt.close()

    logger.info("EDA plots saved to reports/")


def _compute_shap(pipeline, X_train, X_test, model_name, feature_names, reports_dir, logger):
    logger.info("Computing SHAP values...")
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["model"]

        X_train_proc = preprocessor.transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        model_class = type(clf).__name__
        tree_models = {
            "RandomForestClassifier", "DecisionTreeClassifier",
            "ExtraTreesClassifier", "GradientBoostingClassifier",
            "XGBClassifier", "LGBMClassifier", "CatBoostClassifier",
        }
        linear_models = {"LogisticRegression", "LinearSVC"}

        if model_class in tree_models:
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_test_proc)
            if isinstance(sv, list):
                sv = sv[1]
        elif model_class in linear_models:
            explainer = shap.LinearExplainer(clf, X_train_proc)
            sv = explainer.shap_values(X_test_proc)
        else:
            n_bg = min(50, X_train_proc.shape[0])
            bg = shap.sample(X_train_proc, n_bg)
            explainer = shap.KernelExplainer(clf.predict_proba, bg)
            sv = explainer.shap_values(X_test_proc[:30])
            if isinstance(sv, list):
                sv = sv[1]

        shap.summary_plot(sv, X_test_proc, feature_names=feature_names, show=False, plot_size=(10, 6))
        plt.title(f"SHAP Summary — {model_name}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(reports_dir / "shap_summary.png", bbox_inches="tight")
        plt.close("all")

        shap.summary_plot(sv, X_test_proc, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance — {model_name}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(reports_dir / "shap_importance.png", bbox_inches="tight")
        plt.close("all")

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean_SHAP": np.abs(sv).mean(axis=0),
        }).sort_values("Mean_SHAP", ascending=False)
        shap_df.to_json(reports_dir / "shap_importance.json", orient="records", indent=2)

        logger.info("SHAP analysis complete")

    except Exception as exc:
        logger.warning(f"SHAP computation failed: {exc}")


def main():
    logger = get_logger("train", config.LOGS_DIR)
    logger.info("=" * 60)
    logger.info("LOAN DEFAULT PREDICTION — TRAINING PIPELINE")
    logger.info("=" * 60)

    df = load_raw_data(config.TRAIN_FILE)
    logger.info(f"Dataset loaded: {df.shape}")

    df = clean_data(df)
    df = engineer_features(df)

    _run_eda(df, config.REPORTS_DIR, logger)

    df = encode_target(df)

    X = df.drop(config.TARGET_COLUMN, axis=1)
    y = df[config.TARGET_COLUMN]

    numerical_features = [f for f in config.NUMERICAL_FEATURES if f in X.columns]
    categorical_features = [f for f in config.CATEGORICAL_FEATURES if f in X.columns]
    feature_names = numerical_features + categorical_features

    logger.info(f"Features: {len(feature_names)} ({len(numerical_features)} numerical, {len(categorical_features)} categorical)")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    preprocessor = build_preprocessor(numerical_features, categorical_features)

    trained = train_all_models(
        X_train, y_train,
        preprocessor,
        cv_folds=config.CV_FOLDS,
        logger=logger,
    )

    logger.info("Evaluating all models on test set...")
    all_metrics = {}
    for name, info in trained.items():
        m = evaluate_model(info["pipeline"], X_test, y_test)
        m["cv_auc_mean"] = info["cv_auc_mean"]
        m["cv_auc_std"] = info["cv_auc_std"]
        m["best_params"] = info["best_params"]
        all_metrics[name] = m
        logger.info(
            f"  {name:20s} | AUC: {m['roc_auc']:.4f} | Acc: {m['accuracy']:.4f} "
            f"| F1: {m['f1']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f}"
        )

    best_name = max(all_metrics, key=lambda n: all_metrics[n]["roc_auc"])
    best_pipeline = trained[best_name]["pipeline"]
    logger.info(f"Best model: {best_name} (AUC: {all_metrics[best_name]['roc_auc']:.4f})")

    cm_array = np.array(all_metrics[best_name]["confusion_matrix"])
    plot_confusion_matrix(cm_array, config.REPORTS_DIR / "confusion_matrix.png")
    plot_roc_curve(best_pipeline, X_test, y_test, best_name, config.REPORTS_DIR / "roc_curve.png")
    plot_pr_curve(best_pipeline, X_test, y_test, best_name, config.REPORTS_DIR / "pr_curve.png")
    plot_learning_curve(best_pipeline, X_train, y_train, config.REPORTS_DIR / "learning_curve.png", cv=config.CV_FOLDS)

    clf = best_pipeline.named_steps["model"]
    if hasattr(clf, "feature_importances_") and len(clf.feature_importances_) == len(feature_names):
        plot_feature_importance(feature_names, clf.feature_importances_, config.REPORTS_DIR / "feature_importance.png")

    comparison = []
    for name, m in all_metrics.items():
        comparison.append({
            "model": name,
            "accuracy": round(m["accuracy"], 4),
            "roc_auc": round(m["roc_auc"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "cv_auc_mean": round(m["cv_auc_mean"], 4),
            "cv_auc_std": round(m["cv_auc_std"], 4),
            "is_best": name == best_name,
        })
    comparison.sort(key=lambda x: x["roc_auc"], reverse=True)
    plot_model_comparison(comparison, config.REPORTS_DIR / "model_comparison.png")

    if _SHAP:
        _compute_shap(best_pipeline, X_train, X_test, best_name, feature_names, config.REPORTS_DIR, logger)

    joblib.dump(best_pipeline, config.MODEL_PATH)
    joblib.dump(feature_names, config.FEATURE_NAMES_PATH)

    n_bg = min(config.SHAP_BACKGROUND_SIZE, len(X_train))
    bg_sample = X_train.sample(n=n_bg, random_state=config.RANDOM_STATE)
    preprocessed_bg = best_pipeline.named_steps["preprocessor"].transform(bg_sample)
    joblib.dump(preprocessed_bg, config.SHAP_BACKGROUND_PATH)

    final_m = {k: v for k, v in all_metrics[best_name].items() if k != "confusion_matrix"}
    final_m["model_name"] = best_name
    with open(config.METRICS_PATH, "w") as f:
        json.dump(final_m, f, indent=2)

    with open(config.MODEL_COMPARISON_PATH, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Best Model   : {best_name}")
    logger.info(f"  Test AUC     : {all_metrics[best_name]['roc_auc']:.4f}")
    logger.info(f"  Test Accuracy: {all_metrics[best_name]['accuracy']:.4f}")
    logger.info(f"  Test F1      : {all_metrics[best_name]['f1']:.4f}")
    logger.info(f"  Model saved  : {config.MODEL_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
