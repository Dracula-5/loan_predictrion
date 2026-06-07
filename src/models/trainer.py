from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)

try:
    from xgboost import XGBClassifier
    _XGBOOST = True
except ImportError:
    _XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    _LIGHTGBM = True
except ImportError:
    _LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    _CATBOOST = True
except ImportError:
    _CATBOOST = False


def build_preprocessor(numerical_features: list, categorical_features: list) -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return ColumnTransformer([
        ("num", num_pipeline, numerical_features),
        ("cat", cat_pipeline, categorical_features),
    ])


def _model_configs():
    configs = {
        "LogisticRegression": {
            "model": LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"),
            "params": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__solver": ["lbfgs", "liblinear"],
            },
            "search": "grid",
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            "params": {
                "model__max_depth": [3, 5, 7, None],
                "model__min_samples_split": [2, 5, 10],
                "model__criterion": ["gini", "entropy"],
            },
            "search": "grid",
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [5, 10, 15, None],
                "model__min_samples_split": [2, 5],
                "model__max_features": ["sqrt", "log2"],
            },
            "search": "random",
            "n_iter": 15,
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "model__n_neighbors": [3, 5, 7, 11, 15],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan"],
            },
            "search": "grid",
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42, class_weight="balanced"),
            "params": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale", "auto"],
            },
            "search": "grid",
        },
    }

    if _XGBOOST:
        configs["XGBoost"] = {
            "model": XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
                "model__subsample": [0.7, 0.8, 1.0],
                "model__colsample_bytree": [0.7, 0.8, 1.0],
            },
            "search": "random",
            "n_iter": 15,
        }

    if _LIGHTGBM:
        configs["LightGBM"] = {
            "model": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 5, 7, -1],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__num_leaves": [15, 31, 50],
                "model__subsample": [0.7, 0.8, 1.0],
            },
            "search": "random",
            "n_iter": 15,
        }

    if _CATBOOST:
        configs["CatBoost"] = {
            "model": CatBoostClassifier(random_state=42, verbose=0),
            "params": {
                "model__iterations": [100, 200, 300],
                "model__depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__l2_leaf_reg": [1, 3, 5],
            },
            "search": "random",
            "n_iter": 12,
        }

    return configs


def train_all_models(
    X_train,
    y_train,
    preprocessor: ColumnTransformer,
    cv_folds: int = 5,
    logger=None,
) -> dict:
    configs = _model_configs()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    for name, cfg in configs.items():
        if logger:
            logger.info(f"Training {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", cfg["model"]),
        ])

        if cfg["search"] == "grid":
            searcher = GridSearchCV(
                pipeline, cfg["params"],
                cv=cv, scoring="roc_auc",
                n_jobs=-1, refit=True,
            )
        else:
            searcher = RandomizedSearchCV(
                pipeline, cfg["params"],
                n_iter=cfg.get("n_iter", 10),
                cv=cv, scoring="roc_auc",
                n_jobs=-1, random_state=42, refit=True,
            )

        searcher.fit(X_train, y_train)

        cv_scores = cross_val_score(
            searcher.best_estimator_,
            X_train, y_train,
            cv=cv, scoring="roc_auc",
        )

        results[name] = {
            "pipeline": searcher.best_estimator_,
            "best_params": {
                k: (float(v) if hasattr(v, "item") else v)
                for k, v in searcher.best_params_.items()
            },
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
        }

        if logger:
            logger.info(
                f"  {name} | CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} "
                f"| Best params: {searcher.best_params_}"
            )

    return results
