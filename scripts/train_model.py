from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from scripts.database_connection import FEATURES_TABLE, get_engine, read_table



TARGET_COLUMN = "is_A_winner"
ID_COLUMN = "match_id"
LEAKAGE_KEYWORDS = ["crowns", "kingTowerHitPoints", "princessTowersHitPoints", "trophyChange"]
SEED = 42

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "clash_royale_winner_prediction"
REGISTERED_MODEL_NAME = "clash_royale_winner_predictor"
BEST_MODEL_PATH = Path("artifacts/best_model.joblib")
CONFUSION_MATRIX_PATH = Path("artifacts/confusion_matrix.png")

MODEL_CANDIDATES = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=SEED),
        "param_grid": {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]},
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=SEED),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_leaf": [1, 5],
        },
    },
    "xgboost": {
        "estimator": XGBClassifier(
            n_estimators=200, random_state=SEED, eval_metric="logloss"
        ),
        "param_grid": {
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "reg_alpha": [0, 0.1],
        },
    },
    "neural_network": {
        "estimator": MLPClassifier(max_iter=300, early_stopping=True, random_state=SEED),
        "param_grid": {
            "hidden_layer_sizes": [(32,), (64, 32)],
            "alpha": [0.0001, 0.001, 0.01],
        },
    }
}


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    leakage_cols = [c for c in df.columns if any(k in c for k in LEAKAGE_KEYWORDS)]
    if leakage_cols:
        df = df.drop(columns=leakage_cols)
    print(f"Dropped leakage columns ({len(leakage_cols)}): {leakage_cols}")
    return df


def load_dataset(table_name: str = FEATURES_TABLE) -> pd.DataFrame:
    engine = get_engine()
    return read_table(engine, table_name)


def split_dataset(df: pd.DataFrame, seed: int = SEED) -> dict:
    df = drop_leakage_columns(df)

    y = df[TARGET_COLUMN]
    match_id = df[ID_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])

    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X, y, match_id, test_size=0.3, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test, id_val, id_test = train_test_split(
        X_temp, y_temp, id_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )

    return {
        "train": (X_train, y_train, id_train),
        "val": (X_val, y_val, id_val),
        "test": (X_test, y_test, id_test),
    }


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def plot_confusion_matrix(y_true, y_pred, output_path: Path = CONFUSION_MATRIX_PATH) -> Path:
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def evaluate_on_test(best: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    model = best["model"]
    pred = model.predict(X_test)
    test_metrics = evaluate(model, X_test, y_test)

    cm_path = plot_confusion_matrix(y_test, pred)

    with mlflow.start_run(run_id=best["run_id"]):
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)
        mlflow.log_artifact(str(cm_path))

    print(f"Test metrics for best model '{best['name']}': {test_metrics}")
    print(classification_report(y_test, pred))
    print(f"Confusion matrix saved to {cm_path}")
    return test_metrics


def train_candidate(name: str, spec: dict, X_train, y_train, X_val, y_val) -> dict:
    search = GridSearchCV(
        spec["estimator"], spec["param_grid"], cv=5, scoring="roc_auc", n_jobs=1
    )

    with mlflow.start_run(run_name=name) as run:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        val_metrics = evaluate(best_model, X_val, y_val)

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("cv_best_roc_auc", search.best_score_)
        for metric_name, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", value)

        mlflow.sklearn.log_model(
            best_model, artifact_path="model", serialization_format="cloudpickle"
        )

        print(
            f"[{name}] best_params={search.best_params_} "
            f"cv_roc_auc={search.best_score_:.4f} val={val_metrics}"
        )

        return {
            "name": name,
            "model": best_model,
            "val_metrics": val_metrics,
            "run_id": run.info.run_id,
        }


def select_best(results: list) -> dict:
    return max(results, key=lambda r: r["val_metrics"]["roc_auc"])


def register_best_model(best: dict, feature_columns: list) -> None:
    model_uri = f"runs:/{best['run_id']}/model"
    mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best["model"],
            "feature_columns": feature_columns,
            "model_name": best["name"],
        },
        BEST_MODEL_PATH,
    )
    print(f"Registered best model '{best['name']}' and saved bundle to {BEST_MODEL_PATH}")


def run_training(df: pd.DataFrame, seed: int = SEED) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    splits = split_dataset(df, seed=seed)
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]

    results = [
        train_candidate(name, spec, X_train, y_train, X_val, y_val)
        for name, spec in MODEL_CANDIDATES.items()
    ]

    summary = pd.DataFrame(
        [{"model": r["name"], **r["val_metrics"]} for r in results]
    ).sort_values("roc_auc", ascending=False)
    print("Validation comparison:")
    print(summary.to_string(index=False))

    best = select_best(results)
    print(f"Best model: {best['name']} (val_roc_auc={best['val_metrics']['roc_auc']:.4f})")

    X_test, y_test, _ = splits["test"]
    test_metrics = evaluate_on_test(best, X_test, y_test)

    register_best_model(best, list(X_train.columns))

    return {"results": results, "best": best, "test_metrics": test_metrics, "splits": splits}


def main():
    parser = argparse.ArgumentParser(description="Train Clash Royale winner prediction model.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    df = load_dataset()
    run_training(df, seed=args.seed)


if __name__ == "__main__":
    main()
