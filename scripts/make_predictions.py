from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from scripts.database_connection import PREPROCESSED_TABLE, get_engine, read_table, write_table
from scripts.feature_engineering import SEED, add_features


BEST_MODEL_PATH = Path("artifacts/best_model.joblib")
SCALER_PATH = Path("artifacts/scaler.pkl")
PREDICTIONS_TABLE = "match_predictions"


def load_model_bundle(model_path: Path = BEST_MODEL_PATH) -> dict:
    return joblib.load(model_path)


def load_scaler_bundle(scaler_path: Path = SCALER_PATH) -> dict:
    return joblib.load(scaler_path)


def apply_saved_scaler(df: pd.DataFrame, scaler_bundle: dict) -> pd.DataFrame:
    df = df.copy()
    scaler = scaler_bundle["scaler"]
    cols = [c for c in scaler_bundle["columns"] if c in df.columns]
    df[cols] = scaler.transform(df[cols].astype(float))
    return df


def predict(df: pd.DataFrame, model_bundle: dict, scaler_bundle: dict, assume_preprocessed: bool = True) -> pd.DataFrame:
    featured = add_features(df, assume_preprocessed=assume_preprocessed, seed=SEED)

    match_id = featured["match_id"]
    actual = featured["is_A_winner"] if "is_A_winner" in featured.columns else None

    scaled = apply_saved_scaler(featured, scaler_bundle)

    feature_columns = model_bundle["feature_columns"]
    X = scaled.reindex(columns=feature_columns, fill_value=0)

    model = model_bundle["model"]
    predicted_label = model.predict(X)
    predicted_probability = model.predict_proba(X)[:, 1]

    result = pd.DataFrame(
        {
            "match_id": match_id.to_numpy(),
            "predicted_label": predicted_label,
            "predicted_probability": predicted_probability,
        }
    )
    if actual is not None:
        result["actual_is_A_winner"] = actual.to_numpy()

    return result


def run_predictions(
    input_path: Path | None = None,
    table_name: str = PREPROCESSED_TABLE,
    predictions_table: str = PREDICTIONS_TABLE,
    model_path: Path = BEST_MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
) -> pd.DataFrame:
    engine = get_engine()

    if input_path is None:
        df = read_table(engine, table_name)
        assume_preprocessed = True
    else:
        df = pd.read_csv(input_path)
        assume_preprocessed = False

    model_bundle = load_model_bundle(model_path)
    scaler_bundle = load_scaler_bundle(scaler_path)

    predictions = predict(df, model_bundle, scaler_bundle, assume_preprocessed=assume_preprocessed)

    write_table(engine, predictions, predictions_table, if_exists="replace")
    print(f"Wrote {len(predictions)} predictions to table: {predictions_table}")
    print(predictions.head())

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate winner predictions for Clash Royale matches.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional CSV in the preprocessed_match shape. If omitted, reads from PostgreSQL.",
    )
    parser.add_argument("--model-path", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--scaler-path", type=Path, default=SCALER_PATH)
    args = parser.parse_args()

    run_predictions(
        input_path=args.input_path,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
    )


if __name__ == "__main__":
    main()
