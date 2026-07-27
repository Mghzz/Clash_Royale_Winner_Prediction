from __future__ import annotations

import argparse
from pathlib import Path

from scripts.feature_engineering import run_feature_engineering
from scripts.load_data import load_csv_to_postgres
from scripts.make_predictions import run_predictions
from scripts.preprocess import run_preprocess
from scripts.train_model import SEED, load_dataset, run_training


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: load -> preprocess -> feature engineering -> train -> predict."
    )
    parser.add_argument("--csv-path", type=Path, required=True, help="Input CSV path.")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for loading.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Load only the first N rows.")
    parser.add_argument("--no-reset", action="store_true", help="Do not truncate tables before load.")
    parser.add_argument("--preprocessed-path", type=Path, default=Path("artifacts/preprocessed_data.csv"))
    parser.add_argument("--featured-path", type=Path, default=Path("artifacts/model_ready_data.csv"))
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for the train/val/test split and A/B sides.")
    args = parser.parse_args()

    load_csv_to_postgres(
        csv_path=args.csv_path,
        chunk_size=args.chunk_size,
        reset=not args.no_reset,
        limit_rows=args.limit_rows,
    )

    run_preprocess(output_path=args.preprocessed_path)

    run_feature_engineering(output_path=args.featured_path, seed=args.seed)

    df = load_dataset()
    run_training(df, seed=args.seed)

    run_predictions()


if __name__ == "__main__":
    main()
