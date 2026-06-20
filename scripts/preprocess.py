from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.database_connection import (
    MATCH_NUMERIC,
    PER_SIDE_CARD_AGG,
    PER_SIDE_NUMERIC,
    PREPROCESSED_TABLE,
    build_match_base_query,
    get_engine,
    write_table,
)


SIDES = ("w", "l")

NUMERIC_COLUMNS = (
    list(MATCH_NUMERIC)
    + [f"{s}_{c}" for s in SIDES for c in PER_SIDE_NUMERIC]
    + [f"{s}_{c}" for s in SIDES for c in PER_SIDE_CARD_AGG]
    + [f"{s}_clan_badge_id" for s in SIDES]
)

CATEGORICAL_COLUMNS = [f"{s}_clan_tag" for s in SIDES]

HP_COLUMNS = [
    f"{s}_{c}" for s in SIDES for c in ("kingTowerHitPoints", "princessTowersHitPoints")
]

DROP_ID_COLUMNS = ["match_id"]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        print(f"Dropped fully-empty columns: {empty_cols}")

    df["battleTime"] = pd.to_datetime(df["battleTime"], errors="coerce", utc=True).dt.tz_convert(None)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            median = df[col].median()
            if pd.isna(median):
                median = 0
            df[col] = df[col].fillna(median)

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("unknown").str.strip()
            df.loc[df[col] == "", col] = "unknown"

    for col in HP_COLUMNS:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    df["battle_hour"] = df["battleTime"].dt.hour.fillna(-1).astype("int64")
    df["battle_dayofweek"] = df["battleTime"].dt.dayofweek.fillna(-1).astype("int64")
    df["battle_month"] = df["battleTime"].dt.month.fillna(-1).astype("int64")
    df["is_weekend"] = df["battle_dayofweek"].isin([5, 6]).astype("int64")

    for col in DROP_ID_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True, dropna=False)
            df[f"{col}_freq"] = df[col].map(freq).fillna(0.0).astype(float)

    df = df.drop_duplicates()

    return df


def run_preprocess(
    output_path: Path | None = Path("artifacts/preprocessed_data.csv"),
    input_path: Path | None = None,
    table_name: str = PREPROCESSED_TABLE,
) -> pd.DataFrame:
    engine = get_engine()

    if input_path is None:
        query = build_match_base_query()
        df = pd.read_sql_query(query, engine)
    else:
        df = pd.read_csv(input_path)

    cleaned = preprocess_dataframe(df)

    write_table(engine, cleaned, table_name)
    print(f"Preprocessed dataset written to DB table: {table_name}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(output_path, index=False)
        print(f"Preprocessed dataset saved to: {output_path}")

    print(f"Shape: {cleaned.shape}")
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Preprocess Clash Royale match-level dataset.")
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/preprocessed_data.csv"))
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional CSV path. If omitted, data is loaded directly from PostgreSQL.",
    )
    args = parser.parse_args()

    run_preprocess(output_path=args.output_path, input_path=args.input_path)


if __name__ == "__main__":
    main()
