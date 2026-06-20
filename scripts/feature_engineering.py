from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from scripts.database_connection import (
    FEATURES_TABLE,
    PER_SIDE_CARD_AGG,
    PER_SIDE_NUMERIC,
    PREPROCESSED_TABLE,
    get_engine,
    read_table,
    write_table,
)
from scripts.preprocess import preprocess_dataframe


TARGET_COLUMN = "is_A_winner"
SEED = 42

AB_FEATURES = list(PER_SIDE_NUMERIC) + list(PER_SIDE_CARD_AGG) + ["clan_badge_id", "clan_tag_freq"]
COMPARE_FEATURES = list(PER_SIDE_NUMERIC) + list(PER_SIDE_CARD_AGG)
RATIO_FEATURES = ["startingTrophies", "elixir_average", "avg_card_level", "totalcard_level"]

SCALER_EXCLUDE = {
    TARGET_COLUMN,
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
}


def assign_ab_sides(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed)
    swap = rng.random(len(df)) < 0.5

    for c in AB_FEATURES:
        w_col, l_col = f"w_{c}", f"l_{c}"
        if w_col in df.columns and l_col in df.columns:
            w_vals = df[w_col].to_numpy()
            l_vals = df[l_col].to_numpy()
            df[f"A_{c}"] = np.where(swap, l_vals, w_vals)
            df[f"B_{c}"] = np.where(swap, w_vals, l_vals)

    df[TARGET_COLUMN] = np.where(swap, 0, 1).astype("int64")
    return df


def add_comparison_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in COMPARE_FEATURES:
        a_col, b_col = f"A_{c}", f"B_{c}"
        if a_col in df.columns and b_col in df.columns:
            df[f"diff_{c}"] = df[a_col] - df[b_col]

    for c in RATIO_FEATURES:
        a_col, b_col = f"A_{c}", f"B_{c}"
        if a_col in df.columns and b_col in df.columns:
            df[f"ratio_{c}"] = df[a_col] / (df[b_col] + 1e-6)

    for side in ("A", "B"):
        density_parts = [f"{side}_{c}" for c in ("troop_count", "structure_count", "spell_count")]
        if all(col in df.columns for col in density_parts):
            df[f"{side}_deck_density"] = sum(df[col] for col in density_parts)
        power_parts = [f"{side}_{c}" for c in ("avg_card_level", "num_cards", "high_level_cards")]
        if all(col in df.columns for col in power_parts):
            df[f"{side}_card_power_index"] = (
                df[f"{side}_avg_card_level"] * df[f"{side}_num_cards"] + df[f"{side}_high_level_cards"]
            )

    if "A_deck_density" in df.columns and "B_deck_density" in df.columns:
        df["diff_deck_density"] = df["A_deck_density"] - df["B_deck_density"]
    if "A_card_power_index" in df.columns and "B_card_power_index" in df.columns:
        df["diff_card_power_index"] = df["A_card_power_index"] - df["B_card_power_index"]

    return df


def add_features(df: pd.DataFrame, assume_preprocessed: bool = True, seed: int = SEED) -> pd.DataFrame:
    df = df.copy()

    if not assume_preprocessed:
        df = preprocess_dataframe(df)

    df = assign_ab_sides(df, seed=seed)
    df = add_comparison_features(df)

    df["arena_id"] = pd.to_numeric(df["arena_id"], errors="coerce").fillna(-1).astype("int64")
    df["gameMode_id"] = pd.to_numeric(df["gameMode_id"], errors="coerce").fillna(-1).astype("int64")

    df["hour_sin"] = np.sin(2 * np.pi * df["battle_hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["battle_hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["battle_dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["battle_dayofweek"] / 7.0)

    drop_cols = [c for c in df.columns if c.startswith("w_") or c.startswith("l_")]
    drop_cols += [c for c in ["battleTime"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def drop_redundant_features(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    corr_threshold: float = 0.95,
) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target]

    near_constant = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    remaining = [c for c in feature_cols if c not in near_constant]

    high_corr: list[str] = []
    if len(remaining) > 1:
        corr = df[remaining].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        high_corr = [c for c in upper.columns if (upper[c] > corr_threshold).any()]

    to_drop = sorted(set(near_constant) | set(high_corr))
    if to_drop:
        df = df.drop(columns=to_drop)
    print(
        f"[feature-selection] dropped {len(to_drop)} redundant column(s); "
        f"near-constant={near_constant}; "
        f"correlated(>|{corr_threshold}|)={high_corr}"
    )
    return df


def standardize_features(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    scaler_path: Path | None = Path("artifacts/scaler.pkl"),
) -> tuple[pd.DataFrame, StandardScaler | None]:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in numeric_cols if c not in SCALER_EXCLUDE]
    if not cols:
        print("[scaling] no numeric columns to standardize; skipping.")
        return df, None

    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols].astype(float))

    if scaler_path is not None:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": scaler, "columns": cols}, scaler_path)
        print(f"[scaling] standardized {len(cols)} column(s); scaler saved to {scaler_path}")
    else:
        print(f"[scaling] standardized {len(cols)} column(s).")
    return df, scaler


def run_feature_engineering(
    output_path: Path | None = Path("artifacts/model_ready_data.csv"),
    input_path: Path | None = None,
    source_table: str = PREPROCESSED_TABLE,
    table_name: str = FEATURES_TABLE,
    corr_threshold: float = 0.95,
    scaler_path: Path | None = Path("artifacts/scaler.pkl"),
    seed: int = SEED,
) -> pd.DataFrame:
    engine = get_engine()

    if input_path is None:
        df = read_table(engine, source_table)
        featured = add_features(df, assume_preprocessed=True, seed=seed)
    else:
        df = pd.read_csv(input_path)
        featured = add_features(df, assume_preprocessed=False, seed=seed)

    featured = drop_redundant_features(featured, corr_threshold=corr_threshold)
    featured, _ = standardize_features(featured, scaler_path=scaler_path)

    write_table(engine, featured, table_name)
    print(f"Feature-engineered dataset written to DB table: {table_name}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        featured.to_csv(output_path, index=False)
        print(f"Feature-engineered dataset saved to: {output_path}")

    print(f"Shape: {featured.shape}")
    return featured


def main():
    parser = argparse.ArgumentParser(description="Feature engineering for Clash Royale match-level dataset.")
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/model_ready_data.csv"))
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional CSV path. If omitted, data is loaded directly from PostgreSQL.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.95,
        help="Absolute correlation above which one of a feature pair is dropped.",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=Path("artifacts/scaler.pkl"),
        help="Where to persist the fitted StandardScaler for reuse in modeling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed for the random A/B side assignment.",
    )
    args = parser.parse_args()

    run_feature_engineering(
        output_path=args.output_path,
        input_path=args.input_path,
        corr_threshold=args.corr_threshold,
        scaler_path=args.scaler_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
