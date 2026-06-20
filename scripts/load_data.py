from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from scripts.database_connection import ensure_schema, get_engine, reset_tables


PRINCESS_HP_COLUMNS = [
    "winner.princessTowersHitPoints",
    "loser.princessTowersHitPoints",
]


def parse_princess_hp(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    if isinstance(value, (int, float, np.generic)):
        return float(value)

    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return 0.0
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, (list, tuple)):
        nums = [float(x) for x in parsed if x is not None]
        return float(sum(nums))
    return float(parsed)


NUMERIC_COLUMNS = [
    "arena.id",
    "gameMode.id",
    "average.startingTrophies",
    "winner.startingTrophies",
    "winner.trophyChange",
    "winner.crowns",
    "winner.kingTowerHitPoints",
    "winner.princessTowersHitPoints",
    "winner.clan.badgeId",
    "loser.startingTrophies",
    "loser.trophyChange",
    "loser.crowns",
    "loser.kingTowerHitPoints",
    "loser.princessTowersHitPoints",
    "loser.clan.badgeId",
] + [f"{side}.card{i}.id" for side in ("winner", "loser") for i in range(1, 9)] + [
    f"{side}.card{i}.level" for side in ("winner", "loser") for i in range(1, 9)
]


def clean_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.generic,)):
        return v.item()
    return v


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def batch_iterable(values: Sequence, batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def insert_dataframe(engine, table_name: str, df: pd.DataFrame, batch_size: int = 5000) -> None:
    if df.empty:
        return

    columns = list(df.columns)
    sql = f'INSERT INTO {quote_ident(table_name)} ({", ".join(quote_ident(c) for c in columns)}) VALUES %s'
    records = [tuple(clean_value(v) for v in row) for row in df.itertuples(index=False, name=None)]

    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            for batch in batch_iterable(records, batch_size):
                execute_values(cur, sql, batch, page_size=len(batch))
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()


def prepare_chunk(chunk: pd.DataFrame, match_start_id: int, player_start_id: int):
    chunk = chunk.copy()

    chunk["battleTime"] = pd.to_datetime(chunk["battleTime"], errors="coerce", utc=True).dt.tz_localize(None)
    for col in NUMERIC_COLUMNS:
        if col in chunk.columns and col not in PRINCESS_HP_COLUMNS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    for col in PRINCESS_HP_COLUMNS:
        if col in chunk.columns:
            chunk[col] = chunk[col].map(parse_princess_hp)

    match_rows = []
    player_rows = []
    player_card_rows = []
    card_rows = []
    clan_rows = []

    match_id = match_start_id
    player_id = player_start_id

    for _, row in chunk.iterrows():
        match_rows.append(
            {
                "id": match_id,
                "battleTime": row.get("battleTime"),
                "arena_id": clean_value(row.get("arena.id")),
                "gameMode_id": clean_value(row.get("gameMode.id")),
                "tournamentTag": clean_value(row.get("tournamentTag")),
                "average_startingTrophies": clean_value(row.get("average.startingTrophies")),
            }
        )

        for side in ("winner", "loser"):
            clan_tag = clean_value(row.get(f"{side}.clan.tag"))
            clan_badge = clean_value(row.get(f"{side}.clan.badgeId"))
            if clan_tag is not None:
                clan_rows.append({"tag": clan_tag, "badgeId": clan_badge})

            player_rows.append(
                {
                    "id": player_id,
                    "match_id": match_id,
                    "player_tag": clean_value(row.get(f"{side}.tag")),
                    "clan_tag": clan_tag,
                    "is_winner": side == "winner",
                    "startingTrophies": clean_value(row.get(f"{side}.startingTrophies")),
                    "elixir_average": clean_value(row.get(f"{side}.elixir.average")),
                    "totalcard_level": clean_value(row.get(f"{side}.totalcard.level")),
                    "troop_count": clean_value(row.get(f"{side}.troop.count")),
                    "structure_count": clean_value(row.get(f"{side}.structure.count")),
                    "spell_count": clean_value(row.get(f"{side}.spell.count")),
                    "common_count": clean_value(row.get(f"{side}.common.count")),
                    "rare_count": clean_value(row.get(f"{side}.rare.count")),
                    "epic_count": clean_value(row.get(f"{side}.epic.count")),
                    "legendary_count": clean_value(row.get(f"{side}.legendary.count")),
                    "trophyChange": clean_value(row.get(f"{side}.trophyChange")),
                    "crowns": clean_value(row.get(f"{side}.crowns")),
                    "kingTowerHitPoints": clean_value(row.get(f"{side}.kingTowerHitPoints")),
                    "princessTowersHitPoints": clean_value(row.get(f"{side}.princessTowersHitPoints")),
                }
            )

            for i in range(1, 9):
                card_id = clean_value(row.get(f"{side}.card{i}.id"))
                card_level = clean_value(row.get(f"{side}.card{i}.level"))
                if card_id is not None:
                    card_rows.append({"id": int(card_id)})
                    player_card_rows.append(
                        {
                            "match_player_id": player_id,
                            "card_id": int(card_id),
                            "card_level": clean_value(card_level),
                        }
                    )

            player_id += 1

        match_id += 1

    match_df = pd.DataFrame(match_rows)
    player_df = pd.DataFrame(player_rows)
    player_card_df = pd.DataFrame(player_card_rows)
    card_df = pd.DataFrame(card_rows).drop_duplicates(subset=["id"], keep="first") if card_rows else pd.DataFrame(columns=["id"])
    clan_df = pd.DataFrame(clan_rows).drop_duplicates(subset=["tag"], keep="first") if clan_rows else pd.DataFrame(columns=["tag", "badgeId"])

    return match_df, player_df, player_card_df, card_df, clan_df, match_id, player_id


def load_csv_to_postgres(
    csv_path: Path,
    chunk_size: int = 5000,
    reset: bool = True,
    limit_rows: Optional[int] = None,
) -> None:
    engine = get_engine()
    ensure_schema(engine)
    if reset:
        reset_tables(engine)

    seen_cards = set()
    seen_clans = set()
    match_id = 1
    player_id = 1
    total_rows = 0

    reader = pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        low_memory=False,
        nrows=limit_rows,
    )

    for chunk_idx, chunk in enumerate(reader, start=1):
        if chunk.empty:
            continue

        match_df, player_df, player_card_df, card_df, clan_df, match_id, player_id = prepare_chunk(
            chunk, match_id, player_id
        )

        if not clan_df.empty:
            clan_df = clan_df[clan_df["tag"].notna()]
            clan_df = clan_df[~clan_df["tag"].isin(seen_clans)]
            if not clan_df.empty:
                seen_clans.update(clan_df["tag"].tolist())

        if not card_df.empty:
            card_df = card_df[~card_df["id"].isin(seen_cards)]
            if not card_df.empty:
                seen_cards.update(card_df["id"].tolist())

        match_df = match_df.dropna(subset=["id"])
        player_df = player_df.dropna(subset=["id", "match_id"])
        player_card_df = player_card_df.dropna(subset=["match_player_id", "card_id"])
        card_df = card_df.dropna(subset=["id"])
        clan_df = clan_df.dropna(subset=["tag"])

        print(
            f"[chunk {chunk_idx}] matches={len(match_df)} players={len(player_df)} "
            f"player_cards={len(player_card_df)} cards={len(card_df)} clans={len(clan_df)}"
        )

        insert_dataframe(engine, "Match", match_df, batch_size=5000)
        insert_dataframe(engine, "Clan", clan_df, batch_size=5000)
        insert_dataframe(engine, "Card", card_df, batch_size=5000)
        insert_dataframe(engine, "MatchPlayer", player_df, batch_size=5000)
        insert_dataframe(engine, "MatchPlayerCard", player_card_df, batch_size=10000)

        total_rows += len(chunk)
        print(f"Processed rows: {total_rows}")

    print("Load finished successfully.")


def main():
    parser = argparse.ArgumentParser(description="Load Clash Royale CSV into PostgreSQL.")
    parser.add_argument("--csv-path", required=True, type=Path, help="Path to the source CSV.")
    parser.add_argument("--chunk-size", type=int, default=5000, help="CSV chunk size.")
    parser.add_argument("--no-reset", action="store_true", help="Append without truncating tables.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Load only the first N rows.")
    args = parser.parse_args()

    load_csv_to_postgres(
        csv_path=args.csv_path,
        chunk_size=args.chunk_size,
        reset=not args.no_reset,
        limit_rows=args.limit_rows,
    )


if __name__ == "__main__":
    main()
