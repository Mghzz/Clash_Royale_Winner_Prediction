from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def export_sample_csv(input_path: Path, output_path: Path, rows: int = 100_000) -> Path:
    df = pd.read_csv(input_path, nrows=rows, low_memory=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample CSV saved to: {output_path}")
    print(f"Rows written: {len(df)}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export a GitHub-safe sample CSV.")
    parser.add_argument("--input-path", type=Path, required=True, help="Original large CSV.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/sample/BattlesStaging_sample.csv"),
        help="Where to save the sampled CSV.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100_000,
        help="Number of top rows to export.",
    )
    args = parser.parse_args()

    export_sample_csv(args.input_path, args.output_path, rows=args.rows)


if __name__ == "__main__":
    main()
