"""Build the 1-shot training set by repeating one DAPO problem.

    python data/prep/prepare_dapo_oneshot.py
    python data/prep/prepare_dapo_oneshot.py --target-row 59 --repeat 64

Writes math/DAPO-oneshot/t22.parquet (64 rows). Requires
math/DAPO/train.parquet, whose row order determines which problem is selected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _paths import DATA_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DATA_ROOT / "math" / "DAPO" / "train.parquet",
        help="Input parquet path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_ROOT / "math" / "DAPO-oneshot" / "t22.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--target-row",
        type=int,
        default=22,
        help="Zero-based row position to repeat. Negative values follow pandas iloc.",
    )
    parser.add_argument(
        "--repeat", type=int, default=64, help="Number of output rows (= train_batch_size)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat <= 0:
        raise ValueError(f"--repeat must be positive, got {args.repeat}")

    df = pd.read_parquet(args.source)
    if df.empty:
        raise ValueError(f"{args.source} has no rows")
    if not -len(df) <= args.target_row < len(df):
        raise IndexError(f"--target-row {args.target_row} out of range for {len(df)} rows")

    source_df = df.iloc[[args.target_row]]
    out_df = pd.concat([source_df] * args.repeat, ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)

    print(f"Source: {args.source}")
    print(f"Selected row: iloc[{args.target_row}]")
    print(f"Output: {args.out}")
    print(f"Rows: {len(out_df)}")
    print(f"Columns: {list(out_df.columns)}")


if __name__ == "__main__":
    main()
