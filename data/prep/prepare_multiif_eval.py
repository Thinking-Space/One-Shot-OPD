"""Build the Multi-IF evaluation set from the source CSV.

    python data/prep/prepare_multiif_eval.py --src multiIF_20241018.csv

Writes if/MultiIF/eval_3turn.parquet (4445 rows, 3 turns per row). This file is
read by eval/run_if_eval.py, not by verl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _paths import DATA_ROOT


#: Where `ms_fetch.sh datasets facebook/Multi-IF <dir>` drops the CSV.
DEFAULT_SRC = DATA_ROOT / ".cache" / "multiif" / "multiIF_20241018.csv"

TURNS = (1, 2, 3)


def _parse_kwargs(raw: str) -> list[dict]:
    """Multi-IF's kwargs column is double-encoded: a JSON list of JSON strings.

    ``'["{}", "{\\"end_phrase\\": \\"...\\"}"]'`` -> ``[{}, {"end_phrase": ...}]``.
    Rows written by a newer dump have the inner objects already decoded, so
    accept both rather than assuming.
    """
    return [json.loads(k) if isinstance(k, str) else k for k in json.loads(raw)]


def build(src: Path) -> pd.DataFrame:
    df = pd.read_csv(src, keep_default_na=False)
    rows = []
    for _, r in df.iterrows():
        if not r["turn_3_prompt"]:
            continue
        row = {"key": str(r["key"]), "language": str(r["language"])}
        for t in TURNS:
            row[f"turn_{t}_prompt"] = json.loads(r[f"turn_{t}_prompt"])["content"]
            row[f"turn_{t}_instruction_id_list"] = json.loads(r[f"turn_{t}_instruction_id_list"])
            row[f"turn_{t}_kwargs"] = json.dumps(_parse_kwargs(r[f"turn_{t}_kwargs"]))
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="multiIF_20241018.csv")
    p.add_argument("--out", type=Path, default=DATA_ROOT / "if" / "MultiIF" / "eval_3turn.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise SystemExit(
            f"{args.src} not found. Fetch it first:\n"
            f"  bash data/prep/ms_fetch.sh datasets facebook/Multi-IF {args.src.parent}\n"
            "(hf-mirror.com resets the connection as of 2026-08-29; ModelScope carries "
            "the same file.)"
        )
    df = build(args.src)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"{len(df)} rows -> {args.out}")
    print(df["language"].value_counts().to_dict())


if __name__ == "__main__":
    main()
