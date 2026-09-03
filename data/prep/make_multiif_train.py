"""Build the IF training set from the Multi-IF source CSV.

    python data/prep/make_multiif_train.py --src multiIF_20241018.csv

Writes if/MultiIF/train.parquet (4501 rows, first turn only).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from align_mopd_schema import REF_SCHEMA, write_aligned
from _paths import DATA_ROOT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="multiIF_20241018.csv")
    p.add_argument("--out", type=Path, default=DATA_ROOT / "if" / "MultiIF" / "train.parquet")
    return p.parse_args()


def build(src: Path) -> pd.DataFrame:
    rows = []
    for _, r in pd.read_csv(src).iterrows():
        turn = json.loads(r["turn_1_prompt"])
        instruction_ids = json.loads(r["turn_1_instruction_id_list"])
        # kwargs arrives as a list of JSON *strings*, one per instruction id.
        kwargs = [json.loads(k) for k in json.loads(r["turn_1_kwargs"])]

        ground_truth = {
            # MultiIF keys look like "1019:16:en", not IFEval's bare integers.
            # The checkers only carry the key through InputExample as a label,
            # so the string goes in unchanged.
            "key": str(r["key"]),
            "prompt": turn["content"],
            "instruction_id_list": instruction_ids,
            "kwargs": kwargs,
        }
        rows.append(
            {
                # Field order matters: the shared schema is (content, role).
                "prompt": [{"content": turn["content"], "role": "user"}],
                "data_source": "MultiIF",
                "ability": "instruction_following",
                "reward_model": {"ground_truth": json.dumps(ground_truth), "style": "rule"},
                # The language survives in `source` rather than in a key of its
                # own -- the shared struct has no room for one, and no scorer
                # reads extra_info.
                "extra_info": {
                    "difficulty": "",
                    "index": str(r["key"]),
                    "n_tests": 0,
                    "source": f"MultiIF-{r['language']}",
                    "split": "train",
                },
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = build(args.src)
    write_aligned(df, args.out, REF_SCHEMA())
    print(f"{len(df)} rows -> {args.out}")
    print(df["extra_info"].map(lambda e: e["source"]).value_counts().to_dict())


if __name__ == "__main__":
    main()
