"""Align parquet schemas so MOPD can concatenate math, code and IF in one run.

    python data/prep/align_mopd_schema.py                    # verify all six files
    python data/prep/align_mopd_schema.py --src in.parquet --dst out.parquet

Run without arguments to verify. `datasets` requires field-by-field identical
schemas when concatenating; the target schema is defined in this file.
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from _paths import DATA_ROOT

#: The five files MOPD concatenates, TACO first -- it defines the schema.
MOPD_FILES = [
    "code/TACO/train.parquet",
    "math/DAPO/train.parquet",
    "if/MultiIF/train.parquet",
    "math/MATH-500/test.parquet",
    "code/LCBv6/test.parquet",
]


#: TACO's schema written out. See the module docstring for why it is a literal
#: rather than whichever file happens to be on disk.
SCHEMA = pa.schema(
    [
        pa.field("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
        pa.field("data_source", pa.large_string()),
        pa.field("ability", pa.large_string()),
        pa.field("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
        pa.field(
            "extra_info",
            pa.struct(
                [
                    ("difficulty", pa.string()),
                    ("index", pa.string()),
                    ("n_tests", pa.int64()),
                    ("source", pa.string()),
                    ("split", pa.string()),
                ]
            ),
        ),
    ]
)


def REF_SCHEMA() -> pa.Schema:
    """The schema every MOPD file is written onto."""
    return SCHEMA


def _canon(schema: pa.Schema) -> pa.Schema:
    """Sort struct children by name, so field order stops being a difference.

    Recurses through list<struct<...>> too -- `prompt` is one, and it is the
    field the two orders actually show up in.
    """

    def canon_type(t: pa.DataType) -> pa.DataType:
        if pa.types.is_struct(t):
            return pa.struct(sorted(((f.name, canon_type(f.type)) for f in t), key=lambda kv: kv[0]))
        if pa.types.is_list(t) or pa.types.is_large_list(t):
            return pa.list_(canon_type(t.value_type))
        return t

    return pa.schema([pa.field(f.name, canon_type(f.type)) for f in schema])


def write_aligned(df: pd.DataFrame, out: Path, schema: pa.Schema) -> None:
    """Write `df` with exactly `schema`'s columns, order and types."""
    columns = [f.name for f in schema if f.name in df.columns]
    missing = [f.name for f in schema if f.name not in df.columns]
    if missing:
        raise ValueError(f"{out}: missing columns {missing}")
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df[columns], schema=schema, preserve_index=False)
    pq.write_table(table, out)


def normalize(src: Path, dst: Path, source: str = "", split: str = "") -> None:
    """Rewrite one parquet onto the reference schema.

    extra_info is rebuilt rather than carried over: the inputs shared only
    `index`, so there is nothing else that could survive generically.
    """
    schema = REF_SCHEMA()
    df = pd.read_parquet(src)
    df["extra_info"] = [
        {
            "difficulty": str(r["extra_info"].get("difficulty", "")),
            "index": str(r["extra_info"]["index"]),
            "n_tests": int(r["extra_info"].get("n_tests", 0)),
            "source": source or str(r["extra_info"].get("source", "")),
            "split": split or str(r["extra_info"].get("split", "")),
        }
        for _, r in df.iterrows()
    ]
    # Rebuild prompt so the struct's fields land in the schema's order.
    df["prompt"] = [[{"content": m["content"], "role": m["role"]} for m in p] for p in df["prompt"]]
    write_aligned(df, dst, schema)
    print(f"{len(df)} rows -> {dst}")


def check() -> int:
    """Compare every prepared MOPD file against SCHEMA. 0 if they all concat."""
    ref = _canon(SCHEMA)
    bad = 0
    for rel in MOPD_FILES:
        p = DATA_ROOT / rel
        if not p.exists():
            print(f"  --   {rel} absent, not prepared yet")
            continue
        s = pq.read_schema(p).remove_metadata()
        if s.equals(SCHEMA):
            print(f"  OK   {rel}")
        elif _canon(s).equals(ref):
            # Same fields, same types, different child order. Concatenates fine.
            print(f"  OK   {rel} (struct field order differs, which datasets normalises)")
        else:
            bad = 1
            print(f"  DIFF {rel}\n{s}")
    return bad


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, help="parquet to rewrite onto the schema")
    p.add_argument("--dst", type=Path, help="where to write it")
    p.add_argument("--source", default="", help="extra_info.source to stamp")
    p.add_argument("--split", default="", help="extra_info.split to stamp")
    a = p.parse_args()
    if a.src or a.dst:
        if not (a.src and a.dst):
            raise SystemExit("--src and --dst go together")
        normalize(a.src, a.dst, a.source, a.split)
        return
    raise SystemExit(check())


if __name__ == "__main__":
    main()
