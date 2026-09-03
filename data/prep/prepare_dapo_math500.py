"""Build the math training and evaluation sets.

    python data/prep/prepare_dapo_math500.py --set all
    python data/prep/prepare_dapo_math500.py --set math500
    python data/prep/prepare_dapo_math500.py --set dapo --verify existing.parquet

Writes math/DAPO/train.parquet (17917 rows) and math/MATH-500/test.parquet
(500 rows). 

"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from _paths import DATA_ROOT
from align_mopd_schema import REF_SCHEMA, write_aligned

#: ModelScope single-file endpoint. `ms_fetch.sh` uses the same one for repos.
_MS_FILE = "https://www.modelscope.cn/api/v1/datasets/{repo}/repo?Revision=master&FilePath={path}"

#: Answer-format template wrapped around every DAPO prompt; stripped at both ends.
_DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of "
    "the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
_DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'

def _fetch(repo: str, path: str, cache: Path) -> Path:
    """Fetch one file from ModelScope into the cache; no-op if already present."""
    if cache.exists() and cache.stat().st_size > 0:
        print(f"  have  {cache}")
        return cache
    url = _MS_FILE.format(repo=repo, path=path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"  get   {url}\n     -> {cache}")
    # Download to .part then rename, so an interrupted download leaves no file
    # that looks complete.
    part = cache.with_suffix(cache.suffix + ".part")
    urllib.request.urlretrieve(url, part)
    part.rename(cache)
    return cache


def _row(content: str, data_source: str, ability: str, ground_truth: str, index: str, split: str) -> dict:
    """One MOPD-schema row. Field order is set by write_aligned; this sets values."""
    return {
        "prompt": [{"content": content, "role": "user"}],
        "data_source": data_source,
        "ability": ability,
        "reward_model": {"ground_truth": ground_truth, "style": "rule"},
        "extra_info": {
            "difficulty": "",
            "index": index,
            "n_tests": 0,
            "source": data_source,
            "split": split,
        },
    }


# ----------------------------------------------------------------------------
# DAPO-Math-17k -> math/DAPO/train.parquet
# ----------------------------------------------------------------------------
def build_dapo(src: Path | None, cache_dir: Path) -> pd.DataFrame:
    src = src or _fetch(
        "BytedTsinghua-SIA/DAPO-Math-17k", "data/dapo-math-17k.parquet", cache_dir / "dapo-math-17k.parquet"
    )
    up = pd.read_parquet(src)
    print(f"  raw rows: {len(up)}")

    # Order-preserving dedup: each problem ships 100 times; keep the first.
    index = up["extra_info"].map(lambda e: str(e["index"]))
    up = up[~index.duplicated()].reset_index(drop=True)
    print(f"  unique prompts: {len(up)}")

    rows = []
    for i, r in enumerate(up.itertuples(index=False)):
        content = r.prompt[0]["content"]
        if content.startswith(_DAPO_PREFIX):
            content = content[len(_DAPO_PREFIX) :]
        if content.endswith(_DAPO_SUFFIX):
            content = content[: -len(_DAPO_SUFFIX)]
        rows.append(
            _row(content.strip(), "DAPO", "math", str(r.reward_model["ground_truth"]), f"DAPO-{i}", "train")
        )

    df = pd.DataFrame(rows)
    # The 1-shot set is sliced by row number, so order changes must fail here.
    t22 = df.iloc[22]["reward_model"]["ground_truth"]
    if t22 != "37":
        print(
            f"  WARNING: row 22's answer is {t22!r}, expected '37'. "
            "prepare_dapo_oneshot.py slices this row -- upstream ordering has changed.",
            file=sys.stderr,
        )
    return df


# ----------------------------------------------------------------------------
# MATH-500 -> math/MATH-500/test.parquet
# ----------------------------------------------------------------------------
def build_math500(src: Path | None, cache_dir: Path) -> pd.DataFrame:
    src = src or _fetch("HuggingFaceH4/MATH-500", "test.jsonl", cache_dir / "math500_test.jsonl")
    with open(src) as f:
        raw = [json.loads(line) for line in f if line.strip()]
    print(f"  raw rows: {len(raw)}")
    return pd.DataFrame(
        [
            _row(ex["problem"].strip(), "MATH-500", "math", str(ex["answer"]), f"MATH-500-{i}", "test")
            for i, ex in enumerate(raw)
        ]
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
_BUILDERS = {
    "dapo": (build_dapo, "math/DAPO/train.parquet"),
    "math500": (build_math500, "math/MATH-500/test.parquet"),
}


def _norm(v):
    """Normalization for comparison: 4.0 equals 4, surrounding whitespace ignored."""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    return v


def _verify(df: pd.DataFrame, ref: Path) -> int:
    """Compare the rebuild against an existing parquet, row by row.

    Reports two counts: byte-identical rows and equivalent rows. The exit code
    follows the latter. They differ when the upstream mirror and the reference
    file disagree on trailing whitespace or write `4` as `4.0`, neither of which
    changes scoring. Both counts are reported.
    """
    old = pd.read_parquet(ref)
    if len(old) != len(df):
        print(f"  VERIFY: row count {len(df)} != {len(old)} in {ref}")
        return abs(len(old) - len(df))

    exact = equiv = shown = 0
    for i in range(len(df)):
        a, b = df.iloc[i], old.iloc[i]
        fields = [
            (a["prompt"][0]["content"], b["prompt"][0]["content"]),
            (a["data_source"], b["data_source"]),
            (a["ability"], b["ability"]),
            (a["reward_model"]["ground_truth"], b["reward_model"]["ground_truth"]),
            (dict(a["extra_info"]), dict(b["extra_info"])),
        ]
        if all(x == y for x, y in fields):
            exact += 1
            equiv += 1
            continue
        same = True
        for x, y in fields:
            if isinstance(x, str) and x.lstrip().startswith(("{", "[")):
                try:
                    x, y = json.loads(x), json.loads(y)
                except json.JSONDecodeError:
                    pass
            if _norm(x) != _norm(y):
                same = False
        if same:
            equiv += 1
        elif shown < 3:
            shown += 1
            print(f"  VERIFY: row {i} differs beyond formatting")

    n = len(df)
    print(f"  VERIFY vs {ref}: {exact}/{n} byte-identical, {equiv}/{n} equivalent")
    return n - equiv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", dest="which", choices=[*_BUILDERS, "all"], required=True)
    p.add_argument("--src", type=Path, default=None, help="pre-downloaded upstream file; otherwise fetched from ModelScope")
    p.add_argument("--out", type=Path, default=None, help="output path; defaults to the standard location under <data-root>")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--cache-dir", type=Path, default=None, help="download cache for upstream files; defaults to <data-root>/.cache/upstream")
    p.add_argument("--verify", type=Path, default=None, help="compare row by row against this parquet; nonzero exit on mismatch")
    a = p.parse_args()

    if a.which == "all" and (a.src or a.out or a.verify):
        raise SystemExit("--src / --out / --verify are invalid with --set all (two files)")

    cache_dir = a.cache_dir or (a.data_root / ".cache" / "upstream")
    schema = REF_SCHEMA()
    names = list(_BUILDERS) if a.which == "all" else [a.which]

    bad = 0
    for name in names:
        fn, rel = _BUILDERS[name]
        out = a.out or (a.data_root / rel)
        print(f"\n=== build [{name}] -> {out} ===")
        df = fn(a.src, cache_dir)
        if a.verify:
            bad += _verify(df, a.verify)
        write_aligned(df, out, schema)
        print(f"  saved: {len(df)} rows -> {out}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
