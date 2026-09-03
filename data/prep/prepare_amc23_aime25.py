"""Build the AMC23 and AIME25 evaluation sets.

    python data/prep/prepare_amc23_aime25.py amc23
    python data/prep/prepare_amc23_aime25.py aime25
    python data/prep/prepare_amc23_aime25.py amc23 --verify existing.parquet

Writes math/AMC23/test.parquet and math/AIME25/test.parquet. These are not
acceptance criteria; they are the other two columns of the math baseline.
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

#: Same endpoint as prepare_dapo_math500.py.
_MS_FILE = "https://www.modelscope.cn/api/v1/datasets/{repo}/repo?Revision=master&FilePath={path}"


def _fetch(repo: str, path: str, cache: Path) -> Path:
    """Fetch one file from ModelScope into the cache; no-op if already present."""
    if cache.exists() and cache.stat().st_size > 0:
        print(f"  have  {cache}")
        return cache
    url = _MS_FILE.format(repo=repo, path=path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"  get   {url}\n     -> {cache}")
    part = cache.with_suffix(cache.suffix + ".part")
    urllib.request.urlretrieve(url, part)
    part.rename(cache)
    return cache


def _row(content: str, data_source: str, ground_truth: str, index: str) -> dict:
    """One MOPD-schema row. Field order is set by write_aligned; this sets values."""
    return {
        "prompt": [{"content": content, "role": "user"}],
        "data_source": data_source,
        "ability": "math",
        "reward_model": {"ground_truth": ground_truth, "style": "rule"},
        "extra_info": {
            "difficulty": "",
            "index": index,
            "n_tests": 0,
            "source": data_source,
            "split": "test",
        },
    }


def _expect(n: int, want: int, name: str) -> None:
    """Stop if the problem count is wrong rather than writing a plausible file.

    An upstream revision is the only silent failure here: a differing count does
    not raise, it just makes scores incomparable with the reference table.
    """
    if n != want:
        raise SystemExit(f"{name}: upstream returned {n} rows, expected {want} -- upstream changed; investigate before editing this number")


# ----------------------------------------------------------------------------
# AMC23 -> math/AMC23/test.parquet
# ----------------------------------------------------------------------------
def build_amc23(src: Path | None, cache_dir: Path) -> pd.DataFrame:
    src = src or _fetch("knoveleng/AMC-23", "data/train-00000-of-00001.parquet", cache_dir / "amc23.parquet")
    up = pd.read_parquet(src)
    print(f"  raw rows: {len(up)}")
    _expect(len(up), 40, "AMC23")
    # Upstream carries both `problem` and `question` with identical content.
    # Take `problem`, the name MATH-500 also uses.
    return pd.DataFrame(
        [
            _row(str(r.problem).strip(), "AMC23", str(r.answer).strip(), f"AMC23-{i}")
            for i, r in enumerate(up.itertuples(index=False))
        ]
    )


# ----------------------------------------------------------------------------
# AIME25 -> math/AIME25/test.parquet
# ----------------------------------------------------------------------------
def build_aime25(src: Path | None, cache_dir: Path) -> pd.DataFrame:
    """AIME 2025 ships as two parts (I / II) of 15 problems, joined I then II.

    `--src` names a single path, so when given it is read as the joined 30 rows;
    otherwise both parts are fetched from ModelScope.
    """
    if src is not None:
        raw = [json.loads(line) for line in Path(src).read_text().splitlines() if line.strip()]
    else:
        raw = []
        for part in ("I", "II"):
            p = _fetch("opencompass/AIME2025", f"aime2025-{part}.jsonl", cache_dir / f"aime2025-{part}.jsonl")
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            _expect(len(rows), 15, f"AIME25 part {part}")
            raw += rows
    print(f"  raw rows: {len(raw)}")
    _expect(len(raw), 30, "AIME25")
    return pd.DataFrame(
        [
            _row(ex["question"].strip(), "AIME25", str(ex["answer"]).strip(), f"AIME25-{i}")
            for i, ex in enumerate(raw)
        ]
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
_BUILDERS = {
    "amc23": (build_amc23, "math/AMC23/test.parquet"),
    "aime25": (build_aime25, "math/AIME25/test.parquet"),
}


def _verify(df: pd.DataFrame, ref: Path) -> int:
    """Compare the rebuild against an existing parquet; return the mismatch count."""
    old = pd.read_parquet(ref)
    if len(old) != len(df):
        print(f"  VERIFY: row count {len(df)} != {len(old)} in {ref}")
        return abs(len(old) - len(df))
    bad = 0
    for i in range(len(df)):
        a, b = df.iloc[i], old.iloc[i]
        if a["prompt"][0]["content"].strip() != b["prompt"][0]["content"].strip() or str(
            a["reward_model"]["ground_truth"]
        ).strip() != str(b["reward_model"]["ground_truth"]).strip():
            bad += 1
            if bad <= 3:
                print(f"  VERIFY: row {i} differs")
    print(f"  VERIFY: {len(df) - bad}/{len(df)} rows match {ref}")
    return bad


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("which", choices=[*_BUILDERS, "all"])
    p.add_argument("--src", type=Path, help="local file to use instead of downloading from ModelScope")
    p.add_argument("--out", type=Path, help="output path; valid only for a single subcommand")
    p.add_argument("--cache-dir", type=Path, default=DATA_ROOT / ".cache")
    p.add_argument("--verify", type=Path, help="compare against this parquet instead of writing")
    args = p.parse_args(argv)

    names = list(_BUILDERS) if args.which == "all" else [args.which]
    if len(names) > 1 and (args.src or args.out or args.verify):
        p.error("--src / --out / --verify apply to a single subcommand only")

    rc = 0
    for name in names:
        build, rel = _BUILDERS[name]
        print(f"== {name}")
        df = build(args.src, args.cache_dir)
        if args.verify:
            rc |= 1 if _verify(df, args.verify) else 0
            continue
        out = args.out or DATA_ROOT / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        write_aligned(df, out, REF_SCHEMA())
        print(f"  wrote {len(df)} rows -> {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
