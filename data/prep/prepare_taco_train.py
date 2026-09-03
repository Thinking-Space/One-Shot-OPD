"""Build the code training set from TACO.

    python data/prep/prepare_taco_train.py
    python data/prep/prepare_taco_train.py --max-tests 5

Writes code/TACO/train.parquet (24701 rows). The 9 source shards (~4.2 GB)
are fetched from ModelScope into $TACO_CACHE/train (default
data/.cache/taco/train) unless already there, resuming interrupted downloads;
`ms_fetch.sh datasets BAAI/TACO "$TACO_CACHE" train` fetches the same files
with curl. TACO_SHARD_URL, a template with {fname}, points the download at
another mirror, e.g.
https://hf-mirror.com/datasets/BAAI/TACO/resolve/main/train/{fname}.

Keeps 10 test cases per problem by default (--max-tests 0 for all). Problems
ship 30-200+ tests, and reward computation runs on every training step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _paths import DATA_ROOT
from _ms import fetch
from align_mopd_schema import REF_SCHEMA, write_aligned


CODE_PROMPT_TEMPLATE = (
    "You will be given a question (problem specification) and will generate a correct Python "
    "program that matches the specification and passes all tests.\n\nQuestion: {question}\n\n"
    "Read the inputs from stdin, solve the problem, and write the answer to stdout. "
    "Enclose your code within delimiters as follows.\n```python\n# YOUR CODE HERE\n```"
)


def _truncate_tests(input_output_str: str, max_tests: int) -> tuple[str, int]:
    """Parse a TACO input_output string, truncated to the first max_tests tests.

    Args:
        input_output_str: raw TACO input_output field (JSON string).
        max_tests: maximum tests to keep; 0 keeps all.

    Returns:
        (new JSON string, tests kept). Returns ("", 0) if parsing fails.
    """
    if not input_output_str or not isinstance(input_output_str, str):
        return "", 0
    try:
        parsed = json.loads(input_output_str)
    except (json.JSONDecodeError, ValueError):
        return "", 0
    inputs = parsed.get("inputs", [])
    outputs = parsed.get("outputs", [])
    n = min(len(inputs), len(outputs))
    if n == 0:
        return "", 0
    if max_tests > 0 and n > max_tests:
        inputs = inputs[:max_tests]
        outputs = outputs[:max_tests]
        n = max_tests
    truncated = {"inputs": inputs, "outputs": outputs}
    fn_name = parsed.get("fn_name")
    if fn_name:
        truncated["fn_name"] = fn_name
    return json.dumps(truncated), n


def build_taco_train(
    difficulties: list[str] | None = None,
    max_rows: int | None = None,
    max_tests: int = 10,
) -> pd.DataFrame:
    """Load the BAAI/TACO training set into the verl parquet schema.

    Args:
        difficulties: difficulty filter (None keeps all). One of EASY, MEDIUM,
                      MEDIUM_HARD, HARD, VERY_HARD.
        max_rows: row limit; None for no limit.
        max_tests: tests kept per problem, bounding reward cost. 0 keeps all.

    Returns:
        pd.DataFrame with verl schema.
    """
    print("Loading BAAI/TACO train split ...")
    import os

    # Not /tmp: it is periodically cleared on many machines, which would discard
    # the 4.2 GB of shards. Defaults under data/; TACO_CACHE overrides.
    cache_dir = Path(os.environ.get("TACO_CACHE") or (DATA_ROOT / ".cache" / "taco")) / "train"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # datasets 3.x dropped dataset scripts (TACO.py); load the arrow files directly.
    # ModelScope's copy has the same train/data-*.arrow layout and bytes as
    # huggingface.co, which (like hf-mirror.com) is unreachable from the
    # training cluster -- see data/prep/README.md. Not HF_ENDPOINT as the
    # override: shells here export it as hf-mirror.com, which is the host that
    # fails.
    url_template = os.environ.get("TACO_SHARD_URL")
    n_shards = 9
    for i in range(n_shards):
        fname = f"data-{i:05d}-of-{n_shards:05d}.arrow"
        url = url_template.format(fname=fname) if url_template else None
        fetch("BAAI/TACO", f"train/{fname}", cache_dir / fname, url=url)

    from datasets import Dataset

    ds = Dataset.from_file(str(cache_dir / "data-00000-of-00009.arrow"))
    for i in range(1, n_shards):
        fname = f"data-{i:05d}-of-{n_shards:05d}.arrow"
        shard = Dataset.from_file(str(cache_dir / fname))
        from datasets import concatenate_datasets

        ds = concatenate_datasets([ds, shard])
    print(f"  raw rows: {len(ds)}")

    if difficulties:
        difficulties_set = set(d.upper() for d in difficulties)
        ds = ds.filter(lambda x: (x.get("difficulty") or "").upper() in difficulties_set)
        print(f"  after difficulty filter ({difficulties}): {len(ds)}")

    if max_rows and len(ds) > max_rows:
        ds = ds.select(range(max_rows))
        print(f"  truncated to: {len(ds)}")

    rows = []
    skipped_no_q = 0
    skipped_no_test = 0
    n_tests_total = 0
    for idx, ex in enumerate(ds):
        question = ex.get("question", "")
        if not question or len(question.strip()) < 20:
            skipped_no_q += 1
            continue

        gt_str, n_kept = _truncate_tests(ex.get("input_output", ""), max_tests)
        if n_kept == 0:
            skipped_no_test += 1
            continue
        n_tests_total += n_kept

        prompt_text = CODE_PROMPT_TEMPLATE.format(question=question)
        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt_text}],
                "data_source": "taco",
                "ability": "code",
                "reward_model": {"ground_truth": gt_str, "style": "rule"},
                "extra_info": {
                    "index": f"taco-{idx}",
                    "split": "train",
                    "difficulty": ex.get("difficulty", ""),
                    "source": ex.get("source", ""),
                    "n_tests": n_kept,
                },
            }
        )
    avg_tests = n_tests_total / max(len(rows), 1)
    print(
        f"  kept: {len(rows)}, skipped (no question: {skipped_no_q}, no test: {skipped_no_test}), "
        f"avg tests/problem: {avg_tests:.1f}"
    )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_ROOT / "code" / "TACO" / "train.parquet",
        help="output parquet path.",
    )
    parser.add_argument(
        "--difficulties",
        nargs="*",
        default=None,
        help="difficulty filter, e.g. EASY MEDIUM. Defaults to all.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="row limit.",
    )
    parser.add_argument(
        "--max-tests",
        type=int,
        default=10,
        help="tests kept per problem, bounding reward cost. 0 keeps all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_taco_train(
        difficulties=args.difficulties, max_rows=args.max_rows, max_tests=args.max_tests
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Through write_aligned, not df.to_parquet: TACO is one of MOPD's six
    # concatenated files, and plain to_parquet leaves the arrow types to pandas'
    # inference -- which is where `string` vs `large_string` came from in the
    # first place (align_mopd_schema.py), and that one does stop the concat.
    write_aligned(df, args.out, REF_SCHEMA())
    print(f"\nSaved: {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
