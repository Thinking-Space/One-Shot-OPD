"""Build the code evaluation sets.

    python data/prep/prepare_code_eval.py --bench lcb_v6
    python data/prep/prepare_code_eval.py --bench all

Writes code/LCBv6/test.parquet (175 rows), code/HumanEvalPlus/test.parquet
(164 rows) and code/MBPPPlus/test.parquet (378 rows). data_source must stay
"livecodebench" or the verifier silently falls back to the math branch.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import zlib
from pathlib import Path
from typing import Any

import pandas as pd

from _paths import DATA_ROOT
from _ms import fetch
from align_mopd_schema import REF_SCHEMA, write_aligned


# ----------------------------------------------------------------------------
# Prompt templates
# ----------------------------------------------------------------------------
HUMANEVAL_PROMPT_TEMPLATE = (
    "Complete the following Python function. Only output the full function in a markdown "
    "```python``` code block, do not include test cases.\n\n```python\n{prompt}```"
)

MBPP_PROMPT_TEMPLATE = (
    "You are an expert Python programmer. Solve the following task and put the final "
    "function in a markdown ```python``` code block.\n\nTask: {text}\n\n"
    "Your code should pass these tests (treat them as a specification):\n```python\n{test_example}\n```"
)

LCB_STDIN_PROMPT_TEMPLATE = (
    "You will be given a question (problem specification) and will generate a correct Python "
    "program that matches the specification and passes all tests.\n\nQuestion: {question}\n\n"
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not "
    "directly test on the sample inputs). Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, runs the algorithm and "
    "writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```"
)

LCB_FUNCTIONAL_PROMPT_TEMPLATE = (
    "You will be given a question (problem specification) and will generate a correct Python "
    "program that matches the specification and passes all tests.\n\nQuestion: {question}\n\n"
    "You will use the following starter code to write the solution to the problem and "
    "enclose your code within delimiters.\n```python\n{starter_code}\n```"
)


# ----------------------------------------------------------------------------
# Build: HumanEval+ and MBPP+
# ----------------------------------------------------------------------------
#: evalscope's ModelScope mirrors of evalplus/humanevalplus and evalplus/mbppplus.
#: The parquets are byte-identical to the huggingface.co originals (same sha256
#: as the LFS objects), which are unreachable from the cluster.
_EVALPLUS_MS = {
    "humanevalplus": ("evalscope/humanevalplus", "data/test-00000-of-00001-5973903632b82d40.parquet", 164),
    "mbppplus": ("evalscope/mbppplus", "data/test-00000-of-00001-d5781c9c51e02795.parquet", 378),
}


def _load_evalplus(name: str) -> list[dict]:
    """Rows of one evalplus benchmark, fetched into <data-root>/.cache on first use."""
    repo, path, want = _EVALPLUS_MS[name]
    src = fetch(repo, path, DATA_ROOT / ".cache" / f"{name}.parquet")
    rows = pd.read_parquet(src).to_dict("records")
    print(f"  raw rows: {len(rows)}")
    if len(rows) != want:
        raise SystemExit(
            f"{name}: upstream returned {len(rows)} rows, expected {want} -- upstream changed; "
            "investigate before editing this number"
        )
    return rows


def build_humanevalplus() -> pd.DataFrame:
    """Load evalplus/humanevalplus into a parquet dataframe.

    Returns:
        pd.DataFrame with verl schema.
    """
    print("Loading evalplus/humanevalplus from ModelScope...")
    ds = _load_evalplus("humanevalplus")

    rows = []
    for idx, ex in enumerate(ds):
        prompt_text = HUMANEVAL_PROMPT_TEMPLATE.format(prompt=ex["prompt"])
        # The evalplus test field defines check(candidate) without calling it.
        # Append check(<entry_point>), or nothing is asserted and wrong code passes.
        test_code = ex["test"]
        entry_point = ex["entry_point"]
        if entry_point and f"check({entry_point})" not in test_code:
            test_code = test_code.rstrip() + f"\n\ncheck({entry_point})\n"
        ground_truth = json.dumps(
            {
                "test": test_code,
                "entry_point": entry_point,
            }
        )
        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt_text}],
                "data_source": "humanevalplus",
                "ability": "code",
                "reward_model": {"ground_truth": ground_truth, "style": "rule"},
                "extra_info": {
                    "difficulty": "",
                    "index": f"humanevalplus-{idx}",
                    "n_tests": 0,
                    "source": "humanevalplus",
                    "split": "test",
                },
            }
        )
    return pd.DataFrame(rows)


def build_mbppplus() -> pd.DataFrame:
    """Load evalplus/mbppplus into a parquet dataframe."""
    print("Loading evalplus/mbppplus from ModelScope...")
    ds = _load_evalplus("mbppplus")

    rows = []
    for idx, ex in enumerate(ds):
        # MBPP+ supplies one example assert as a signature hint. read_parquet
        # hands the field over as a numpy array, hence list() before any `if`.
        test_list = list(ex.get("test_list") if ex.get("test_list") is not None else [])
        test_example = test_list[0] if test_list else ""
        prompt_text = MBPP_PROMPT_TEMPLATE.format(
            text=ex.get("prompt") or ex.get("text", ""),
            test_example=test_example,
        )
        # evalplus fields: prompt(=text), test(=augmented), code(=canonical), test_list (sanitized), entry_point
        ground_truth = json.dumps(
            {
                "test": ex["test"],
                "entry_point": ex.get("entry_point", ""),
            }
        )
        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt_text}],
                "data_source": "mbppplus",
                "ability": "code",
                "reward_model": {"ground_truth": ground_truth, "style": "rule"},
                "extra_info": {
                    "difficulty": "",
                    "index": f"mbppplus-{idx}",
                    "n_tests": len(test_list),
                    "source": "mbppplus",
                    "split": "test",
                },
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Build: LiveCodeBench v6 (175-problem increment)
# ----------------------------------------------------------------------------
def _parse_lcb_tests(ex: dict) -> list[dict]:
    """LCB public/private_test_cases are JSON or base64+zlib+pickle strings.

    Returns a uniform list[{input, output, testtype, metadata}].
    """
    public = ex.get("public_test_cases", "[]")
    private = ex.get("private_test_cases", "[]")
    metadata_str = ex.get("metadata", "{}")

    metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
    fn_name = metadata.get("func_name", None)

    def _load(blob):
        if not isinstance(blob, str) or not blob:
            return []
        try:
            return json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            pass
        # Try base64+zlib+pickle (the common LCB v5+ private-case format).
        try:
            decoded = pickle.loads(zlib.decompress(base64.b64decode(blob.encode("utf-8"))))
            if isinstance(decoded, str):
                return json.loads(decoded)
            return decoded
        except Exception:
            return []

    cases = []
    for t in _load(public) + _load(private):
        case = {
            "input": t.get("input", ""),
            "output": t.get("output", ""),
            "testtype": t.get("testtype", "stdin"),
            "metadata": {},
        }
        if case["testtype"] == "functional" and fn_name is not None:
            case["metadata"]["func_name"] = fn_name
        cases.append(case)
    return cases


#: hf-mirror was reachable when this was written and is not any more (SNI reset,
#: same as huggingface.co -- see data/prep/README.md), so the default moved to
#: the ModelScope mirror of the same repo, which answers 200 and serves the same
#: 134 MB test6.jsonl. Override the whole URL with LCB_V6_JSONL_URL, or drop the
#: file into the cache path below by hand.
_LCB_V6_JSONL_URL = os.environ.get(
    "LCB_V6_JSONL_URL",
    "https://www.modelscope.cn/api/v1/datasets/livecodebench/code_generation_lite/repo"
    "?Revision=master&FilePath=test6.jsonl",
)


def _ensure_lcb_v6_jsonl(cache_path: Path) -> Path:
    """Download the LCB v6 jsonl (175-problem increment) into the local cache."""
    if cache_path.exists() and cache_path.stat().st_size > 1_000_000:
        return cache_path
    import urllib.request

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {_LCB_V6_JSONL_URL} -> {cache_path}")
    urllib.request.urlretrieve(_LCB_V6_JSONL_URL, cache_path)
    return cache_path


def build_lcb_v6() -> pd.DataFrame:
    """Load the LiveCodeBench v6 increment (test6.jsonl, 175 problems).

    Recent datasets versions disable dataset scripts, so the jsonl is fetched
    directly from the mirror.
    """
    # Under <data-root>/.cache, not /tmp: 134 MB over a link that has already
    # been unreliable once, on a machine whose /tmp is a scratch disk that gets
    # wiped. DATA_BASE points at the persistent one.
    cache_path = DATA_ROOT / ".cache" / "lcb_test6.jsonl"
    _ensure_lcb_v6_jsonl(cache_path)

    print(f"Loading LCB v6 jsonl from {cache_path}...")
    ds: list[dict] = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ds.append(json.loads(line))
    print(f"  raw rows: {len(ds)}")

    rows = []
    skipped = 0
    for idx, ex in enumerate(ds):
        starter_code = ex.get("starter_code") or ""
        question = ex.get("question_content", "")
        if starter_code:
            prompt_text = LCB_FUNCTIONAL_PROMPT_TEMPLATE.format(
                question=question, starter_code=starter_code
            )
        else:
            prompt_text = LCB_STDIN_PROMPT_TEMPLATE.format(question=question)

        tests = _parse_lcb_tests(ex)
        if not tests:
            skipped += 1
            continue

        ground_truth = json.dumps(tests)
        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt_text}],
                "data_source": "livecodebench",
                "ability": "code",
                "reward_model": {"ground_truth": ground_truth, "style": "rule"},
                "extra_info": {
                    "difficulty": str(ex.get("difficulty", "")),
                    "index": f"lcb_v6-{idx}",
                    "n_tests": len(tests),
                    "source": f"LCBv6-{ex.get('platform', '')}",
                    "split": "test",
                },
            }
        )
    print(f"  kept: {len(rows)}, skipped (no tests): {skipped}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
_BUILDERS = {
    "humanevalplus": (build_humanevalplus, "code/HumanEvalPlus/test.parquet"),
    "mbppplus": (build_mbppplus, "code/MBPPPlus/test.parquet"),
    "lcb_v6": (build_lcb_v6, "code/LCBv6/test.parquet"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--bench",
        choices=list(_BUILDERS.keys()) + ["all"],
        required=True,
        help="benchmark to build.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output parquet path; defaults to <data-root>/code/<Bench>/test.parquet.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="data root used to build the default --out. Defaults to <repo>/data (DATA_BASE overrides).",
    )
    return parser.parse_args()


def _save_one(name: str, builder, default_out: str, root: Path, override: Path | None) -> None:
    out = override if override else (root / default_out)
    print(f"\n=== build [{name}] -> {out} ===")
    df = builder()
    out.parent.mkdir(parents=True, exist_ok=True)
    # write_aligned, not to_parquet: LCBv6 is one of the six files MOPD
    # concatenates, and `datasets` refuses the concat if the arrow types or the
    # extra_info fields differ from its siblings. It used to be aligned after
    # the fact with `align_mopd_schema.py --src/--dst`; nothing enforced that
    # second step, and it stopped being optional once the file no longer
    # shipped and every user had to build it here.
    write_aligned(df, out, REF_SCHEMA())
    print(f"  saved: {len(df)} rows, {len(df.columns)} cols -> {out}")


def main() -> None:
    args = parse_args()
    if args.bench == "all":
        if args.out is not None:
            raise ValueError("--out is invalid with --bench all (multiple outputs)")
        for name, (fn, default_out) in _BUILDERS.items():
            _save_one(name, fn, default_out, args.data_root, None)
    else:
        fn, default_out = _BUILDERS[args.bench]
        _save_one(args.bench, fn, default_out, args.data_root, args.out)


if __name__ == "__main__":
    main()
