"""Build the function-calling training set from xlam.

    python data/prep/prepare_xlam_train.py --src <xlam-file>

Writes agentic/xlam/train.parquet (60000 rows).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from _paths import DATA_ROOT


#: No default path: this file lives wherever whoever accepted xlam's licence
#: put it, which is a property of the machine and not of this repo.
DEFAULT_SRC = os.environ.get("XLAM_JSON", "")

# Function-calling system prompt: lists the tools schema and requires JSON list
# output. Both the teacher (Hammer) and the student (Qwen2.5-Coder) emit function
# calls in this format.
FC_SYSTEM_PROMPT_TEMPLATE = (
    "You are a function calling assistant. You have access to the following tools:\n\n"
    "{tools_json}\n\n"
    "For the user request, decide which tool(s) to call. Respond ONLY with a JSON "
    'list of function calls in the format: [{{"name": "func_name", "arguments": '
    '{{"arg_name": value, ...}}}}, ...]. If no tool is suitable, respond with '
    "an empty list []."
)


def build_fc_prompt(query: str, tools_str: str) -> list[dict]:
    """Build the chat message list for a function-calling row.

    Args:
        query    : user question (xlam 'query' field, raw string).
        tools_str: tools schema as JSON (xlam 'tools' field).

    Returns:
        [{"role":"system", "content": ...}, {"role":"user", "content": query}]
    """
    # Reformat tools_str for readability in the system prompt.
    try:
        tools_obj = json.loads(tools_str)
        tools_pretty = json.dumps(tools_obj, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        tools_pretty = tools_str  # fallback

    system_content = FC_SYSTEM_PROMPT_TEMPLATE.format(tools_json=tools_pretty)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]


def _safe_json_loads(s: str) -> object:
    """json.loads that returns None on failure."""
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def build_xlam_train(src: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load the xlam-60k JSON into the verl parquet schema.

    Args:
        src     : path to xlam_function_calling_60k.json.
        max_rows: row limit; None for all.

    Returns:
        pd.DataFrame with verl schema.
    """
    print(f"Loading {src} ...")
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  raw rows: {len(data)}")

    if max_rows is not None and len(data) > max_rows:
        data = data[:max_rows]
        print(f"  truncated to: {len(data)}")

    rows = []
    skipped_no_q = 0
    skipped_no_tools = 0
    skipped_bad_answers = 0
    for ex in data:
        query = ex.get("query", "")
        tools_str = ex.get("tools", "")
        answers_str = ex.get("answers", "")
        idx = ex.get("id", "")

        if not query or not isinstance(query, str) or len(query.strip()) < 3:
            skipped_no_q += 1
            continue
        tools_obj = _safe_json_loads(tools_str)
        if not isinstance(tools_obj, list) or len(tools_obj) == 0:
            skipped_no_tools += 1
            continue
        answers_obj = _safe_json_loads(answers_str)
        if not isinstance(answers_obj, list):
            skipped_bad_answers += 1
            continue

        prompt = build_fc_prompt(query, tools_str)
        rows.append(
            {
                "prompt": prompt,
                "data_source": "xlam",
                "ability": "function_calling",
                "reward_model": {
                    "ground_truth": answers_str,  # raw JSON string, parsed by fc_eval
                    "style": "rule",
                },
                "extra_info": {
                    "index": f"xlam-{idx}",
                    "split": "train",
                    "n_tools": len(tools_obj),
                    "n_calls": len(answers_obj),
                },
            }
        )

    print(
        f"  kept: {len(rows)}, skipped "
        f"(no_query: {skipped_no_q}, no_tools: {skipped_no_tools}, "
        f"bad_answers: {skipped_bad_answers})"
    )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(DEFAULT_SRC) if DEFAULT_SRC else None,
        help="xlam_function_calling_60k.json (defaults to $XLAM_JSON).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_ROOT / "agentic" / "xlam" / "train.parquet",
        help="output parquet path.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="row limit, for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.src is None or not args.src.exists():
        raise SystemExit(
            f"xlam source not found: {args.src or '<--src / $XLAM_JSON unset>'}\n"
            "Get Salesforce/xlam-function-calling-60k (it is gated -- you accept its "
            "licence on the hub), then point --src at xlam_function_calling_60k.json."
        )
    df = build_xlam_train(args.src, max_rows=args.max_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nSaved: {len(df)} rows -> {args.out}")
    print("\n== sample row 0 ==")
    print(df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
