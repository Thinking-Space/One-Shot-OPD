"""Build the BFCL-v3 evaluation sets used for function calling.

    python data/prep/prepare_bfcl_eval.py
    python data/prep/prepare_bfcl_eval.py --subset simple

Writes agentic/BFCL/{simple,multiple,parallel,parallel_multiple,live_simple}/
test.parquet (400/200/200/200/258 rows).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

# Reuse the xlam training prompt template so training and validation match.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from prepare_xlam_train import FC_SYSTEM_PROMPT_TEMPLATE  # noqa: E402

from _paths import DATA_ROOT  # noqa: E402


#: No default path -- where the BFCL checkout lives is a property of the
#: machine. `--src` overrides it.
DEFAULT_SRC = os.environ.get("BFCL_DIR", "")

SUBSETS = ["simple", "multiple", "parallel", "parallel_multiple", "live_simple"]


def _load_jsonl(path: Path) -> list[dict]:
    """Read a jsonl file, one JSON object per line."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_messages(question: list[list[dict]], tools: list[dict]) -> list[dict]:
    """Build chat messages from the BFCL question and function fields.

    Args:
        question: BFCL question field, shaped [[{role, content}, ...]]. The
                  outer list is turns (always len 1 here); the inner list is
                  the message sequence.
        tools   : BFCL function field, list[{name, description, parameters}].

    Returns:
        [{"role":"system", "content": tools + format}, ...messages from question]
    """
    tools_pretty = json.dumps(tools, indent=2, ensure_ascii=False)
    system_msg = {
        "role": "system",
        "content": FC_SYSTEM_PROMPT_TEMPLATE.format(tools_json=tools_pretty),
    }
    if not question or not isinstance(question, list) or not question[0]:
        return [system_msg]
    msgs = [system_msg]
    for m in question[0]:
        if isinstance(m, dict) and "role" in m and "content" in m:
            msgs.append({"role": m["role"], "content": m["content"]})
    return msgs


def build_subset(src: Path, subset: str) -> pd.DataFrame:
    """Build the verl parquet dataframe for one BFCL subset.

    Args:
        src   : BFCL data directory.
        subset: subset name from SUBSETS, without the "BFCL_v3_" prefix.

    Returns:
        pd.DataFrame with verl schema.
    """
    q_path = src / f"BFCL_v3_{subset}.json"
    gt_path = src / "possible_answer" / f"BFCL_v3_{subset}.json"
    if not q_path.exists():
        raise FileNotFoundError(f"BFCL prompt file not found: {q_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"BFCL GT file not found: {gt_path}")

    print(f"[{subset}] loading {q_path.name} ...")
    questions = _load_jsonl(q_path)
    print(f"[{subset}] loading {gt_path.name} ...")
    gts = _load_jsonl(gt_path)

    # Align on id; usually already aligned.
    gt_by_id = {g["id"]: g["ground_truth"] for g in gts if "id" in g}
    print(f"[{subset}] prompts: {len(questions)}, GTs: {len(gts)}")

    rows = []
    skipped = 0
    for ex in questions:
        ex_id = ex.get("id")
        if ex_id not in gt_by_id:
            skipped += 1
            continue
        question = ex.get("question", [])
        tools = ex.get("function", [])
        if not tools:
            skipped += 1
            continue

        msgs = _build_messages(question, tools)
        gt = gt_by_id[ex_id]
        gt_json = json.dumps(gt, ensure_ascii=False)

        rows.append(
            {
                "prompt": msgs,
                "data_source": f"bfcl_{subset}",
                "ability": "function_calling",
                "reward_model": {"ground_truth": gt_json, "style": "rule"},
                "extra_info": {
                    "index": str(ex_id),
                    "split": "test",
                    "subset": subset,
                    "n_tools": len(tools),
                    "n_calls": len(gt) if isinstance(gt, list) else 1,
                },
            }
        )

    print(f"[{subset}] kept: {len(rows)}, skipped: {skipped}")
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(DEFAULT_SRC) if DEFAULT_SRC else None,
        help="BFCL data directory (defaults to $BFCL_DIR).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=[*SUBSETS, "all"],
        help="BFCL subset name, or 'all' for all five.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_ROOT / "agentic" / "BFCL",
        help="output root; each subset is written to <out-dir>/<subset>/test.parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.src is None or not args.src.is_dir():
        raise SystemExit(
            f"BFCL directory not found: {args.src or '<--src / $BFCL_DIR unset>'}\n"
            "Clone gorilla-llm/Berkeley-Function-Calling-Leaderboard and point --src at it."
        )
    targets = SUBSETS if args.subset == "all" else [args.subset]
    for s in targets:
        df = build_subset(args.src, s)
        out_path = args.out_dir / s / "test.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"Saved: {len(df)} rows -> {out_path}\n")
    print("Done.")


if __name__ == "__main__":
    main()
