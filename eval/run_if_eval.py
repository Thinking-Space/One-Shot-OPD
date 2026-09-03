#!/usr/bin/env python3
"""Offline IF evaluation: Multi-IF (3 turns x 8 languages).

    python3 eval/run_if_eval.py multiif --model <hf-dir> --out results/multiif_s50
    python3 eval/run_if_eval.py rescore --out results/multiif_s50   # no GPU

Why this exists at all: Multi-IF is the IF domain's only benchmark here, and
in-training validation structurally cannot produce it. verl generates one
response per prompt and scores it; turn 2's prompt ("Your response should end
with the exact phrase ...") only means something with turn 1's answer in front
of it. Nothing in the validation loop can carry a conversation forward, so the
benchmark has to run outside it -- and the IF training run therefore has no
in-training validation at all.

IFBench used to be the second subcommand here. It was never a criterion; it
existed to give the val loop something to chew on. Removed 2026-08-30.

Sampling defaults are the reference ones (§4.3): T=0.6, top_p=0.95,
max_out_len=16384, one sample per prompt. They are flags, but changing them
makes the numbers incomparable with the criterion table, so they are not
defaults to tune.

## Two things that are not the reference setup, and are said so out loud

1. **The full benchmark, not the "mini" subsample.** The reference numbers come
   from an internal OpenCompass dataset called `multi_if_mini_max16k_gen_81c652`
   -- roughly an 11% stratified slice, ~50-100 conversations per language. This
   runs all 4445. The headline metric averages over languages rather than over
   rows, so the two are comparable in expectation and the full set is the less
   noisy of them; but a per-language cell here rests on ~500 conversations and
   there on ~60, and a 3-point gap at n=60 is not evidence of anything.
2. **Reasoning is stripped from the dialogue history.** A distilled R1 student
   answers inside `<think>...</think>`. The graded text is always the stripped
   answer (that half matches, and matches `if_eval.reward_func`), but what gets
   appended to the conversation before turn 2 is a choice. Default is the
   stripped answer: that is what DeepSeek's own guidance says to do with R1
   history, and keeping three 16k reasoning blocks would push turn 3's context
   past the model length for no benefit. `--history-keeps-reasoning` switches.

## Aggregation, which is not the obvious one

The 8-language score is the **unweighted mean of the eight per-language
scores**, not a pooled mean over conversations. This matters: the corpus is
English 896 / Chinese 454, so pooling would weight English twice Chinese.
Verified against the reference tables -- the per-language means in
`C4_MultiIF_byLang.csv` reproduce `C3_MultiIF_langAvg.csv` exactly, and the
student's (38.38 + 26.47 + 20.84) / 3 = 28.56 is the reference s0 for this
domain.

A turn's `overall` is the mean of its own four sub-metrics, which is
OpenCompass's `MultiIFEvaluator` definition, not something invented here.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import statistics
import sys
import traceback
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "verl"))

from verl.utils.reward_score import if_eval  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
TURNS = (1, 2, 3)
SUB_METRICS = ("prompt_strict", "inst_strict", "prompt_loose", "inst_loose")

# Headroom between the prompt length we measure and the one vLLM builds. We
# render the chat template ourselves to decide what fits; vLLM renders it again
# to run. Same template, but a few tokens of drift would turn a fits/does-not
# decision into the crash this guard exists to prevent.
_PROMPT_RESERVE = 64


# ---------------------------------------------------------------------------
# scoring -- pure, no GPU, no vLLM. `rescore` re-enters here on saved output.
# ---------------------------------------------------------------------------


def _score_response(family: str, turn: dict, response: str) -> dict[str, float]:
    """One response against one turn's instruction list -> the four raw counts.

    Counts rather than rates, because prompt-level and instruction-level
    aggregate over different denominators: a prompt is one unit whatever its
    instruction count, an instruction is one unit each. Averaging per-row rates
    would silently make prompt-level and instruction-level the same number.
    """
    r = if_eval.score_single_turn(
        family,
        turn["prompt"],
        list(turn["instruction_id_list"]),
        list(turn["kwargs"]),
        response,
    )
    return {
        "prompt_strict_correct": r["if_prompt_strict_correct"],
        "prompt_loose_correct": r["if_prompt_loose_correct"],
        "prompt_total": r["if_prompt_total"],
        "inst_strict_correct": r["if_inst_strict_correct"],
        "inst_loose_correct": r["if_inst_loose_correct"],
        "inst_total": r["if_inst_total"],
    }


def _rates(counts: dict[str, float]) -> dict[str, float]:
    p, i = counts["prompt_total"], counts["inst_total"]
    out = {
        "prompt_strict": 100.0 * counts["prompt_strict_correct"] / p if p else 0.0,
        "prompt_loose": 100.0 * counts["prompt_loose_correct"] / p if p else 0.0,
        "inst_strict": 100.0 * counts["inst_strict_correct"] / i if i else 0.0,
        "inst_loose": 100.0 * counts["inst_loose_correct"] / i if i else 0.0,
    }
    out["overall"] = statistics.fmean(out[m] for m in SUB_METRICS)
    out["n"] = int(p)
    return out


def _accumulate(bucket: dict[str, float], counts: dict[str, float]) -> None:
    for k, v in counts.items():
        bucket[k] = bucket.get(k, 0) + v


def score_multiif(records: list[dict]) -> dict:
    """records: [{key, language, turns: [{prompt, instruction_id_list, kwargs, response}, x3]}]."""
    per_lang: dict[str, dict[int, dict]] = {}
    for rec in records:
        lang = rec["language"]
        for t, turn in zip(TURNS, rec["turns"]):
            counts = _score_response("MultiIF", turn, turn["response"])
            _accumulate(per_lang.setdefault(lang, {}).setdefault(t, {}), counts)

    languages = sorted(per_lang)
    by_lang = {
        lang: {f"turn_{t}": _rates(per_lang[lang][t]) for t in TURNS} for lang in languages
    }

    # Unweighted over languages -- see the module docstring.
    lang_avg = {}
    for t in TURNS:
        lang_avg[f"turn_{t}"] = {
            m: statistics.fmean(by_lang[lang][f"turn_{t}"][m] for lang in languages)
            for m in (*SUB_METRICS, "overall")
        }

    english = by_lang.get("English")
    return {
        "benchmark": "Multi-IF",
        "n_conversations": len(records),
        "languages": languages,
        "headline": {
            "8lang_3turn_mean": statistics.fmean(lang_avg[f"turn_{t}"]["overall"] for t in TURNS),
            "8lang_turn3": lang_avg["turn_3"]["overall"],
            "english_3turn_mean": (
                statistics.fmean(english[f"turn_{t}"]["overall"] for t in TURNS)
                if english
                else None
            ),
            "english_turn3": english["turn_3"]["overall"] if english else None,
        },
        "lang_avg": lang_avg,
        "by_language": by_lang,
    }


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_multiif(path: Path, limit: int | None) -> list[dict]:
    df = pd.read_parquet(path)
    if limit:
        # Head, not sample: a head of a language-sorted file would be one
        # language. This file is in corpus order, which interleaves them, and a
        # deterministic slice is what makes a smoke run reproducible.
        df = df.head(limit)
    return [
        {
            "key": r["key"],
            "language": r["language"],
            "turns": [
                {
                    "prompt": r[f"turn_{t}_prompt"],
                    "instruction_id_list": list(r[f"turn_{t}_instruction_id_list"]),
                    "kwargs": json.loads(r[f"turn_{t}_kwargs"]),
                }
                for t in TURNS
            ],
        }
        for _, r in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def _visible_devices() -> list[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [d for d in env.split(",") if d.strip()]
    import torch

    return [str(i) for i in range(torch.cuda.device_count())]


def _prompt_len(tokenizer, history: list[dict]) -> int:
    """Token count of a rendered chat prompt, the way vLLM will count it.

    Render to text, then encode. Not `apply_chat_template(tokenize=True)`:
    transformers v5 returns a *dict* from that (`input_ids`, `attention_mask`),
    so `len(...)` of it is 2 -- a length check built on it passes everything and
    silently does nothing. `add_special_tokens=False` because the template
    already emits the BOS itself; adding another would double it.
    """
    text = tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _split_by_budget(
    prompt_lens: list[int], max_model_len: int, reserve: int
) -> tuple[list[int], list[int]]:
    """Partition conversation indices into (fits, overflows) by prompt length.

    A three-turn conversation carries both earlier answers in its turn-3 prompt,
    and an answer that hit the 16384-token cap has no `</think>` to strip, so
    reasoning-stripping returns all 16k of it. Two capped turns is therefore
    ~32k of history against `max_model_len=32768`, and vLLM rejects the whole
    `chat()` call -- not just that one request -- when any prompt is too long.
    One conversation in a few hundred used to take its entire shard with it.

    `reserve` is headroom, not policy: it only covers drift between the template
    we render to measure and the one vLLM renders to run. Everything that can
    physically generate a token still does.
    """
    fits, overflows = [], []
    for i, n in enumerate(prompt_lens):
        (fits if n + reserve < max_model_len else overflows).append(i)
    return fits, overflows


def _generate_here(records: list[dict], args) -> None:
    """Fill `turn["response"]` in place, one batched vLLM pass per turn."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        seed=args.seed,
    )
    params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_out_len,
        seed=args.seed,
    )

    # One conversation per record, extended in place. All records advance
    # through the turns together so each turn is a single large batch -- turn by
    # turn is the only ordering vLLM can batch, since turn 2's prompt contains
    # turn 1's answer.
    histories: list[list[dict]] = [[] for _ in records]
    tokenizer = llm.get_tokenizer()
    n_turns = len(records[0]["turns"])
    for t in range(n_turns):
        for hist, rec in zip(histories, records):
            hist.append({"role": "user", "content": rec["turns"][t]["prompt"]})
        lens = [_prompt_len(tokenizer, h) for h in histories]
        fits, overflows = _split_by_budget(lens, args.max_model_len, _PROMPT_RESERVE)
        for i in overflows:
            # No response is the honest record: the model was handed a context
            # it cannot read, so it follows none of the turn's instructions and
            # scores zero. Truncating the history to make it fit would score a
            # different conversation than the one the benchmark asks for.
            records[i]["turns"][t]["response"] = ""
            records[i]["turns"][t]["finish_reason"] = "prompt_too_long"
            histories[i].append({"role": "assistant", "content": ""})
        outputs = llm.chat([histories[i] for i in fits], params, add_generation_prompt=True)
        for i, out in zip(fits, outputs):
            hist, rec = histories[i], records[i]
            raw = out.outputs[0].text
            rec["turns"][t]["response"] = raw
            rec["turns"][t]["finish_reason"] = out.outputs[0].finish_reason
            carried = (
                raw
                if args.history_keeps_reasoning
                else if_eval.extract_non_reasoning_content_internal(raw)
            )
            hist.append({"role": "assistant", "content": carried})
        capped = sum(1 for r in records if r["turns"][t]["finish_reason"] == "length")
        print(
            f"[turn {t + 1}] {len(fits)} generated, {capped} hit the token cap, "
            f"{len(overflows)} skipped as over-long",
            flush=True,
        )


def _shard_worker(args, records: list[dict], shard_path: str) -> None:
    """Run one shard, write its slice, and leave without waiting on vLLM.

    Two things about `multiprocessing` shape this. `_bootstrap` runs
    `util._exit_function` in a `finally`, so it fires whether the target
    returned or raised -- and that function joins the `EngineCore` subprocess
    vLLM spawned, which does not reliably exit. A crashed shard therefore hangs
    with its traceback still queued behind the join, which is how a corpus-wide
    ValueError looked, from the outside, like eight healthy idle GPUs.

    So: print our own traceback while we still can, terminate the children we
    started, and `_exit` past the atexit handlers. The parent reads the shard
    file and the exit code; nothing here needs a clean interpreter shutdown.
    """
    code = 0
    try:
        _generate_here(records, args)
        with open(shard_path, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    for child in multiprocessing.active_children():
        child.terminate()  # our own EngineCore, nobody else's
    os._exit(code)


def generate(records: list[dict], args) -> None:
    """Generate every response, sharding the conversations over the visible GPUs.

    Data parallel rather than tensor parallel, because the models this recipe
    evaluates are 1.5B: they fit on one card with room for a 32k KV cache, so
    splitting them across cards buys nothing and costs a collective per layer.
    The work is ~13k generations of up to 16k tokens, which is throughput-bound,
    and eight independent engines is the shape that matches.

    It is also the shape the model *allows*. Qwen2-1.5B has 12 attention heads;
    vLLM requires the head count to divide by the TP size, so TP=8 is rejected
    outright. `--tp` stays available for larger models and for 2/3/4/6, and the
    shard count is whatever is left over.
    """
    # vLLM launches its own EngineCore in a subprocess, using whichever start
    # method this variable names -- fork by default. Under fork it inherits a
    # CUDA context that the importing process has already created (importing
    # `verl` pulls in torch and probes the devices) and dies with "Cannot
    # re-initialize CUDA in forked subprocess". Spawn costs one extra
    # interpreter start per shard, once, against runs measured in hours.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    devices = _visible_devices()
    n_shards = max(1, len(devices) // args.tp)
    if n_shards == 1:
        _generate_here(records, args)
        return

    ctx = multiprocessing.get_context("spawn")
    tmp = Path(args.out) / "shards"
    tmp.mkdir(parents=True, exist_ok=True)
    procs, paths, slices = [], [], []
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        for i in range(n_shards):
            # Strided, not contiguous: response length correlates with the
            # question, and a contiguous slice of a corpus that is grouped by
            # language would leave one shard generating long-form Hindi while
            # the rest finished. Every shard then waits for that one.
            part = records[i::n_shards]
            if not part:
                continue
            path = tmp / f"shard{i}.jsonl"
            # Set in the parent so the child inherits it at spawn, i.e. before
            # it imports torch. Setting it inside the child races with whatever
            # CUDA has already cached by then.
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                devices[i * args.tp : (i + 1) * args.tp]
            )
            p = ctx.Process(target=_shard_worker, args=(args, part, str(path)))
            p.start()
            procs.append(p)
            paths.append(path)
            slices.append(part)
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved

    failed = []
    for i, (p, path) in enumerate(zip(procs, paths)):
        p.join()
        if p.exitcode != 0 or not path.exists():
            failed.append(f"shard {i} (exit {p.exitcode})")
    if failed:
        raise SystemExit(
            "generation failed in " + ", ".join(failed) + ". "
            "The shards that did finish are under "
            f"{tmp}; fix the cause and rerun, the finished ones are cheap to keep."
        )

    # The children mutated copies, so read the responses back onto the originals.
    for part, path in zip(slices, paths):
        for rec, line in zip(part, path.open()):
            done = json.loads(line)
            for turn, done_turn in zip(rec["turns"], done["turns"]):
                turn["response"] = done_turn["response"]
                turn["finish_reason"] = done_turn["finish_reason"]


# ---------------------------------------------------------------------------


def _write(out_dir: Path, records: list[dict], report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "generations.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


def _print_report(report: dict) -> None:
    print()
    print(f"Multi-IF  n={report['n_conversations']}  langs={len(report['languages'])}")
    print(f"  {'turn':<8} " + "".join(f"{m:>14}" for m in (*SUB_METRICS, "overall")))
    for t in TURNS:
        row = report["lang_avg"][f"turn_{t}"]
        print(f"  {t:<8} " + "".join(f"{row[m]:14.2f}" for m in (*SUB_METRICS, "overall")))
    h = report["headline"]
    print()
    print(f"  8-lang 3-turn mean   {h['8lang_3turn_mean']:6.2f}   (criterion >= 37.5)")
    print(f"  8-lang turn 3        {h['8lang_turn3']:6.2f}   (criterion >= 26.5)")
    if h["english_3turn_mean"] is not None:
        print(f"  English 3-turn mean  {h['english_3turn_mean']:6.2f}   (criterion >= 44.0)")
        print(f"  English turn 3       {h['english_turn3']:6.2f}   (criterion >= 35.0)")
    print()
    print(f"  {'language':<12}" + "".join(f"{f'turn{t}':>10}" for t in TURNS))
    for lang in report["languages"]:
        cells = "".join(f"{report['by_language'][lang][f'turn_{t}']['overall']:10.2f}" for t in TURNS)
        print(f"  {lang:<12}{cells}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("benchmark", choices=["multiif", "rescore"])
    p.add_argument("--model", help="HF directory (merge FSDP shards first: verl.model_merger)")
    p.add_argument("--data", type=Path, help="override the default parquet")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--limit", type=int, default=0, help="first N rows only, for smoke runs")
    p.add_argument(
        "--tp",
        type=int,
        default=1,
        help="tensor parallel size per engine; the visible GPUs left over become "
        "data-parallel shards (default: one engine per GPU)",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-out-len", type=int, default=16384, help="per turn; the reference budget")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--history-keeps-reasoning",
        action="store_true",
        help="carry <think> blocks into later turns (default: carry the stripped answer)",
    )
    args = p.parse_args(argv)
    if args.benchmark != "rescore" and not args.model:
        p.error("--model is required unless benchmark is 'rescore'")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.benchmark == "rescore":
        # Scoring is minutes and generation is hours, so a checker fix or an
        # aggregation change must not cost another GPU pass.
        records = [json.loads(line) for line in (args.out / "generations.jsonl").open()]
    else:
        path = args.data or DATA_ROOT / "if" / "MultiIF" / "eval_3turn.parquet"
        if not path.exists():
            raise SystemExit(
                f"{path} not found.\nBuild it: python3 data/prep/prepare_multiif_eval.py"
            )
        records = load_multiif(path, args.limit or None)
        generate(records, args)

    report = score_multiif(records)
    report["model"] = args.model
    _write(args.out, records, report)
    _print_report(report)
    print(f"\nwrote {args.out}/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
