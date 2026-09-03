"""Rule-based Multi-IF scoring.

One family, because IF has one benchmark. Multi-IF is not a verl validation
reward function -- it is a three-turn benchmark, so it cannot go through a
validation loop that generates a single response per prompt. What it needs from
here is the checker registry: `eval/run_if_eval.py` drives the turns and calls
`score_single_turn` once per turn. `reward_func` is the verl door, used to grade
the IF *training* rows under `grpo.sh` (verifier reward, no teacher).

There used to be two more families, IFEval and IFBench. IFBench went out with
its benchmark, and IFEval with its own.

Dropping IFEval does not change any score, and the reason is worth writing down
because the code used to claim otherwise: the two families were kept apart on
the premise that Multi-IF's checkers are multilingual and IFEval's are
English-only. In the opencompass tree this loads from, that premise is false.
The two directories are the same code -- `instructions_util.py` is byte
identical, the registries expose the same 25 ids, and both `_LANGUAGES` tables
hold the same 30 codes. They differ in import style and in some `absl` flag
boilerplate IFEval carries and Multi-IF does not. So the family name here has
only ever selected between two copies of one implementation, and the surviving
copy is the one named after the benchmark IF is actually judged on.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import threading
import types
from pathlib import Path
from typing import Any


# Multi-IF scoring is OpenCompass's instruction checkers, loaded
# by file path -- there is no pip package that exposes them. The tree therefore
# has to exist somewhere, and where is a property of the machine, not of this
# repo: set OPENCOMPASS_ROOT to point at it. Deliberately no default. The
# default used to be one particular cluster's path, which meant a machine
# without that directory got "OpenCompass root not found: /home/test/..." and
# had to work out that the path was never meant to be theirs.
OPENCOMPASS_ROOT = Path(os.environ.get("OPENCOMPASS_ROOT", ""))

#: family -> the four files to load, in dependency order. `instructions_util`
#: first: `instructions` imports it relatively, and a relative import can only
#: resolve against a module already in `sys.modules`.
_FAMILIES = {
    "MultiIF": "evaluation_main",
}

#: data_source -> (family, module holding InputExample and the two testers).
_SOURCE_TO_FAMILY = {
    "MultiIF": "MultiIF",
}

_MODULES_LOADED = False
_MODULES_LOCK = threading.Lock()


def _ensure_opencompass_modules():
    """Load the Multi-IF evaluation modules without triggering opencompass __init__.

    Serialised, and the flag is only set once the modules are actually in
    ``sys.modules``. verl 0.9.0 grades through ``loop.run_in_executor``, so
    several rows land here on different pool threads at once: setting the flag
    on entry let the second thread skip the load and go straight to
    ``sys.modules[...]``, which the first thread had not filled in yet.
    """
    global _MODULES_LOADED
    if _MODULES_LOADED:
        return

    with _MODULES_LOCK:
        if _MODULES_LOADED:
            return
        _load_opencompass_modules()
        _MODULES_LOADED = True


def _load_opencompass_modules():
    oc_root = OPENCOMPASS_ROOT
    if not str(oc_root) or not oc_root.exists():
        raise RuntimeError(
            f"OpenCompass root not found: {oc_root or '<OPENCOMPASS_ROOT unset>'}\n"
            "Multi-IF scoring needs OpenCompass's instruction checkers, "
            "which are loaded from a source tree rather than a package. Clone opencompass "
            "and set OPENCOMPASS_ROOT to it. Its checkers also import nltk and need "
            "the punkt/punkt_tab tokenizer data."
        )

    for family, entry in _FAMILIES.items():
        pkg = f"opencompass.datasets.{family}"
        _ensure_namespace_packages(["opencompass", "opencompass.datasets", pkg])
        family_dir = oc_root / "opencompass" / "datasets" / family
        if not family_dir.is_dir():
            raise RuntimeError(
                f"{family_dir} is missing. OPENCOMPASS_ROOT={oc_root} points at a tree "
                f"without the {family} checkers; Multi-IF in particular is not in every "
                "opencompass release."
            )
        # instructions_util -> instructions -> instructions_registry -> entry.
        # Any other order breaks on the relative imports inside them.
        for name in ("instructions_util", "instructions", "instructions_registry", entry):
            _load_file(f"{pkg}.{name}", family_dir / f"{name}.py")


def _ensure_namespace_packages(names):
    """Stand in for the real packages so relative imports inside the checkers resolve.

    Importing `opencompass.datasets` for real runs opencompass's ``__init__``,
    which pulls in its registry, its config system and a large slice of its
    dependency tree. The checkers need none of that -- they need a parent
    package to exist so ``from . import instructions_util`` has something to
    resolve against.
    """
    for pkg in names:
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__package__ = pkg
            mod.__path__ = []
            sys.modules[pkg] = mod


def _load_file(module_name: str, file_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec, not after: `instructions` imports
    # `instructions_util` relatively while it is still executing, and an
    # unregistered parent-package member is not importable.
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def extract_non_reasoning_content_internal(
    text: str,
    think_start_token: str = "<think>",
    think_end_token: str = "</think>",
    overflow_empty: bool = True,
) -> str:
    """Match OpenCompass mb_internal reasoning stripping behavior."""
    if not isinstance(text, str):
        return ""
    has_start = think_start_token in text
    has_end = think_end_token in text

    if has_start and has_end:
        reasoning_regex = re.compile(
            rf"{re.escape(think_start_token)}(.*?){re.escape(think_end_token)}",
            re.DOTALL,
        )
        result = reasoning_regex.sub("", text).strip()
    elif not has_start and has_end:
        result = text.split(think_end_token)[-1].strip()
    elif has_start and not has_end:
        _, _, after = text.partition(think_start_token)
        result = after.strip()
    else:
        result = text.strip()

    if overflow_empty and len(result) > 80_000:
        return ""
    return result


def _clean_kwargs(reference: dict[str, Any]) -> None:
    for kwarg in reference["kwargs"]:
        for key in list(kwarg.keys()):
            if kwarg[key] is None:
                kwarg.pop(key, None)


def _score_one(data_source: str, prediction: str, reference: dict[str, Any]) -> dict[str, Any]:
    _ensure_opencompass_modules()

    family = _SOURCE_TO_FAMILY.get(data_source)
    if family is None:
        raise NotImplementedError(
            f"Unsupported instruction-following source: {data_source!r} "
            f"(known: {sorted(_SOURCE_TO_FAMILY)})"
        )
    mod = sys.modules[f"opencompass.datasets.{family}.{_FAMILIES[family]}"]

    InputExample = mod.InputExample
    test_instruction_following_loose = mod.test_instruction_following_loose
    test_instruction_following_strict = mod.test_instruction_following_strict

    reference = dict(reference)
    reference["kwargs"] = [dict(item) for item in reference["kwargs"]]
    _clean_kwargs(reference)

    example_input = InputExample(
        key=reference["key"],
        instruction_id_list=reference["instruction_id_list"],
        prompt=reference["prompt"],
        kwargs=reference["kwargs"],
    )

    strict_result = test_instruction_following_strict(example_input, prediction)
    loose_result = test_instruction_following_loose(example_input, prediction)

    strict_list = strict_result.follow_instruction_list
    loose_list = loose_result.follow_instruction_list
    inst_total = len(strict_result.instruction_id_list)
    prompt_strict_correct = int(all(strict_list))
    prompt_loose_correct = int(all(loose_list))
    inst_strict_correct = int(sum(strict_list))
    inst_loose_correct = int(sum(loose_list))

    if prompt_strict_correct:
        grade = "strict"
    elif prompt_loose_correct:
        grade = "loose"
    else:
        grade = "none"

    return {
        "score": float(prompt_strict_correct),
        "acc": bool(prompt_strict_correct),
        "prompt_strict_acc": float(prompt_strict_correct),
        "inst_strict_acc": inst_strict_correct / inst_total if inst_total else 0.0,
        "prompt_loose_acc": float(prompt_loose_correct),
        "inst_loose_acc": inst_loose_correct / inst_total if inst_total else 0.0,
        "if_prompt_strict_correct": prompt_strict_correct,
        "if_prompt_loose_correct": prompt_loose_correct,
        "if_prompt_total": 1,
        "if_inst_strict_correct": inst_strict_correct,
        "if_inst_loose_correct": inst_loose_correct,
        "if_inst_total": inst_total,
        "pred": prediction,
        "grade": grade,
    }


def score_single_turn(
    family: str,
    prompt: str,
    instruction_id_list: list[str],
    kwargs: list[dict[str, Any]],
    response: str,
    strip_reasoning: bool = True,
) -> dict[str, Any]:
    """Score one response against one turn's instruction list.

    The offline entry point. `reward_func` below is the verl one and differs in
    two ways that matter: it takes verl's `(data_source, solution_str,
    ground_truth)` triple, and it is called once per row. A multi-turn benchmark
    has to build its own conversation before it has anything to score, so it
    calls this per turn and aggregates itself.

    `strip_reasoning` is on for the same reason it is in `reward_func`: a
    distilled R1 student answers inside `<think>...</think>` and the checkers
    would grade the reasoning. Off is for scoring text that has already been
    stripped, so a caller cannot strip twice and silently empty a response whose
    answer happens to contain the token.
    """
    if strip_reasoning:
        response = extract_non_reasoning_content_internal(response)
    return _score_one(
        family,
        response,
        {"key": 0, "prompt": prompt, "instruction_id_list": instruction_id_list, "kwargs": kwargs},
    )


def reward_func(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    import json as _json
    if isinstance(ground_truth, str):
        ground_truth = _json.loads(ground_truth)
    prediction = extract_non_reasoning_content_internal(solution_str)
    return _score_one(data_source, prediction, ground_truth)
