"""Mixed validation reward for OPD's four domains: math, code, IF and FC."""

from __future__ import annotations

from verl.utils.reward_score import if_eval
from verl.utils.reward_score import ttrl_mathverify
from verl.utils.reward_score.code_eval_v2 import _LCB_SOURCES as _CODE_LCB_SOURCES
from verl.utils.reward_score.code_eval_v2 import _HUMANEVAL_SOURCES as _CODE_HE_SOURCES
from verl.utils.reward_score.code_eval_v2 import reward_func as _code_reward_func
from verl.utils.reward_score.fc_eval import _BFCL_SOURCES, _XLAM_SOURCES
from verl.utils.reward_score.fc_eval import reward_func as _fc_reward_func


IF_DATA_SOURCES = {"MultiIF"}
CODE_DATA_SOURCES = _CODE_LCB_SOURCES | _CODE_HE_SOURCES
FC_DATA_SOURCES = _XLAM_SOURCES | _BFCL_SOURCES

# Every data_source we have announced a routing decision for. The math branch is
# the fallback, so a benchmark whose data_source we do not recognise is scored
# by the math verifier and quietly gets ~0 instead of erroring -- which reads
# exactly like "the model cannot do this task". Announcing the decision once per
# data_source turns that into one greppable line:
#
#     grep '\[opd_verify\]' <training log>
#
# Anything reported as '-> math (fallback)' that is not a math benchmark is a
# misroute, not a bad score.
_ANNOUNCED: set[str] = set()


def _announce(data_source, branch):
    if data_source in _ANNOUNCED:
        return
    _ANNOUNCED.add(data_source)
    print(f"[opd_verify] data_source={data_source!r} -> {branch}", flush=True)


def reward_func(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source in IF_DATA_SOURCES:
        _announce(data_source, "if_eval")
        return if_eval.reward_func(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )

    if data_source in CODE_DATA_SOURCES:
        _announce(data_source, "code_eval_v2")
        return _code_reward_func(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )

    if data_source in FC_DATA_SOURCES:
        _announce(data_source, "fc_eval")
        return _fc_reward_func(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )

    _announce(data_source, "math (fallback)")
    return ttrl_mathverify.reward_func(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
