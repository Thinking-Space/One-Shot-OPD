from verl.utils.reward_score.ttrl_math import compute_score as boxed_score
from verl.utils.reward_score import math_verify

def reward_func(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    strict = boxed_score(solution_str, str(ground_truth))
    if strict["format_score"] == 1.0:
        return strict

    tail = solution_str[-2000:]
    ok = float(math_verify.compute_score(tail, str(ground_truth)))
    return {
        "score": ok,
        "format_score": 0.0,
        "acc": bool(ok),
        "extracted_gt": ground_truth,
        "pred": "",
    }