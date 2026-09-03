# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ttrl_math_v2: 与 ttrl_math 相同，但代码评测路由到 code_eval_v2。"""

from latex2sympy2_extended import latex2sympy
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
import traceback

from .math_utils import extract_boxed_answer, is_latex_equal, grade_answer_mathd, grade_answer_sympy, timeout_ours


def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def grade(model_answer: str, gt_answer: str, fast: bool = True):
    if "\\boxed" in gt_answer:
        gt_answer = extract_answer(gt_answer)
    correct = grade_answer_mathd(model_answer, gt_answer) or grade_answer_sympy(model_answer, gt_answer)
    if not fast:
        correct = correct or is_latex_equal(model_answer, gt_answer)
    return correct

@timeout_ours(timeout_seconds=10)
def simplify_expression_string(expression_string: str) -> str:
    try:
        sympy_expr = parse_expr(expression_string, transformations="all", evaluate=False)
        simplified_expr = simplify(sympy_expr)
        return str(simplified_expr)
    except TimeoutError:
        return expression_string
    except Exception as e:
        try:
            sympy_expr = latex2sympy(expression_string)
            simplified_expr = simplify(sympy_expr)
            return str(simplified_expr)
        except TimeoutError:
            return expression_string
        except Exception as e:
            return expression_string

def compute_score(model_response, gt_answer, fast=False):
    model_answer = extract_answer(model_response)
    if model_answer is None:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": "",
        }
    is_correct = False
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        is_correct = grade(model_answer, gt_answer, fast)
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            is_correct |= grade(model_answer, gt, fast)
    if is_correct:
        return {"score": 1.0, "format_score": 1.0, "acc": True, "extracted_gt": gt_answer, "pred": model_answer}
    else:
        return {"score": 0.0, "format_score": 1.0, "acc": False, "extracted_gt": gt_answer, "pred": model_answer}

# ----------------------------------------------------------------------------
# 代码评测路由: 指向 code_eval_v2
# ----------------------------------------------------------------------------
_CODE_DATA_SOURCES = {
    "livecodebench", "livecodebench/code_generation", "livecodebench/code_generation_lite",
    "humanevalplus", "evalplus/humanevalplus", "openai/openai_humaneval", "openai_humaneval",
    "mbppplus", "mbpp+", "evalplus/mbppplus", "mbpp", "google-research-datasets/mbpp",
    "taco", "BAAI/TACO", "apps", "codeparrot/apps",
    "code_contests", "deepmind/code_contests", "codeforces",
}

_FC_DATA_SOURCES = {
    "xlam", "bfcl_simple", "bfcl_multiple",
    "bfcl_parallel", "bfcl_parallel_multiple", "bfcl_live_simple",
}


def reward_func(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    """统一 reward 入口 (v2): 代码评测走 code_eval_v2。"""
    try:
        if data_source in _CODE_DATA_SOURCES:
            from verl.utils.reward_score.code_eval_v2 import compute_score as code_compute_score

            return code_compute_score(data_source, solution_str, ground_truth, extra_info)

        if data_source in _FC_DATA_SOURCES:
            from verl.utils.reward_score.fc_eval import compute_score as fc_compute_score

            return fc_compute_score(data_source, solution_str, ground_truth, extra_info)

        res = compute_score(solution_str, str(ground_truth))

        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])
    except Exception as e:
        print(f"[ERROR] Error in process_completion for task data_source={data_source}: {str(e)}")
        traceback.print_exc()
        raise
