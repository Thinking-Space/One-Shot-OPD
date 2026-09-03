"""端到端冒烟测试: 3 个 bench 各跑 1 题 (1 个正确解 + 1 个错误解).

运行:
    cd <repo>/verl && python -m verl.utils.reward_score.code_eval_v2._smoke_test

不依赖 HF datasets, 直接硬编码典型 case, 验证 dispatcher + runner 通路.
"""

from __future__ import annotations

import json

from . import compute_score


# ----------------------------------------------------------------------------
# Case 1: HumanEval+ - has_close_elements (HumanEval 第 0 题)
# ----------------------------------------------------------------------------
HE_TEST_CODE = '''
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False

inputs = [
    ([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3),
    ([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05),
    ([1.0, 2.0, 5.9, 4.0, 5.0], 0.95),
    ([1.0, 2.0, 5.9, 4.0, 5.0], 0.8),
]
results = [True, False, True, False]

check(has_close_elements)
'''

HE_RIGHT = """```python
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i, a in enumerate(numbers):
        for j, b in enumerate(numbers):
            if i != j and abs(a - b) < threshold:
                return True
    return False
```"""

HE_WRONG = """```python
def has_close_elements(numbers, threshold):
    return False
```"""


# ----------------------------------------------------------------------------
# Case 2: MBPP+ - min_cost (MBPP-3, similar EvalPlus 风格)
# 用一个简单的 add 题目, 因为 MBPP+ 的 test 格式跟 HE+ 完全一样 (check + asserts)
# ----------------------------------------------------------------------------
MBPP_TEST_CODE = '''
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(-1, 1) == 0
    assert candidate(100, 200) == 300

inputs = [(1, 2), (-1, 1), (100, 200)]
results = [3, 0, 300]

check(add)
'''

MBPP_RIGHT = """```python
def add(a, b):
    return a + b
```"""

MBPP_WRONG = """```python
def add(a, b):
    return a - b
```"""


# ----------------------------------------------------------------------------
# Case 3: LCB - stdin 模式 (a+b)
# ----------------------------------------------------------------------------
LCB_STDIN_TESTS = [
    {"input": "1 2\n", "output": "3\n", "testtype": "stdin", "metadata": {}},
    {"input": "10 20\n", "output": "30\n", "testtype": "stdin", "metadata": {}},
]

LCB_STDIN_RIGHT = """```python
a, b = map(int, input().split())
print(a + b)
```"""

LCB_STDIN_WRONG = """```python
a, b = map(int, input().split())
print(a - b)
```"""


# ----------------------------------------------------------------------------
# Case 4: LCB - functional 模式 (twoSum 风格)
# ----------------------------------------------------------------------------
LCB_FN_TESTS = [
    {
        "input": "[2, 7, 11, 15]\n9",
        "output": "[0, 1]",
        "testtype": "functional",
        "metadata": {"func_name": "twoSum"},
    },
]

LCB_FN_RIGHT = """```python
class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, n in enumerate(nums):
            if target - n in seen:
                return [seen[target - n], i]
            seen[n] = i
```"""


# ----------------------------------------------------------------------------
# 主测试逻辑
# ----------------------------------------------------------------------------
def _run(name: str, data_source: str, completion: str, ground_truth, expected: bool):
    """单 case 验证, 失败 raise AssertionError."""
    res = compute_score(data_source, completion, ground_truth)
    actual = res["acc"]
    status = "OK" if actual == expected else "FAIL"
    print(
        f"[{status}] {name:40s} data_source={data_source:20s} "
        f"expected={expected}  actual={actual}  passed_tests={res['metadata'].get('passed_tests','?')}/"
        f"{res['metadata'].get('total_tests','?')}"
    )
    if actual != expected:
        print(f"      full result: {json.dumps(res, default=str)[:500]}")
        raise AssertionError(f"{name} expected acc={expected} got {actual}")


def main():
    print("=" * 70)
    print("code_eval smoke test")
    print("=" * 70)

    # HumanEval+
    _run(
        "HE+ correct",
        "humanevalplus",
        HE_RIGHT,
        json.dumps({"test": HE_TEST_CODE, "entry_point": "has_close_elements"}),
        expected=True,
    )
    _run(
        "HE+ wrong",
        "humanevalplus",
        HE_WRONG,
        json.dumps({"test": HE_TEST_CODE, "entry_point": "has_close_elements"}),
        expected=False,
    )

    # MBPP+
    _run(
        "MBPP+ correct",
        "mbppplus",
        MBPP_RIGHT,
        json.dumps({"test": MBPP_TEST_CODE, "entry_point": "add"}),
        expected=True,
    )
    _run(
        "MBPP+ wrong",
        "mbppplus",
        MBPP_WRONG,
        json.dumps({"test": MBPP_TEST_CODE, "entry_point": "add"}),
        expected=False,
    )

    # LCB stdin
    _run(
        "LCB stdin correct",
        "livecodebench",
        LCB_STDIN_RIGHT,
        json.dumps(LCB_STDIN_TESTS),
        expected=True,
    )
    _run(
        "LCB stdin wrong",
        "livecodebench",
        LCB_STDIN_WRONG,
        json.dumps(LCB_STDIN_TESTS),
        expected=False,
    )

    # LCB functional
    _run(
        "LCB functional correct",
        "livecodebench",
        LCB_FN_RIGHT,
        json.dumps(LCB_FN_TESTS),
        expected=True,
    )

    print("=" * 70)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
