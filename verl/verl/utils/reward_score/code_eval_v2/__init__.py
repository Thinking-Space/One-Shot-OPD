"""代码评测 v2 — 对齐 opencompass 配置

HumanEval+: 使用 evalplus 官方库 (sanitize + evaluate)，超时和并行由 evalplus 管理。
LCB:        使用 ProcessPoolExecutor 并行评测多道题，每道题 multiprocessing.Process 隔离。

verl 通过 val.n 控制采样数，本模块只负责对 *单个 completion* 做 pass/fail 判定。

支持的 data_source:
  - "humanevalplus" / "openai_humaneval" : HumanEval+ (evalplus 官方)
  - "livecodebench" 等                   : LCB v6 (ProcessPoolExecutor 并行)
  - "taco" / "apps" / "code_contests"    : 走 LCB runner
"""

from __future__ import annotations

import json
import multiprocessing
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Optional

from ._lcb_runner import run_test as lcb_run_test


# ============================================================================
# 代码抠取
# ============================================================================
def extract_code_from_model(model_response: str) -> Optional[str]:
    """从模型 markdown 输出中抠出最后一个代码块."""
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


def clean_code_main_block(code: str) -> str:
    """剥掉 if __name__ == '__main__' 块."""
    code_lines = code.split("\n")
    filtered_lines = []
    skip_block = False
    for line in code_lines:
        if line.strip().startswith('if __name__ == "__main__"') or line.strip().startswith(
            "if __name__ == '__main__'"
        ):
            skip_block = True
            continue
        if skip_block:
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                skip_block = False
            else:
                continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)


# ============================================================================
# HumanEval+ 评测 (evalplus 官方库)
# ============================================================================
def _prune_to_code(text: str, max_chars: int = 20000) -> str:
    """从模型输出提取代码块，加速 sanitize（对齐 opencompass 逻辑）."""
    if not isinstance(text, str):
        return text
    if '```' in text:
        blocks = re.findall(r'```[Pp]ython\s*\n?(.*?)```', text, re.DOTALL)
        if not blocks:
            blocks = re.findall(r'```\w*\s*\n?(.*?)```', text, re.DOTALL)
        if blocks:
            text = blocks[0].strip()
    if '```' not in text:
        m = re.search(r'(^|\n)(import\s|from\s|def\s|class\s)', text)
        if m:
            text = text[m.start(2):]
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def heplus_check_correctness(
    test_code: str, code: str, entry_point: str | None = None,
    timeout_per_test: int = 1,
) -> tuple[bool, dict]:
    """HumanEval+ 单题评测（subprocess 模式，v2 修复版）.

    Args:
        test_code: evalplus test 代码（含 check 函数 + 调用）.
        code: 模型生成的代码.
        entry_point: 函数入口名（暂未使用）.
        timeout_per_test: 单 case 时间预算（用于计算总 timeout，但硬上限 60s）.

    Returns:
        (passed, detail_dict)
    """
    from ._heplus_runner import run_test as heplus_run_test, _MAX_TIMEOUT

    cleaned_code = clean_code_main_block(code)
    # timeout 直接用硬上限 60s
    succ, output = heplus_run_test(cleaned_code, test_code, timeout=_MAX_TIMEOUT)
    return succ, {"output": output[:500] if output else ""}


def heplus_check_correctness_batch(
    completions: list[str],
    dataset_path: str | None = None,
    parallel: int | None = None,
    sanitize_parallel: int = 8,
) -> list[bool]:
    """批量评测 HumanEval+ — 对齐 opencompass 的 HumanEvalPlusEvaluator.

    一次传入所有 completion（已按 task_id 顺序排列），用 evalplus 批量跑。
    verl 的 val.n 采样在外层处理，这里每个 task_id 对应一个 completion。

    Args:
        completions: list[str]，每个元素是一道 HumanEval 题对应的一个 completion（模型原始输出）。
                     长度应为 164（HumanEval 题数）或其整数倍（n 次采样展平）。
        dataset_path: HumanEvalPlus.jsonl.gz 路径，None 则用 evalplus 默认。
        parallel: evalplus evaluate 的并行数，None 用默认。
        sanitize_parallel: sanitize 并行线程数。

    Returns:
        list[bool]，每个 completion 是否通过。
    """
    import os
    import tempfile
    from concurrent.futures import ThreadPoolExecutor, as_completed as t_as_completed

    try:
        from evalplus.data import get_human_eval_plus, write_jsonl
        from evalplus.eval import PASS, estimate_pass_at_k
        from evalplus.evaluate import evaluate
        from evalplus.sanitize import sanitize
    except ImportError:
        raise ImportError("请安装 evalplus: pip install evalplus")

    if dataset_path:
        os.environ['HUMANEVAL_OVERRIDE_PATH'] = dataset_path

    dataset_metadata = get_human_eval_plus(mini=False, noextreme=False, version='default')
    task_ids = sorted(dataset_metadata.keys())  # HumanEval/0 .. HumanEval/163
    n_tasks = len(task_ids)

    # 推断每道题有多少个 sample
    assert len(completions) % n_tasks == 0, (
        f"completions 长度 {len(completions)} 不是题数 {n_tasks} 的整数倍"
    )
    n_samples = len(completions) // n_tasks

    # sanitize 所有 completions
    entry_points = {tid: dataset_metadata[tid]['entry_point'] for tid in task_ids}

    def _sanitize_one(idx: int):
        sample_idx = idx // n_samples
        tid = task_ids[sample_idx]
        ep = entry_points[tid]
        pruned = _prune_to_code(completions[idx])
        return {"task_id": tid, "solution": sanitize(pruned, entrypoint=ep)}

    with ThreadPoolExecutor(max_workers=sanitize_parallel) as ex:
        futures = [ex.submit(_sanitize_one, i) for i in range(len(completions))]
        sanitized_samples = [None] * len(completions)
        for fut in t_as_completed(futures):
            res = fut.result()
            # 需要保留顺序，用 index
            pass
    # 顺序版（保证顺序正确）
    sanitized_samples = [_sanitize_one(i) for i in range(len(completions))]

    with tempfile.TemporaryDirectory() as tmp_dir:
        samples_path = os.path.join(tmp_dir, "samples.jsonl")
        write_jsonl(samples_path, sanitized_samples)

        evaluate(
            dataset="humaneval",
            samples=samples_path,
            base_only=False,
            parallel=parallel,
            i_just_wanna_run=True,
            test_details=False,
            mini=False,
            noextreme=False,
            version="default",
        )

        results_path = samples_path.replace(".jsonl", "_eval_results.json")
        if not os.path.exists(results_path):
            results_path = samples_path.replace(".jsonl", ".eval_results.json")
        with open(results_path, "r") as f:
            results = json.load(f)

    # 解析结果
    eval_data = results.get("eval", {})
    pass_results = []
    for i in range(len(completions)):
        sample_idx = i // n_samples
        within_idx = i % n_samples
        tid = task_ids[sample_idx]
        task_results = eval_data.get(tid, [])
        if within_idx < len(task_results):
            r = task_results[within_idx]
            passed = (r.get("base_status", "") == PASS and
                      r.get("plus_status", "") == PASS)
        else:
            passed = False
        pass_results.append(passed)

    return pass_results


# ============================================================================
# LCB 评测 (ProcessPoolExecutor 并行，对齐 opencompass)
# ============================================================================
def _postprocess_lcb_sample(sample: list[dict]) -> dict:
    """把 list[dict] 转成 lcb_run_test 期望的格式."""
    sample_inputs = [s["input"] for s in sample]
    sample_outputs = [s["output"] for s in sample]
    sample_dict: dict[str, Any] = {"inputs": sample_inputs, "outputs": sample_outputs}
    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name", None)
        assert fn_name is not None
        sample_dict["fn_name"] = fn_name
    return {"input_output": json.dumps(sample_dict)}


def _lcb_eval_single(args: tuple) -> tuple[bool, dict]:
    """单道 LCB 题的评测（在子进程中执行）."""
    processed, generation, timeout = args

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()

    def _worker(sample, gen, debug, res, meta, to):
        r, m = lcb_run_test(sample, test=gen, debug=debug, timeout=to)
        res.append(r)
        meta.append(m)

    p = multiprocessing.Process(
        target=_worker,
        args=(processed, generation, False, result, metadata_list, timeout),
    )
    p.start()
    n_cases = len(json.loads(processed["input_output"])["inputs"])
    p.join(timeout=(timeout + 1) * n_cases + 5)

    if p.is_alive():
        p.kill()
        p.join()

    if not result:
        return False, {"error": "global timeout"}

    all_passed = all(r is True for r in result[0])
    return all_passed, {"results": [r is True for r in result[0]]}


def lcb_check_correctness(
    sample: list[dict], generation: str, timeout: int = 6
) -> tuple[bool, dict]:
    """LCB 单题评测（兼容旧接口）."""
    processed = _postprocess_lcb_sample(sample)
    return _lcb_eval_single((processed, generation, timeout))


def lcb_check_correctness_batch(
    samples: list[list[dict]],
    generations: list[str],
    timeout: int = 6,
    num_workers: int = 4,
) -> list[bool]:
    """批量并行评测 LCB（对齐 opencompass ProcessPoolExecutor 模式）.

    Args:
        samples: list of test case lists, 每个元素对应一道题的 test cases.
        generations: list of code strings, 与 samples 一一对应.
        timeout: 单 test case 超时秒数.
        num_workers: 并行 worker 数.

    Returns:
        list[bool], 每道题是否全部通过.
    """
    processed_list = [_postprocess_lcb_sample(s) for s in samples]
    args_list = [(p, g, timeout) for p, g in zip(processed_list, generations)]

    results = [False] * len(args_list)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(_lcb_eval_single, args): idx
            for idx, args in enumerate(args_list)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                passed, _ = future.result()
                results[idx] = passed
            except Exception:
                results[idx] = False

    return results


# ============================================================================
# 统一入口 (单个 completion 评测，供 verl reward_func 调用)
# ============================================================================
_LCB_SOURCES = {
    "livecodebench", "livecodebench/code_generation",
    "livecodebench/code_generation_lite",
    "taco", "BAAI/TACO", "apps", "codeparrot/apps",
    "code_contests", "deepmind/code_contests", "codeforces",
}
_HUMANEVAL_SOURCES = {
    "humanevalplus", "evalplus/humanevalplus",
    "openai/openai_humaneval", "openai_humaneval",
}


def compute_score(
    data_source: str,
    completion: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
) -> dict:
    """代码评测统一入口（单 completion）.

    Args:
        data_source: 数据源标识.
        completion: 模型完整输出.
        ground_truth: 测试用例.
        extra_info: 保留.

    Returns:
        {"score": 0/1, "acc": bool, "format_score": float}
    """
    model_code = extract_code_from_model(completion)
    if model_code is None:
        return {"score": 0.0, "acc": False, "format_score": 0.0}

    gt = _load_ground_truth(ground_truth)

    try:
        if data_source in _LCB_SOURCES:
            tests = gt
            if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
                tests = _lcb_dict_to_list(tests)
            if not isinstance(tests, list) or len(tests) == 0:
                return _err_result(data_source, "LCB ground_truth 格式错误")
            passed, _ = lcb_check_correctness(tests, model_code, timeout=6)
        elif data_source in _HUMANEVAL_SOURCES:
            # 单题模式：直接用 subprocess + evalplus 做简单评测
            # 批量模式应使用 heplus_check_correctness_batch
            test_code = gt.get("test") if isinstance(gt, dict) else gt
            entry_point = gt.get("entry_point") if isinstance(gt, dict) else None
            if not isinstance(test_code, str):
                return _err_result(data_source, "HumanEval+ ground_truth 缺 'test' 字段")
            passed, _ = heplus_check_correctness(test_code, model_code, entry_point)
        else:
            raise NotImplementedError(f"code_eval_v2 未实现 data_source={data_source}")
    except Exception as e:
        traceback.print_exc()
        return _err_result(data_source, f"runner exception: {e}")

    return {"score": 1.0 if passed else 0.0, "acc": bool(passed), "format_score": 1.0}


# ============================================================================
# 辅助函数
# ============================================================================
import ast
import base64
import pickle
import zlib


def _load_ground_truth(ground_truth: Any) -> Any:
    """解析 ground_truth 字段."""
    if isinstance(ground_truth, (dict, list)):
        return ground_truth
    if not isinstance(ground_truth, str):
        return ground_truth
    try:
        return json.loads(ground_truth)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        decoded = pickle.loads(zlib.decompress(base64.b64decode(ground_truth.encode("utf-8"))))
        if isinstance(decoded, str):
            return json.loads(decoded)
        return decoded
    except Exception:
        try:
            return ast.literal_eval(ground_truth)
        except Exception:
            return ground_truth


def _lcb_dict_to_list(d: dict) -> list[dict]:
    inputs = d.get("inputs", [])
    outputs = d.get("outputs", [])
    fn_name = d.get("fn_name", None) or d.get("func_name", None)
    cases = []
    for inp, out in zip(inputs, outputs, strict=False):
        case = {"input": inp, "output": out, "metadata": {}}
        if fn_name is not None:
            case["testtype"] = "functional"
            case["metadata"]["func_name"] = fn_name
        else:
            case["testtype"] = "stdin"
        cases.append(case)
    return cases


def _err_result(data_source: str, msg: str) -> dict:
    print(f"[code_eval_v2][WARN] {data_source}: {msg}")
    return {"score": 0.0, "acc": False, "format_score": 0.0}


# ============================================================================
# verl hydra 兼容接口
# ============================================================================
def reward_func(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    sandbox_fusion_url: Optional[str] = None,
    concurrent_semaphore: Any = None,
) -> dict:
    """供 verl hydra custom_reward_function.name=reward_func 调用."""
    try:
        return compute_score(data_source, solution_str, ground_truth, extra_info)
    except Exception as e:
        print(f"[code_eval_v2][ERROR] data_source={data_source}: {e}")
        traceback.print_exc()
        return _err_result(data_source, f"top-level exception: {e}")


__all__ = [
    "compute_score",
    "reward_func",
    "extract_code_from_model",
    "lcb_check_correctness",
    "lcb_check_correctness_batch",
    "heplus_check_correctness",
    "heplus_check_correctness_batch",
]
