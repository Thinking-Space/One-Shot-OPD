"""Function-Calling 评测统一入口 (OPD FC 任务).

设计目标:
1. 训练监控: xlam 训练时, GT 是 list[{name, arguments}] 的 JSON,
   reward = 是否 exact match (1 个 score, 写入 critic/true_reward).
2. 验证评测: BFCL v3 (simple/multiple/parallel/parallel_multiple/live_simple)
   GT 是 list[{fn_name: {arg: [可能值1, 可能值2, ...]}}] 的 JSON,
   每个参数任一可能值匹配即可, 空字符串 "" 代表参数可缺失.

支持的 data_source:
  - "xlam"
  - "bfcl_simple" / "bfcl_multiple" / "bfcl_parallel" /
    "bfcl_parallel_multiple" / "bfcl_live_simple"

ground_truth 传入约定 (parquet 里存 JSON 字符串):
  - xlam : '[{"name": "...", "arguments": {...}}, ...]'   (严格值)
  - bfcl : '[{"fn_name": {"arg": [val1, val2, ...]}}, ...]'  (任一可能值)

返回:
  compute_score -> dict {"score": 0.0/1.0, "acc": bool, "format_score": float}
  reward_func   -> 同 ttrl_math.reward_func 接口
"""

from __future__ import annotations

import ast
import json
import re
import traceback
from typing import Any, Optional


# ----------------------------------------------------------------------------
# 从模型输出中提取 function call JSON list
# ----------------------------------------------------------------------------
def _try_json_then_pyliteral(s: str) -> Optional[Any]:
    """尝试 json.loads, 失败再 ast.literal_eval (兼容 Python dict 风格)."""
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    return None


def extract_fc_calls(model_response: str) -> Optional[list]:
    """从模型输出中抠出 function call list.

    支持的输出格式:
      1. 整段就是 JSON list:                  [{"name": "f", "arguments": {...}}]
      2. ```json ... ``` 包裹的 markdown:    ```json\n[...]\n```
      3. ``` ... ``` 包裹的 markdown:         ```\n[...]\n```
      4. 散落在文本中, 最后一个看起来像 list 的片段
      5. 单个 dict (没包成 list): {"name": "f", ...} → 包成 [dict]

    Args:
        model_response: 模型完整输出.

    Returns:
        list[{"name": str, "arguments": dict}] | None.
        失败返回 None.
    """
    if not isinstance(model_response, str) or not model_response.strip():
        return None

    candidates: list[str] = []

    # 1. markdown 代码块
    for m in re.finditer(r"```(?:json|python)?\s*\n(.*?)```", model_response, re.DOTALL):
        candidates.append(m.group(1).strip())

    # 2. 整段
    candidates.append(model_response.strip())

    # 3. 最长 [...] 片段 (贪婪, 适合裸 JSON list)
    for m in re.finditer(r"\[\s*\{.*?\}\s*\]", model_response, re.DOTALL):
        candidates.append(m.group(0).strip())

    # 4. 最长 {...} 片段 (单个 dict)
    for m in re.finditer(r"\{[^{}]*\"name\"[^{}]*\"arguments\"[^{}]*\}", model_response, re.DOTALL):
        candidates.append(m.group(0).strip())

    for cand in candidates:
        obj = _try_json_then_pyliteral(cand)
        if obj is None:
            continue
        if isinstance(obj, dict):
            if "name" in obj:
                return [obj]
            continue
        if isinstance(obj, list) and all(
            isinstance(c, dict) and "name" in c for c in obj
        ):
            return obj
    return None


# ----------------------------------------------------------------------------
# 训练侧 GT match (xlam: 严格值)
# ----------------------------------------------------------------------------
def _normalize_value(v: Any) -> Any:
    """把 value 规范化, 让 "1" 和 1, True 和 "true" 之类的比较更鲁棒."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        s = v.strip()
        # 尝试 int / float
        try:
            iv = int(s)
            return iv
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        # 大小写不敏感 bool
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        return s
    return v


def _match_call_strict(pred: dict, gt: dict) -> bool:
    """xlam 风格严格匹配: name 相同且 arguments 完全匹配."""
    if not isinstance(pred, dict) or not isinstance(gt, dict):
        return False
    if pred.get("name") != gt.get("name"):
        return False
    pred_args = pred.get("arguments", {})
    gt_args = gt.get("arguments", {})
    if not isinstance(pred_args, dict) or not isinstance(gt_args, dict):
        return False
    # 所有 GT 必需参数都要在 pred 中且值匹配
    for k, v in gt_args.items():
        if k not in pred_args:
            return False
        if _normalize_value(pred_args[k]) != _normalize_value(v):
            return False
    # 不允许 pred 多余的"不在 GT 中"的参数 (xlam GT 就是完整答案)
    for k in pred_args:
        if k not in gt_args:
            return False
    return True


def _score_xlam(pred_calls: list[dict], gt_calls: list[dict]) -> bool:
    """xlam: pred_calls 要和 gt_calls 一一对应 (顺序无关).

    用贪心匹配: 每个 GT call 都要在 pred 中找到一个未配对的匹配.
    """
    if len(pred_calls) != len(gt_calls):
        return False
    used = [False] * len(pred_calls)
    for gt in gt_calls:
        found = False
        for i, pred in enumerate(pred_calls):
            if used[i]:
                continue
            if _match_call_strict(pred, gt):
                used[i] = True
                found = True
                break
        if not found:
            return False
    return True


# ----------------------------------------------------------------------------
# 验证侧 GT match (BFCL: possible_answer 任一匹配)
# ----------------------------------------------------------------------------
def _match_call_bfcl(pred: dict, gt_entry: dict) -> bool:
    """BFCL 风格匹配: gt_entry 形如 {fn_name: {arg: [可能值1, 可能值2, ...]}}.

    任一可能值匹配即可. 空字符串 "" 表示该参数可缺失.

    Args:
        pred    : 模型输出的单个 call dict, {"name": ..., "arguments": {...}}.
        gt_entry: BFCL 的单个 GT entry, {fn_name: {arg: list_of_possible_values}}.

    Returns:
        是否匹配.
    """
    if not isinstance(pred, dict) or not isinstance(gt_entry, dict):
        return False
    if len(gt_entry) != 1:
        return False
    gt_fn = next(iter(gt_entry.keys()))
    gt_args_spec = gt_entry[gt_fn]
    if not isinstance(gt_args_spec, dict):
        return False
    if pred.get("name") != gt_fn:
        return False
    pred_args = pred.get("arguments", {})
    if not isinstance(pred_args, dict):
        return False

    # 对每个 GT 参数: 找到任一可接受值匹配
    for arg_name, possible_values in gt_args_spec.items():
        if not isinstance(possible_values, list) or len(possible_values) == 0:
            return False
        # 空字符串 "" 表示可缺失, 其他值需要严格匹配
        accept_missing = any(
            isinstance(pv, str) and pv == "" for pv in possible_values
        )
        if arg_name not in pred_args:
            if accept_missing:
                continue
            return False
        pred_val = _normalize_value(pred_args[arg_name])
        matched = False
        for pv in possible_values:
            if isinstance(pv, str) and pv == "":
                continue
            if _normalize_value(pv) == pred_val:
                matched = True
                break
            # list 类型参数 (例如 [3, 5]): 严格列表比较
            if isinstance(pv, list) and isinstance(pred_args.get(arg_name), list):
                if [_normalize_value(x) for x in pv] == [
                    _normalize_value(x) for x in pred_args[arg_name]
                ]:
                    matched = True
                    break
        if not matched:
            return False

    # pred 里的额外参数: BFCL 允许 (例如 default 参数显式填了),
    # 只要不违反 GT 必填的, 就 ok.
    return True


def _score_bfcl(pred_calls: list[dict], gt_entries: list[dict]) -> bool:
    """BFCL: pred 和 GT 长度一致, 顺序无关一一匹配."""
    if len(pred_calls) != len(gt_entries):
        return False
    used = [False] * len(pred_calls)
    for gt in gt_entries:
        found = False
        for i, pred in enumerate(pred_calls):
            if used[i]:
                continue
            if _match_call_bfcl(pred, gt):
                used[i] = True
                found = True
                break
        if not found:
            return False
    return True


# ----------------------------------------------------------------------------
# 统一入口
# ----------------------------------------------------------------------------
_XLAM_SOURCES = {"xlam"}
_BFCL_SOURCES = {
    "bfcl_simple",
    "bfcl_multiple",
    "bfcl_parallel",
    "bfcl_parallel_multiple",
    "bfcl_live_simple",
}


def _load_ground_truth(gt: Any) -> Any:
    """parquet 里 GT 可能是 JSON str / dict / list."""
    if isinstance(gt, (list, dict)):
        return gt
    if not isinstance(gt, str):
        return gt
    obj = _try_json_then_pyliteral(gt)
    return obj if obj is not None else gt


def _format_score(format_ok: bool, content_ok: bool) -> dict:
    return {
        "score": 1.0 if content_ok else 0.0,
        "acc": bool(content_ok),
        "format_score": 1.0 if format_ok else 0.0,
    }


def compute_score(
    data_source: str,
    completion: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
) -> dict:
    """FC 评测统一入口.

    Args:
        data_source : "xlam" 或 "bfcl_*".
        completion  : 模型完整输出文本.
        ground_truth: GT, JSON 字符串或已 parse 的 list/dict.
        extra_info  : 暂未使用.

    Returns:
        {"score": 0/1, "acc": bool, "format_score": 0/1}.
    """
    pred_calls = extract_fc_calls(completion)
    format_ok = pred_calls is not None
    if not format_ok:
        return _format_score(format_ok=False, content_ok=False)

    gt = _load_ground_truth(ground_truth)
    if not isinstance(gt, list):
        return _format_score(format_ok=True, content_ok=False)

    try:
        if data_source in _XLAM_SOURCES:
            ok = _score_xlam(pred_calls, gt)
        elif data_source in _BFCL_SOURCES:
            ok = _score_bfcl(pred_calls, gt)
        else:
            print(f"[fc_eval][WARN] unknown data_source: {data_source}, fallback to bfcl matching")
            ok = _score_bfcl(pred_calls, gt)
    except Exception as e:
        print(f"[fc_eval][ERROR] match exception: {e}")
        traceback.print_exc()
        return _format_score(format_ok=True, content_ok=False)

    return _format_score(format_ok=True, content_ok=ok)


# 兼容 hydra custom_reward_function 接口
def reward_func(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    sandbox_fusion_url: Optional[str] = None,
    concurrent_semaphore: Any = None,
) -> dict:
    """供 verl hydra `custom_reward_function.name=reward_func` 调用."""
    try:
        return compute_score(data_source, solution_str, ground_truth, extra_info)
    except Exception as e:
        print(f"[fc_eval][ERROR] data_source={data_source}: {e}")
        traceback.print_exc()
        return {"score": 0.0, "acc": False, "format_score": 0.0}


__all__ = [
    "compute_score",
    "reward_func",
    "extract_fc_calls",
]
