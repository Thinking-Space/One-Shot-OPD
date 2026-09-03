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
"""FSDP teacher worker for On-Policy Distillation (OPD / MOPD).

Runs one or more frozen teacher models under FSDP and scores the *student's own
rollout* with them, producing a dense per-token reward

    rm_scores[t] = -(log p_student(y_t) - log p_teacher(y_t))

which the ``token_reward_direct`` advantage estimator consumes directly.

Relationship to upstream verl 0.9.0:

* This is NOT upstream's ``distillation.teacher_models`` teacher loop. That path
  serves teachers as vLLM/SGLang inference replicas; here we need an exact FSDP
  forward pass to obtain per-token log-probs and top-K logits.
* This is NOT a reward model. verl 0.9.0 removed ``RewardModelWorker``
  entirely, and ``reward_model.*`` is left untouched for the verifier path.
  The teacher is loaded as ``AutoModelForCausalLM``, not a scoring head.

Only the vLLM + FSDP path is supported; no sglang or megatron code is involved.
"""

import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf

from verl.protocol import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id, get_device_name
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.config import FsdpTeacherConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = [
    "TeacherWorker",
    "compute_top_k_teacher_log_probs",
    "compute_top_k_reward",
    "group_rows_by_teacher",
    "local_micro_batch_counts",
]


def group_rows_by_teacher(routes) -> dict[str, list[int]]:
    """Map teacher name -> the row indices routed to it, in batch order."""
    order: dict[str, list[int]] = {}
    for i, name in enumerate(routes):
        order.setdefault(name, []).append(i)
    return order


def local_micro_batch_counts(order: dict[str, list[int]], teacher_names: list[str], micro_bs: int) -> list[int]:
    """Micro-batches this rank needs per teacher, indexed like ``teacher_names``.

    Deliberately keyed on the full configured teacher list rather than on the
    teachers present locally: the caller all-reduces these counts to a global
    maximum so every rank issues the same sequence of (FSDP-collective) teacher
    forwards. A teacher missing from this rank's shard must still appear here,
    as a zero, or the reduction would compare different positions across ranks.
    """
    return [(len(order.get(name, [])) + micro_bs - 1) // micro_bs for name in teacher_names]


def compute_top_k_teacher_log_probs(
    logits: torch.Tensor,
    student_ids: torch.Tensor,
    top_k: int,
    strategy: str = "only_stu",
):
    """Teacher log-probs over a top-K candidate set, plus student/teacher overlap.

    Pure tensor math (CPU-testable). Shapes: ``logits`` is ``(N, V)``,
    ``student_ids`` is ``(N, K_s)``.

    Returns a dict with:
        log_probs:          ``(N, K_s)`` teacher log-prob of each student candidate
        teacher_top_k_ids:  ``(N, top_k)`` teacher's own top-K token ids
        teacher_log_probs:  ``(N, top_k)`` teacher log-prob of those ids
        student_in_teacher: ``(N, K_s)`` bool, student candidate is in teacher top-K
        teacher_in_student: ``(N, top_k)`` bool, teacher candidate is in student set
        overlap_ratio:      scalar fraction of student candidates inside teacher top-K
    """
    log_denom = torch.logsumexp(logits.float(), dim=-1, keepdim=True)

    t_logits, t_ids = torch.topk(logits.float(), k=top_k, dim=-1)
    teacher_log_probs = t_logits - log_denom

    # (N, K_s, top_k) match matrix between the two candidate sets.
    matches = student_ids.unsqueeze(-1) == t_ids.unsqueeze(-2)
    student_in_teacher = matches.any(dim=-1)
    teacher_in_student = matches.any(dim=-2)

    if strategy == "intersection":
        # Only score student candidates the teacher also ranks highly.
        masked = t_logits.unsqueeze(-2).masked_fill(~matches, float("-inf"))
        log_probs = masked.max(dim=-1).values - log_denom
    else:
        log_probs = torch.gather(logits.float(), -1, student_ids) - log_denom

    return {
        "log_probs": log_probs,
        "teacher_top_k_ids": t_ids,
        "teacher_log_probs": teacher_log_probs,
        "student_in_teacher": student_in_teacher,
        "teacher_in_student": teacher_in_student,
        "overlap_ratio": student_in_teacher.float().mean(),
    }


def compute_reward_weights(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    weight_mode: str,
    normalize: bool = True,
) -> torch.Tensor:
    """Per-candidate weights for the top-K reward."""
    if weight_mode == "student_p":
        log_probs = student_log_probs
    elif weight_mode == "teacher_p":
        log_probs = teacher_log_probs
    elif weight_mode == "none":
        log_probs = torch.zeros_like(student_log_probs)
    else:
        raise ValueError(f"[OPD] unknown reward_weight_mode: {weight_mode}")

    log_probs = torch.where(valid_mask, log_probs, torch.full_like(log_probs, float("-inf")))
    if normalize:
        weights = torch.exp(log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True))
    else:
        weights = torch.exp(log_probs)
    return torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)


def compute_top_k_reward(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_in_teacher: torch.Tensor,
    teacher_in_student: torch.Tensor,
    strategy: str,
    weight_mode: str,
) -> torch.Tensor:
    """Weighted top-K distillation reward. Pure tensor math (CPU-testable).

    All inputs are ``(..., K)``. Returns ``(..., K)`` rewards, which
    ``token_reward_direct`` masks and uses as the advantage directly.
    """
    if strategy in ("only_stu", "only_tch"):
        valid = torch.ones_like(student_log_probs, dtype=torch.bool)
        normalize = True
    elif strategy == "intersection":
        valid = student_in_teacher.bool()
        normalize = True
    elif strategy == "union":
        valid = torch.cat(
            [torch.ones_like(student_in_teacher, dtype=torch.bool), ~teacher_in_student.bool()], dim=-1
        )
        normalize = True
    elif strategy == "union-intersection":
        valid = torch.cat([~student_in_teacher.bool(), ~teacher_in_student.bool()], dim=-1)
        # Deliberately unnormalized: this strategy weights by raw probability
        # mass so that the symmetric-difference candidates keep their scale.
        normalize = False
    else:
        raise ValueError(f"[OPD] unknown top_k_strategy: {strategy}")

    kl = student_log_probs - teacher_log_probs
    if strategy == "intersection":
        kl = torch.where(valid, kl, torch.zeros_like(kl))

    weights = compute_reward_weights(
        student_log_probs, teacher_log_probs, valid[..., : student_log_probs.shape[-1]], weight_mode, normalize
    )
    return -kl * weights


class TeacherWorker(Worker):
    """Ray worker hosting one or more frozen FSDP teachers for OPD/MOPD."""

    def __init__(self, config: DictConfig):
        Worker.__init__(self)
        # Same entry point the actor/critic workers use, so the teacher joins the
        # existing Ray process group instead of creating a second one.
        initialize_global_process_group_ray(timeout_second=None)
        self.config = config
        teacher_cfg = config.distillation.fsdp_teacher
        if not isinstance(teacher_cfg, FsdpTeacherConfig):
            teacher_cfg = FsdpTeacherConfig(**convert_to_regular_types(OmegaConf.to_container(teacher_cfg)))
        self.teacher_config: FsdpTeacherConfig = teacher_cfg
        self.teacher_modules: dict[str, torch.nn.Module] = {}
        self.tokenizer = None

    def _build_model(self, model_path: str):
        """Load one frozen teacher and wrap it in FSDP."""
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModelForCausalLM

        from verl.models.transformers.monkey_patch import apply_monkey_patch

        dtype = PrecisionType.to_dtype(self.teacher_config.dtype)
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        init_ctx = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)
        with init_ctx():
            module = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=model_config,
                torch_dtype=dtype,
                attn_implementation=self.teacher_config.attn_implementation,
                trust_remote_code=True,
            )
            apply_monkey_patch(
                model=module,
                ulysses_sp_size=self.teacher_config.ulysses_sequence_parallel_size,
                use_remove_padding=self.teacher_config.use_remove_padding,
            )
            module.to(dtype)

        module.eval()
        for param in module.parameters():
            param.requires_grad = False

        fsdp_config = self.teacher_config.fsdp_config or {}
        module = FSDP(
            module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=get_fsdp_wrap_policy(module=module, config=fsdp_config.get("wrap_policy")),
            device_id=get_device_id(),
            sync_module_states=True,
            # The teacher is frozen and only ever runs forward, so params can
            # live on CPU between micro-batches without a recompute penalty.
            cpu_offload=CPUOffload(offload_params=fsdp_config.get("param_offload", True)),
            mixed_precision=MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype),
            forward_prefetch=fsdp_config.get("forward_prefetch", False),
        )
        return module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        paths = self.teacher_config.teacher_paths()
        if not paths:
            paths = {self.teacher_config.default_teacher or "default": self.teacher_config.model_path}

        logger.info("[MOPD] Building %d teacher modules", len(paths))
        for name, path in paths.items():
            logger.info("[MOPD] Loading teacher '%s' from %s", name, path)
            self.teacher_modules[name] = self._build_model(path)
            logger.info("[MOPD] Teacher '%s' loaded", name)

        if self.teacher_config.is_multi_teacher:
            logger.info(
                "[MOPD] Multi-teacher routing ready: teachers=%s ability_to_teacher=%s strict=%s",
                sorted(paths),
                dict(self.teacher_config.ability_to_teacher),
                self.teacher_config.strict_routing,
            )

    @torch.no_grad()
    def _forward_micro_batch(
        self,
        module,
        micro_batch: dict,
        student_top_k_ids: Optional[torch.Tensor] = None,
        top_k: int = 0,
    ):
        """Teacher forward over one micro-batch; returns per-token teacher log-probs."""
        from verl.utils.torch_functional import logprobs_from_logits

        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        response_length = micro_batch["responses"].size(-1)

        with torch.autocast(device_type=get_device_name(), dtype=PrecisionType.to_dtype(self.teacher_config.dtype)):
            output = module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            # Predict token t from position t-1, so the response slice is shifted left by one.
            logits = output.logits[:, -response_length - 1 : -1, :]
            logits.div_(self.teacher_config.teacher_temperature)

            responses = micro_batch["responses"]
            log_probs = logprobs_from_logits(logits, responses)

            result = {"teacher_log_probs": log_probs}
            if top_k > 0 and student_top_k_ids is not None:
                flat = compute_top_k_teacher_log_probs(
                    logits.reshape(-1, logits.size(-1)),
                    student_top_k_ids.reshape(-1, student_top_k_ids.size(-1)),
                    top_k=top_k,
                    strategy=self.teacher_config.top_k_strategy,
                )
                bsz, seqlen = log_probs.shape
                result["teacher_top_k_log_probs"] = flat["log_probs"].view(bsz, seqlen, -1)
                result["teacher_top_k_ids"] = flat["teacher_top_k_ids"].view(bsz, seqlen, -1)
                result["student_in_teacher"] = flat["student_in_teacher"].view(bsz, seqlen, -1)
                result["teacher_in_student"] = flat["teacher_in_student"].view(bsz, seqlen, -1)
                result["teacher_own_top_k_log_probs"] = flat["teacher_log_probs"].view(bsz, seqlen, -1)
        return result

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto) -> DataProto:
        """Score the student rollout with the routed teacher(s).

        Rows are grouped by their routed teacher so each teacher runs one
        contiguous forward pass, then results are scattered back into the
        original batch order.

        When the student's top-K support set is present in the batch, the
        teacher also scores those candidates. Whether that top-K reward
        *replaces* the plain reverse KL is decided by the trainer via
        ``meta_info["opd_top_k_is_reward"]``: in metric-only mode the overlap
        statistics are still reported but ``rm_scores`` stays (B, T).
        """
        data = data.to(get_device_id())

        abilities = data.non_tensor_batch.get("ability")
        n = len(data)
        routes = self.teacher_config.route(abilities, n)

        # Group rows by teacher, remembering where each came from.
        order = group_rows_by_teacher(routes)

        response_mask = data.batch["response_mask"] if "response_mask" in data.batch.keys() else None
        student_log_probs = data.batch["old_log_probs"]
        teacher_log_probs = torch.empty_like(student_log_probs)

        student_top_k_ids = data.batch.get("student_top_k_ids")
        student_top_k_log_probs = data.batch.get("student_top_k_log_probs")
        top_k = self.teacher_config.log_prob_top_k or self.teacher_config.metric_top_k
        use_top_k = top_k > 0 and student_top_k_ids is not None and student_top_k_log_probs is not None
        top_k_is_reward = bool(data.meta_info.get("opd_top_k_is_reward", False))

        top_k_parts: dict[str, torch.Tensor] = {}
        micro_bs = self.teacher_config.micro_batch_size_per_gpu

        # Every rank has to issue the *same sequence* of teacher forwards. The
        # teachers are FSDP-sharded, so each forward is a collective: a rank whose
        # shard happens to hold no `coding` rows would skip the code teacher and
        # hang every rank that does hold some, waiting in all-gather. Iterating
        # `order` gets all three of those wrong at once — it visits only the
        # teachers present locally, in local first-appearance order, for a locally
        # determined number of micro-batches. So walk every configured teacher in
        # a fixed order instead, and pad each one's micro-batch count up to the
        # global maximum with a throwaway forward.
        teacher_names = sorted(self.teacher_modules)
        n_micro = torch.tensor(
            local_micro_batch_counts(order, teacher_names, micro_bs),
            device=teacher_log_probs.device,
            dtype=torch.long,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(n_micro, op=torch.distributed.ReduceOp.MAX)

        for t, name in enumerate(teacher_names):
            module = self.teacher_modules[name]
            idxs = order.get(name, [])
            for step in range(int(n_micro[t].item())):
                chunk = idxs[step * micro_bs : (step + 1) * micro_bs]
                # This rank is out of rows for this teacher but the collective still
                # has to happen: score row 0 and drop the result.
                rows = chunk or [0]
                micro = {k: data.batch[k][rows] for k in ("input_ids", "attention_mask", "position_ids", "responses")}
                out = self._forward_micro_batch(
                    module,
                    micro,
                    student_top_k_ids=student_top_k_ids[rows] if use_top_k else None,
                    top_k=top_k if use_top_k else 0,
                )
                if not chunk:
                    continue
                teacher_log_probs[chunk] = out["teacher_log_probs"].to(teacher_log_probs.dtype)
                for key in (
                    "teacher_top_k_log_probs",
                    "student_in_teacher",
                    "teacher_in_student",
                ):
                    if key not in out:
                        continue
                    value = out[key]
                    if key not in top_k_parts:
                        top_k_parts[key] = torch.empty(
                            (n, *value.shape[1:]), dtype=value.dtype, device=value.device
                        )
                    # Scatter back by original row index: `order` groups rows out
                    # of batch order, so a plain concat would misalign them.
                    top_k_parts[key][chunk] = value

        # Reverse KL as a dense per-token reward.
        rm_scores = -(student_log_probs - teacher_log_probs)
        if response_mask is not None:
            rm_scores = rm_scores * response_mask

        tensors = {"rm_scores": rm_scores, "teacher_log_probs": teacher_log_probs}
        # Per-row teacher names survive DP concat. A counts dict in meta_info
        # does not: DataProto.concat requires non-metric meta_info to be
        # identical across ranks, and each rank saw a different shard.
        non_tensors = {"mopd_teacher": np.array(routes, dtype=object)}
        metrics: dict = {"mopd_teacher_counts": {k: len(v) for k, v in sorted(order.items())}}

        if use_top_k:
            student_in_teacher = top_k_parts["student_in_teacher"]
            teacher_in_student = top_k_parts["teacher_in_student"]
            student_lp = student_top_k_log_probs.to(rm_scores.dtype)
            teacher_lp = top_k_parts["teacher_top_k_log_probs"].to(rm_scores.dtype)

            top_k_reward = compute_top_k_reward(
                student_log_probs=student_lp,
                teacher_log_probs=teacher_lp,
                student_in_teacher=student_in_teacher,
                teacher_in_student=teacher_in_student,
                strategy=self.teacher_config.top_k_strategy,
                weight_mode=self.teacher_config.reward_weight_mode,
            )

            mask = response_mask.unsqueeze(-1) if response_mask is not None else None
            if mask is not None:
                top_k_reward = top_k_reward * mask

            denom = float(mask.sum().item()) if mask is not None else float(student_in_teacher[..., :1].numel())
            denom = max(denom, 1.0)
            overlap = student_in_teacher.float()
            if response_mask is not None:
                overlap = overlap * response_mask.unsqueeze(-1)
            metrics["top_k_overlap_ratio"] = float(overlap.sum().item()) / (denom * student_in_teacher.shape[-1])
            metrics["top_k_student_mass"] = self._masked_mass(student_lp, response_mask)
            metrics["top_k_teacher_mass"] = self._masked_mass(teacher_lp, response_mask)
            metrics["top_k_reward_mean"] = float(top_k_reward.sum().item()) / denom

            tensors["student_in_teacher"] = student_in_teacher
            tensors["teacher_in_student"] = teacher_in_student
            tensors["teacher_top_k_log_probs"] = teacher_lp
            if top_k_is_reward:
                # Only now does top-K actually drive training; in metric-only
                # mode rm_scores stays the plain (B, T) reverse KL above.
                tensors["rm_scores"] = top_k_reward
            else:
                tensors["top_k_reward"] = top_k_reward

        output = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
        output.meta_info["metrics"] = metrics
        return output.to("cpu")

    @staticmethod
    def _masked_mass(log_probs: torch.Tensor, response_mask: Optional[torch.Tensor]) -> float:
        """Mean probability mass captured by the K candidates, over real tokens."""
        mass = log_probs.exp().sum(dim=-1)
        if response_mask is None:
            return float(mass.mean().item())
        total = float(response_mask.sum().item())
        return float((mass * response_mask).sum().item()) / max(total, 1.0)
