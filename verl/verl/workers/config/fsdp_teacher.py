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
"""Configuration for the OPD FSDP teacher (single-teacher OPD and MOPD).

This is deliberately a *separate* namespace from upstream's
``distillation.teacher_models``. Upstream runs teachers as vLLM/SGLang
inference replicas behind an async teacher loop; OPD instead needs a plain FSDP
forward pass over the student's rollout to get exact per-token teacher
log-probs (and optionally top-K logits). The two cannot share a config block,
so OPD hangs off ``distillation.fsdp_teacher.*`` and leaves every upstream
distillation field untouched.
"""

import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["FsdpTeacherConfig", "route_by_ability"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

#: top-K reward strategies supported by the OPD teacher.
TOP_K_STRATEGIES = ("only_stu", "only_tch", "intersection", "union", "union-intersection")

#: how the per-candidate rewards are weighted when top-K is enabled.
REWARD_WEIGHT_MODES = ("student_p", "teacher_p", "none")


def route_by_ability(
    abilities,
    n_samples: int,
    ability_to_teacher: Optional[dict],
    default_teacher: str,
    strict_routing: bool = False,
) -> list[str]:
    """Map each row of a batch to the teacher that should score it.

    Pure function: no torch, no CUDA, no worker state, so it is unit-testable
    on CPU. This is the heart of MOPD.

    Args:
        abilities: per-row ``ability`` values (from ``non_tensor_batch``), or
            None when the column is missing entirely.
        n_samples: number of rows in the batch.
        ability_to_teacher: mapping of dataset ability -> teacher name. Empty or
            None means single-teacher mode (everything to ``default_teacher``).
        default_teacher: teacher used when no mapping applies.
        strict_routing: if True, a missing ``ability`` column or an ability that
            is absent from the mapping raises ValueError instead of silently
            falling back to ``default_teacher``.

    Returns:
        A list of teacher names, one per row, in the original batch order.
    """
    if not ability_to_teacher:
        return [default_teacher] * n_samples

    if abilities is None:
        if strict_routing:
            raise ValueError(
                "[MOPD] strict_routing=True but 'ability' field missing from batch. "
                "Ensure the training parquet carries an 'ability' column."
            )
        logger.warning("[MOPD] 'ability' missing from batch; routing all samples to '%s'", default_teacher)
        return [default_teacher] * n_samples

    result = []
    for i in range(n_samples):
        ability = str(abilities[i])
        teacher = ability_to_teacher.get(ability)
        if teacher is None:
            if strict_routing:
                raise ValueError(
                    f"[MOPD] strict_routing=True but ability '{ability}' not in "
                    f"ability_to_teacher: {sorted(ability_to_teacher)}"
                )
            logger.warning("[MOPD] unknown ability '%s'; falling back to '%s'", ability, default_teacher)
            teacher = default_teacher
        result.append(teacher)

    logger.info("[MOPD] batch routing: %s", dict(sorted(Counter(result).items())))
    return result


@dataclass
class FsdpTeacherConfig(BaseConfig):
    """OPD teacher configuration.

    Args:
        enabled: master switch for OPD teacher scoring.
        model_path: single-teacher model path. Ignored when ``teachers`` is set.
        teachers: MOPD teacher list, ``[{name: str, path: str}, ...]``.
        ability_to_teacher: dataset ability -> teacher name.
        strict_routing: raise instead of falling back on unknown/missing ability.
        default_teacher: teacher used when routing does not resolve. Defaults to
            the first entry of ``teachers``.
        micro_batch_size_per_gpu: teacher forward micro-batch size.
        teacher_temperature: temperature applied to teacher logits before
            computing log-probs.
        log_prob_top_k: K for top-K distillation, used as the *training reward*.
            0 disables it and the reward is the plain per-token reverse KL.
        top_k_strategy: which candidate set the top-K reward is built over.
        reward_weight_mode: how top-K candidates are weighted.
        kl_estimator: KL estimator used for the token reward.
        metric_top_k: K used for top-K *observability only* (0 = off). The
            student/teacher top-K tensors are computed and the agreement
            metrics are logged, but the reward stays the plain reverse KL.
            Ignored when ``log_prob_top_k > 0``.
        attn_implementation: HuggingFace attention backend for the teacher
            forward. Default ``flash_attention_2``; set ``sdpa`` when
            flash-attn is unavailable for the installed torch/CUDA pair.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teachers", "ability_to_teacher", "default_teacher"}

    enabled: bool = False
    model_path: Optional[str] = None

    # --- MOPD ---
    teachers: list[Any] = field(default_factory=list)
    ability_to_teacher: dict[str, str] = field(default_factory=dict)
    strict_routing: bool = False
    default_teacher: Optional[str] = None

    # --- forward ---
    micro_batch_size_per_gpu: int = 24
    dtype: str = "bfloat16"
    use_remove_padding: bool = True
    use_dynamic_bsz: bool = False
    forward_max_token_len_per_gpu: int = 32768
    ulysses_sequence_parallel_size: int = 1
    fsdp_config: dict[str, Any] = field(default_factory=dict)

    # --- reward shaping ---
    teacher_temperature: float = 1.0
    log_prob_top_k: int = 0
    top_k_strategy: str = "only_stu"
    reward_weight_mode: str = "student_p"
    kl_estimator: str = "k1"
    metric_top_k: int = 0
    attn_implementation: str = "flash_attention_2"

    def __post_init__(self):
        if not self.enabled:
            return

        if self.top_k_strategy not in TOP_K_STRATEGIES:
            raise ValueError(
                f"[OPD] top_k_strategy must be one of {list(TOP_K_STRATEGIES)}, got '{self.top_k_strategy}'"
            )
        if self.reward_weight_mode not in REWARD_WEIGHT_MODES:
            raise ValueError(
                f"[OPD] reward_weight_mode must be one of {list(REWARD_WEIGHT_MODES)}, "
                f"got '{self.reward_weight_mode}'"
            )
        if self.log_prob_top_k < 0:
            raise ValueError(f"[OPD] log_prob_top_k must be >= 0, got {self.log_prob_top_k}")
        if self.metric_top_k < 0:
            raise ValueError(f"[OPD] metric_top_k must be >= 0, got {self.metric_top_k}")
        if self.log_prob_top_k > 0 and self.metric_top_k > 0 and self.metric_top_k != self.log_prob_top_k:
            # Measuring one K while training on another would make the logged
            # agreement describe a different distribution than the reward.
            logger.warning(
                "[OPD] both log_prob_top_k=%d and metric_top_k=%d are set; metric_top_k is ignored "
                "because top-K is driving the reward.",
                self.log_prob_top_k,
                self.metric_top_k,
            )

        names = self.teacher_names()
        if not names and not self.model_path:
            raise ValueError(
                "[OPD] distillation.fsdp_teacher.enabled=True requires either "
                "`model_path` (single teacher) or `teachers` (MOPD)."
            )
        if len(names) != len(set(names)):
            raise ValueError(f"[OPD] duplicate teacher names in `teachers`: {names}")

        if names and self.default_teacher is None:
            self.default_teacher = names[0]
        if self.default_teacher is not None and names and self.default_teacher not in names:
            raise ValueError(f"[OPD] default_teacher '{self.default_teacher}' is not one of {names}")

        for ability, teacher in dict(self.ability_to_teacher).items():
            if names and teacher not in names:
                raise ValueError(
                    f"[MOPD] ability_to_teacher maps '{ability}' -> '{teacher}', "
                    f"which is not a configured teacher {names}"
                )

    def teacher_names(self) -> list[str]:
        """Names of the configured MOPD teachers, in declaration order."""
        return [t["name"] for t in self.teachers]

    def teacher_paths(self) -> dict[str, str]:
        """Teacher name -> model path."""
        return {t["name"]: t["path"] for t in self.teachers}

    @property
    def is_multi_teacher(self) -> bool:
        return len(self.teachers) > 1

    def route(self, abilities, n_samples: int) -> list[str]:
        """Route a batch to teachers. See :func:`route_by_ability`."""
        default = self.default_teacher
        if default is None:
            names = self.teacher_names()
            default = names[0] if names else "default"
        return route_by_ability(
            abilities=abilities,
            n_samples=n_samples,
            ability_to_teacher=dict(self.ability_to_teacher),
            default_teacher=default,
            strict_routing=self.strict_routing,
        )
