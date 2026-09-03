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
"""OPD extensions for :class:`~verl.trainer.ppo.ray_trainer.RayPPOTrainer`.

Everything OPD adds on the trainer side lives here as a mixin so the upstream
trainer keeps its shape -- ``RayPPOTrainer`` inherits it and calls into it from
three short hook sites, rather than being forked into a second trainer class.
That is what keeps ``python -m verl.trainer.main_ppo`` the single entry point
for all four scenarios (OPD / MOPD / 1-shot OPD / template).

* :meth:`OPDTrainerMixin.register_opd_teacher` -- add the teacher worker to the
  actor's colocated worker set, so the teacher shares the actor's GPUs instead
  of demanding a second resource pool.
* :meth:`OPDTrainerMixin.init_opd_teacher` -- load the teacher weights, after
  the actor/rollout group is up so vLLM's KV-cache estimate is unaffected.
* :meth:`OPDTrainerMixin.compute_opd_token_reward` -- run the teacher over the
  student rollout and install the resulting dense per-token reward as
  ``token_level_scores``, preserving the verifier's own score under
  ``true_token_level_scores`` for ``token_reward_direct_plus_grpo``.
* :meth:`OPDTrainerMixin.log_template_metrics` -- filter statistics for
  template mode's self-generated queries.

All of it is inert unless ``distillation.fsdp_teacher.enabled=True`` (or
``template.enabled=True``), so an unmodified PPO/GRPO run behaves exactly as
upstream does.
"""

import logging
import os
from collections import Counter
from typing import Any

import torch

from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import FsdpTeacherConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["OPDTrainerMixin", "opd_teacher_enabled", "template_enabled"]

def _mean_metric(value):
    """Average a worker metric that may be a scalar or a DP-concat list."""
    if value is None:
        return None
    if isinstance(value, list):
        nums = [float(x) for x in value if x is not None]
        return sum(nums) / max(len(nums), 1) if nums else None
    return float(value)


#: Key under which the teacher worker is registered in the colocated worker set.
OPD_TEACHER_KEY = "opd_teacher"


def opd_teacher_enabled(config) -> bool:
    """Whether the OPD FSDP teacher should be built for this run."""
    distillation = config.get("distillation", None)
    if distillation is None:
        return False
    fsdp_teacher = distillation.get("fsdp_teacher", None)
    if fsdp_teacher is None:
        return False
    return bool(fsdp_teacher.get("enabled", False))


def template_enabled(config) -> bool:
    """Whether template mode (student writes its own query) is on."""
    template = config.get("template", None)
    return bool(template is not None and template.get("enabled", False))


class OPDTrainerMixin:
    """Teacher-reward and template-mode hooks for the PPO trainer."""

    # ------------------------------------------------------------------
    # teacher worker lifecycle
    # ------------------------------------------------------------------
    def register_opd_teacher(self, resource_pool) -> None:
        """Colocate the teacher worker with the actor on ``resource_pool``.

        The teacher is forward-only and its params are CPU-offloaded between
        micro-batches, so sharing the actor's GPUs costs far less than standing
        up a second resource pool would.
        """
        from verl.single_controller.ray import RayClassWithInitArgs
        from verl.workers.fsdp_teacher_worker import TeacherWorker

        import ray

        self.opd_teacher_config: FsdpTeacherConfig = omega_conf_to_dataclass(self.config.distillation.fsdp_teacher)
        teacher_cls = RayClassWithInitArgs(cls=ray.remote(TeacherWorker), config=self.config)
        self.resource_pool_to_cls[resource_pool][OPD_TEACHER_KEY] = teacher_cls

    def init_opd_teacher(self, all_wg: dict) -> None:
        """Pick the teacher group out of the spawned workers and load its weights."""
        self.opd_teacher_wg = all_wg[OPD_TEACHER_KEY]
        self.opd_teacher_wg.init_model()

    # ------------------------------------------------------------------
    # top-K: reward mode vs metric-only mode
    # ------------------------------------------------------------------
    def opd_student_top_k(self) -> int:
        """K to extract from the student, or 0 to skip top-K entirely.

        Two independent knobs decide this, and they mean different things:

        * ``log_prob_top_k > 0`` -- top-K *drives the reward*. The token reward
          becomes the (B, T, K) top-K distillation reward.
        * ``metric_top_k > 0`` -- top-K is computed for *observability only*.
          The reward stays the plain (B, T) reverse KL; the top-K tensors are
          used solely to log student/teacher agreement.

        ``log_prob_top_k`` wins when both are set, because you cannot train on
        one K while reporting another without the metric describing a different
        distribution than the reward. Metric mode exists so a run can measure
        how much top-K would change the signal before switching a real training
        run over to it.
        """
        if not getattr(self, "use_opd_teacher", False):
            return 0
        teacher = self.config.distillation.fsdp_teacher
        train_top_k = int(teacher.get("log_prob_top_k", 0) or 0)
        if train_top_k > 0:
            return train_top_k
        return int(teacher.get("metric_top_k", 0) or 0)

    def opd_top_k_is_reward(self) -> bool:
        """True when top-K should replace the reward, not merely be measured."""
        if not getattr(self, "use_opd_teacher", False):
            return False
        return int(self.config.distillation.fsdp_teacher.get("log_prob_top_k", 0) or 0) > 0

    def validate_opd_top_k_config(self) -> None:
        """Reject top-K reward setups whose candidate set would drift under training.

        The training forward re-derives the student's top-K from the current
        logits; the advantage it is multiplied against was built from the top-K
        taken at rollout time. Those two agree only while the actor has not been
        updated since the rollout -- i.e. one epoch over one mini-batch that spans
        the whole rollout batch. Any further update makes the ratio compare
        log-probs of two different candidate sets, which is silently wrong rather
        than loudly broken, so it is refused up front.
        """
        if not self.opd_top_k_is_reward():
            return
        actor = self.config.actor_rollout_ref.actor
        ppo_epochs = int(actor.ppo_epochs)
        mini_batch_size = int(actor.ppo_mini_batch_size)
        train_batch_size = int(self.config.data.train_batch_size)
        if ppo_epochs != 1 or mini_batch_size != train_batch_size:
            raise ValueError(
                "OPD top-K reward mode (distillation.fsdp_teacher.log_prob_top_k > 0) requires strictly "
                "on-policy updates: actor.ppo_epochs=1 and actor.ppo_mini_batch_size == data.train_batch_size. "
                f"Got ppo_epochs={ppo_epochs}, ppo_mini_batch_size={mini_batch_size}, "
                f"train_batch_size={train_batch_size}. Off-policy updates re-derive a different top-K "
                "candidate set than the one the advantage was built from. Use metric_top_k instead if you "
                "only want the agreement statistics."
            )

    # ------------------------------------------------------------------
    # teacher reward
    # ------------------------------------------------------------------
    def compute_opd_token_reward(self, batch: DataProto, metrics: dict[str, Any]) -> DataProto:
        """Replace the outcome reward with the teacher's dense per-token reward.

        ``token_level_scores`` normally holds the verifier's sparse score. OPD
        needs the dense teacher reward there instead, because that is what the
        advantage estimator consumes. The verifier's own tensor is kept under
        ``true_token_level_scores`` so ``token_reward_direct_plus_grpo`` can
        still build a GRPO outcome term from it, and so the usual
        ``critic/score`` metrics stay meaningful.

        When top-K runs in metric-only mode the teacher still computes the
        overlap statistics, but ``rm_scores`` stays the plain (B, T) reverse KL.
        """
        batch.meta_info["opd_top_k_is_reward"] = self.opd_top_k_is_reward()
        teacher_output = self.opd_teacher_wg.compute_rm_score(batch)

        if "token_level_scores" in batch.batch.keys():
            batch.batch["true_token_level_scores"] = batch.batch["token_level_scores"]

        response_mask = batch.batch["response_mask"]
        rm_scores = teacher_output.batch["rm_scores"].to(response_mask.device)
        batch.batch["token_level_scores"] = rm_scores
        batch.batch["teacher_log_probs"] = teacher_output.batch["teacher_log_probs"].to(response_mask.device)

        n_tokens = max(int(response_mask.sum().item()), 1)
        # rm_scores may be (B, T) or (B, T, K) when top-K is on; broadcast the mask.
        mask = response_mask if rm_scores.dim() == 2 else response_mask.unsqueeze(-1)
        metrics["opd/token_reward_mean"] = float((rm_scores * mask).sum().item()) / n_tokens
        metrics["opd/teacher_logp_mean"] = (
            float((batch.batch["teacher_log_probs"] * response_mask).sum().item()) / n_tokens
        )

        teachers = teacher_output.non_tensor_batch.get("mopd_teacher")
        if teachers is not None:
            counts = dict(sorted(Counter(str(t) for t in teachers).items()))
            # The running MOPD job greps for exactly this line.
            print(f"[MOPD] batch routing: {counts}", flush=True)
            logger.info("[MOPD] batch routing: %s", counts)
            for name, count in counts.items():
                metrics[f"mopd/routed/{name}"] = count

        # Worker scalars live under meta_info["metrics"] so DataProto.concat
        # (which special-cases that key) can aggregate across DP ranks.
        worker_metrics = teacher_output.meta_info.get("metrics") or {}
        overlap = _mean_metric(worker_metrics.get("top_k_overlap_ratio"))
        if overlap is not None:
            metrics["opd/top_k_overlap_ratio"] = float(overlap)
            metrics["opd/top_k_mode"] = 1.0 if self.opd_top_k_is_reward() else 0.0
        for key in ("top_k_student_mass", "top_k_teacher_mass", "top_k_reward_mean"):
            value = _mean_metric(worker_metrics.get(key))
            if value is not None:
                metrics[f"opd/{key}"] = float(value)

        return batch

    # ------------------------------------------------------------------
    # template mode
    # ------------------------------------------------------------------
    def log_template_metrics(self, batch: DataProto, metrics: dict[str, Any]) -> None:
        """Report how many self-generated queries would pass the content filter.

        This is monitoring only -- rows are not dropped here, because dropping
        rows mid-step would desynchronise the DP-balanced mini-batching. Set
        ``template.oversample_ratio`` above 1.0 to trade compute for quality
        instead.
        """
        from verl.utils.template_prompt import decode_responses, filter_valid_texts

        template_config = self.config.get("template", {})
        try:
            texts = decode_responses(
                self.tokenizer,
                batch.batch["responses"],
                batch.batch["response_mask"],
            )
            result = filter_valid_texts(
                texts,
                min_char_length=template_config.get("min_char_length", 20),
                task=template_config.get("task", "math"),
            )
        except Exception as e:  # monitoring must never break training
            logger.warning("[Template] filter monitoring failed: %s", e)
            return

        metrics["template/valid_ratio"] = result.valid_ratio
        metrics["template/valid_count"] = result.valid_count
        metrics["template/rejected_count"] = result.rejected_count
        for reason, count in result.reject_reasons.items():
            metrics[f"template/reject_{reason}"] = count

        print(
            f"[Template] step {self.global_steps}: valid={result.valid_count}/{result.total_count} "
            f"({result.valid_ratio:.1%}) rejects={result.reject_reasons}",
            flush=True,
        )

    @staticmethod
    def _masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> float:
        return (tensor * mask).sum().item() / mask.sum().clamp(min=1).item()
