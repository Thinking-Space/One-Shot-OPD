# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Advantage estimators and policy losses for On-Policy Distillation (OPD).

OPD replaces the usual scalar outcome reward with a *dense, token-level* reward
derived from a teacher model's log-probabilities over the student's own rollout:

    r_t = -(log p_student(y_t) - log p_teacher(y_t))       (reverse KL, k1)

Because that reward is already per-token and already on the right scale, the
advantage is simply the reward itself (masked) -- there is no baseline to
subtract and no discounting to apply. That is ``token_reward_direct``.

``token_reward_direct_plus_grpo`` additionally mixes in a conventional GRPO
outcome advantage computed from the *verifier* score, so the student is pulled
both toward the teacher's distribution and toward actually solving the task.

This module is imported at the bottom of ``verl.trainer.ppo.core_algos`` purely
so that the ``@register_adv_est`` / ``@register_policy_loss`` decorators run.
"""

from typing import Any, Optional

import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_grpo_outcome_advantage,
    register_adv_est,
    register_policy_loss,
)
from verl.workers.config import ActorConfig

__all__ = [
    "compute_token_reward_direct_advantage",
    "compute_token_reward_direct_plus_grpo_advantage",
    "compute_policy_loss_vanilla_topk",
]


@register_adv_est("token_reward_direct")
def compute_token_reward_direct_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the token-level teacher reward directly as the advantage.

    Args:
        token_level_rewards: ``(bs, response_length)`` for scalar-per-token OPD,
            or ``(bs, response_length, K)`` when top-K distillation is enabled.
        response_mask: ``(bs, response_length)``.
        config: unused; accepted to match the estimator calling convention.

    Returns:
        ``(advantages, returns)``, both the same shape as ``token_level_rewards``.
    """
    with torch.no_grad():
        mask = response_mask
        if token_level_rewards.dim() == 3:
            # top-K rewards are (bs, T, K); broadcast the (bs, T) mask over K.
            mask = response_mask.unsqueeze(-1)
        advantages = token_level_rewards * mask
        returns = advantages.clone()
    return advantages, returns


@register_adv_est("token_reward_direct_plus_grpo")
def compute_token_reward_direct_plus_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense teacher advantage plus a GRPO outcome advantage from the verifier.

    The GRPO term is computed from ``true_token_level_scores`` when the trainer
    supplies it (the rule-based verifier reward, which OPD otherwise displaces
    from ``token_level_rewards``). If absent, it falls back to
    ``token_level_rewards`` so the estimator still runs standalone.

    ``algorithm.grpo_outcome_weight`` scales the GRPO term (default 1.0).
    """
    direct_adv, _ = compute_token_reward_direct_advantage(
        token_level_rewards, response_mask, config, **kwargs
    )

    rewards_for_grpo = kwargs.get("true_token_level_scores")
    if rewards_for_grpo is None:
        rewards_for_grpo = token_level_rewards
    if rewards_for_grpo.dim() == 3:
        # GRPO needs a scalar outcome per token position; collapse the K axis.
        rewards_for_grpo = rewards_for_grpo.sum(dim=-1)

    norm_adv_by_std_in_grpo = True
    weight = 1.0
    if config is not None:
        norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
        weight = config.get("grpo_outcome_weight", 1.0)

    grpo_adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards_for_grpo,
        response_mask=response_mask,
        index=index,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    if direct_adv.dim() == 3:
        grpo_adv = grpo_adv.unsqueeze(-1)

    combined = direct_adv + weight * grpo_adv
    return combined, combined.clone()


@register_policy_loss("vanilla_topk")
def compute_policy_loss_vanilla_topk(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """PPO clipped objective over top-K candidate tokens per position.

    With top-K distillation the log-probs and advantages carry an extra trailing
    axis of size K: ``(bs, T, K)``. The importance ratio and clipping are applied
    per (position, candidate), then summed over K to yield a per-position loss,
    which is aggregated exactly as the scalar case.

    Falls back to upstream's ``vanilla`` loss when the tensors are 2-D, so the
    same ``policy_loss.loss_mode`` works whether or not top-K is switched on.
    """
    assert config is not None

    if log_prob.dim() != 3 or old_log_prob.dim() != 3:
        from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

        return compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )

    clip_ratio = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    mask_3d = response_mask.unsqueeze(-1)

    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights.unsqueeze(-1)

    # Sum over the K axis -> (bs, T), then aggregate with the standard helper.
    # global_batch_info is forwarded exactly as the vanilla loss does, so the
    # normalization (and hence the loss scale) is identical across the two modes.
    pg_loss = agg_loss(
        loss_mat=torch.sum(pg_losses, dim=-1),
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    denom = mask_3d.sum().clamp(min=1)
    pg_clipfrac = (torch.gt(pg_losses2, pg_losses1).float() * mask_3d).sum() / denom
    ppo_kl = (-negative_approx_kl * mask_3d).sum() / denom

    # Keys carry the same "actor/" prefix the vanilla loss uses; ppo_loss passes
    # them through untouched, so dropping it would rename the metrics.
    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }
