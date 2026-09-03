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
"""Agent loop for template-prefix OPD: the student writes its own query.

Unlike :class:`~verl.experimental.agent_loop.single_turn_agent_loop.SingleTurnAgentLoop`,
this loop never calls ``apply_chat_template``. The prompt is a bare template
prefix (see :mod:`verl.utils.template_prompt`) that stops in the middle of a user
turn, so the model continues by writing the user's message itself and then
answering it. The whole continuation is the response, and the OPD teacher scores
it token by token like any other rollout.

Selected per-row via the ``agent_name`` column, which
:class:`~verl.utils.dataset.template_dataset.TemplatePromptDataset` sets to
``template_agent``. Validation rows come from a normal dataset and keep
``single_turn_agent``, so evaluation still uses real prompts.
"""

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.template_prompt import DEFAULT_TEMPLATE, build_prefix_ids
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("template_agent")
class TemplatePrefixAgentLoop(AgentLoopBase):
    """Single-turn loop seeded with a raw template prefix instead of a chat prompt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        template_config = self.config.get("template", {})
        prefix_mode = str(template_config.get("prefix_mode", "template")).lower()
        system_prompt = template_config.get("system_prompt", None) or None

        # The prefix is identical for every row, so build it once per worker.
        self.prefix_ids = build_prefix_ids(
            tokenizer=self.tokenizer,
            prefix_template=template_config.get("prefix_template", DEFAULT_TEMPLATE),
            prefix_suffix=template_config.get("prefix_suffix", ""),
            prefix_mode=prefix_mode,
            system_prompt=system_prompt,
            max_prompt_length=self.prompt_length,
        )
        logger.info(
            "[Template] prefix_mode=%s, %d prefix tokens: %r",
            prefix_mode,
            len(self.prefix_ids),
            self.tokenizer.decode(self.prefix_ids, skip_special_tokens=False),
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        priority = int(priority)
        prompt_ids = list(self.prefix_ids)

        # Same as single_turn: validation may carry its own budget in
        # sampling_params["max_tokens"], and the server pops that key.
        response_length = int(sampling_params.get("max_tokens") or self.response_length)

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            request_id = f"det-{priority}" if getattr(self.rollout_config, "full_determinism", False) else uuid4().hex
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                priority=priority,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        response_ids = output.token_ids
        agent_output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:response_length],
            response_mask=[1] * len(response_ids[:response_length]),
            response_logprobs=output.log_probs[:response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data={},
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        # Keep the schema consistent with the other agent loops.
        agent_output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return agent_output
