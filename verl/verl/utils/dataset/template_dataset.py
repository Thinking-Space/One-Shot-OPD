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
"""Driver dataset for template-prefix OPD.

Template mode has no prompt corpus: the student writes its own query from a bare
prefix. This dataset therefore emits *placeholder* rows whose only job is to
drive the dataloader and to tag each row with ``agent_name="template_agent"``, so
:class:`~verl.experimental.agent_loop.template_agent_loop.TemplatePrefixAgentLoop`
picks it up and ignores ``raw_prompt`` entirely.

``AgentLoopManager.generate_sequences`` only fills in a default ``agent_name``
when the column is absent, so setting it here selects the loop per row without
patching anything upstream. Validation uses a normal
:class:`~verl.utils.dataset.RLHFDataset`, whose rows carry no ``agent_name`` and
so fall back to ``single_turn_agent`` -- evaluation keeps using real prompts.

Select it from a launch script with::

    data.custom_cls.path=verl/utils/dataset/template_dataset.py
    data.custom_cls.name=TemplatePromptDataset
"""

import logging
import os
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["TemplatePromptDataset"]

#: Agent loop that consumes these rows.
TEMPLATE_AGENT_NAME = "template_agent"


class TemplatePromptDataset(Dataset):
    """Placeholder rows that route the rollout through the template agent loop.

    The signature matches :class:`~verl.utils.dataset.RLHFDataset` so
    ``data.custom_cls`` can swap one for the other.

    Args:
        data_files: ignored; template mode reads no corpus. Accepted so the
            dataset is a drop-in replacement.
        tokenizer: HuggingFace tokenizer (unused here; the agent loop builds the
            prefix itself, once per rollout worker rather than once per row).
        config: the ``data`` config node. The top-level ``template`` node is
            resolved off the config root.
        processor: ignored; template mode is text-only.
    """

    def __init__(
        self,
        data_files=None,
        tokenizer=None,
        config: Optional[DictConfig] = None,
        processor=None,
        max_samples: int = -1,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.config = config

        template_config = self._resolve_template_config(config)
        self.length = int(template_config.get("dataset_length", 1000000))
        if max_samples is not None and int(max_samples) > 0:
            self.length = min(self.length, int(max_samples))
        self.data_source = str(template_config.get("data_source", "template"))
        self.task = str(template_config.get("task", "math"))

        logger.info(
            "[Template] driver dataset: %d placeholder rows, data_source=%s, task=%s",
            self.length,
            self.data_source,
            self.task,
        )

    @staticmethod
    def _resolve_template_config(config) -> dict:
        """Reach the top-level ``template`` node from the ``data`` node.

        ``data.custom_cls`` hands the dataset only its own subtree, but the
        template settings live at the config root. OmegaConf lets a child node
        walk back up to its root, so resolve from there; fall back to defaults
        when the dataset is constructed standalone (e.g. in unit tests).
        """
        from omegaconf import OmegaConf

        if config is None:
            return {}
        try:
            root = config._get_root()
            template = OmegaConf.select(root, "template")
            if template is not None:
                return OmegaConf.to_container(template, resolve=True)
        except Exception as e:
            logger.warning("[Template] could not resolve the `template` config node (%s); using defaults", e)
        return {}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: int) -> dict:
        return {
            # Selects TemplatePrefixAgentLoop, which ignores raw_prompt and
            # seeds generation with the template prefix instead.
            "agent_name": TEMPLATE_AGENT_NAME,
            # Never tokenized in template mode, but several call sites index it.
            "raw_prompt": [{"role": "user", "content": ""}],
            "data_source": self.data_source,
            "ability": self.task,
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"index": item, "split": "train"},
            "index": item,
            "tools_kwargs": {},
            "interaction_kwargs": {},
            # DataProto.batch must not be empty; mirrors RLHFDataset.
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }
