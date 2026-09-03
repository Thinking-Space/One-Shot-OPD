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
"""Template-prefix prompting for input-side On-Policy Distillation.

In the normal OPD setting the student answers a prompt taken from the dataset.
In *template* mode there is no dataset prompt: the rollout is seeded with a bare
template prefix such as ``{bos}{user}<think>\\n`` and the student writes its own
query and then answers it. The teacher then scores that self-generated rollout
exactly as in ordinary OPD, so the two features compose orthogonally.

The defining property of template mode is that the prefix is **not** produced by
``tokenizer.apply_chat_template``. A chat template would emit a complete,
well-formed user turn; here we deliberately stop mid-turn so the model has to
continue writing the user's message itself. Two ways to obtain that prefix are
supported:

``prefix_mode="template"``
    Resolve placeholders in a template string against the tokenizer's special
    tokens and encode the result directly. Placeholders:

    ==================  ==========================================
    ``{bos}``           ``tokenizer.bos_token``
    ``{eos}``           ``tokenizer.eos_token``
    ``{user}``          user role marker, e.g. ``<|User|>``
    ``{assistant}``     assistant role marker
    ``{system_start}``  system message opening marker
    ``{system_end}``    system message closing marker
    ==================  ==========================================

    Examples::

        "{bos}{user}<think>\\n"
        "{bos}{user}</think>\\n\\n</think>\\n\\n"
        "{bos}{system_start}You are a math teacher.{system_end}{user}"

``prefix_mode="raw"``
    Render a chat template around a unique marker and keep only the text *before*
    the marker. This yields the model's native "start of a user turn" prefix
    (e.g. ``<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n`` for Qwen2)
    without hardcoding any model-specific token. The chat template is used only
    to *discover* the prefix; the prefix itself is still fed to the model raw.

Everything in this module is pure Python/torch and runs on CPU, so the prefix
construction and the response filters are unit-testable without a GPU.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "DEFAULT_TEMPLATE",
    "FilterResult",
    "build_prefix_ids",
    "extract_chat_tokens",
    "filter_valid_texts",
    "get_marker_prefix_ids",
    "is_valid_code_content",
    "is_valid_math_content",
    "resolve_template",
    "truncate_after_oversample",
]

#: Default prefix: DeepSeek-R1 style "user turn opens, thinking starts".
DEFAULT_TEMPLATE = "{bos}{user}<think>\n"

#: Fallback role markers for tokenizers that expose no chat template.
_FALLBACK_USER_TOKEN = "<｜User｜>"
_FALLBACK_ASSISTANT_TOKEN = "<｜Assistant｜>"

_USER_SENTINEL = "TEMPLATE_SENTINEL_USER_XYZ"
_SYSTEM_SENTINEL = "TEMPLATE_SENTINEL_SYSTEM_XYZ"
_SPLIT_MARKER = "<<<TEMPLATE_SPLIT_MARKER>>>"


# ---------------------------------------------------------------------------
# Prefix construction
# ---------------------------------------------------------------------------


def extract_chat_tokens(tokenizer) -> dict[str, str]:
    """Recover each role's special-token text from a tokenizer's chat template.

    Renders dummy messages containing unique sentinels through
    ``apply_chat_template`` and reads the surrounding markup back out, so the
    caller never has to hardcode model-specific tokens. Falls back to the
    DeepSeek-R1 markers when the tokenizer has no chat template.

    Returns:
        A dict with keys ``bos``, ``eos``, ``user``, ``assistant``,
        ``system_start`` and ``system_end``. Missing pieces are empty strings.
    """
    tokens = {
        "bos": getattr(tokenizer, "bos_token", "") or "",
        "eos": getattr(tokenizer, "eos_token", "") or "",
        "user": "",
        "assistant": "",
        "system_start": "",
        "system_end": "",
    }

    if not getattr(tokenizer, "chat_template", None):
        tokens["user"] = _FALLBACK_USER_TOKEN
        tokens["assistant"] = _FALLBACK_ASSISTANT_TOKEN
        return tokens

    try:
        user_result = tokenizer.apply_chat_template(
            [{"role": "user", "content": _USER_SENTINEL}],
            tokenize=False,
            add_generation_prompt=True,
        )
        idx = user_result.index(_USER_SENTINEL)
        before_user_content = user_result[:idx]
        after_user_content = user_result[idx + len(_USER_SENTINEL) :]

        bos = tokens["bos"]
        if bos and before_user_content.startswith(bos):
            tokens["user"] = before_user_content[len(bos) :]
        else:
            tokens["user"] = before_user_content

        # Whatever follows the user content is the generation prompt.
        tokens["assistant"] = after_user_content
    except Exception:
        tokens["user"] = _FALLBACK_USER_TOKEN
        tokens["assistant"] = _FALLBACK_ASSISTANT_TOKEN

    try:
        sys_result = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _SYSTEM_SENTINEL},
                {"role": "user", "content": _USER_SENTINEL},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        sys_idx = sys_result.index(_SYSTEM_SENTINEL)
        user_idx = sys_result.index(_USER_SENTINEL)

        bos = tokens["bos"]
        sys_before = sys_result[:sys_idx]
        if bos and sys_before.startswith(bos):
            tokens["system_start"] = sys_before[len(bos) :]
        else:
            tokens["system_start"] = sys_before

        # Everything between the system content and the user content, minus the
        # user marker itself, closes the system message.
        between = sys_result[sys_idx + len(_SYSTEM_SENTINEL) : user_idx]
        user_tok = tokens["user"]
        if user_tok and between.endswith(user_tok):
            tokens["system_end"] = between[: -len(user_tok)]
        else:
            tokens["system_end"] = between
    except Exception:
        # Models without a system role simply keep the empty defaults.
        pass

    return tokens


def resolve_template(template: str, tokenizer) -> str:
    """Substitute ``{bos}`` / ``{user}`` / ... placeholders with real tokens."""
    chat_tokens = extract_chat_tokens(tokenizer)
    resolved = template
    for key, val in chat_tokens.items():
        resolved = resolved.replace(f"{{{key}}}", val)
    return resolved


def get_marker_prefix_ids(tokenizer, system_prompt: Optional[str] = None) -> list[int]:
    """Prefix token ids for ``prefix_mode="raw"``.

    Renders a chat template around a unique marker and keeps only the text before
    it, which is exactly the model's own "a user turn starts here" preamble.
    """
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": _SPLIT_MARKER})

    full_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    prefix_text = full_text.split(_SPLIT_MARKER)[0]
    return tokenizer(prefix_text, add_special_tokens=False)["input_ids"]


def build_prefix_ids(
    tokenizer,
    prefix_template: str = DEFAULT_TEMPLATE,
    prefix_suffix: str = "",
    prefix_mode: str = "template",
    system_prompt: Optional[str] = None,
    max_prompt_length: Optional[int] = None,
) -> list[int]:
    """Build the rollout prefix token ids for template mode.

    Note that ``add_special_tokens=False`` is used throughout: the template
    already spells out whichever BOS/role markers are wanted, and letting the
    tokenizer add its own would duplicate them.

    Args:
        tokenizer: HuggingFace tokenizer.
        prefix_template: template string with ``{bos}``-style placeholders. Used
            when ``prefix_mode="template"``.
        prefix_suffix: extra text appended after the prefix to steer generation.
        prefix_mode: ``"template"`` or ``"raw"``.
        system_prompt: system prompt, only meaningful for ``prefix_mode="raw"``.
        max_prompt_length: left-truncate the prefix to this many tokens.

    Returns:
        The prefix token ids.
    """
    if prefix_mode == "template":
        prefix_text = resolve_template(prefix_template, tokenizer) + prefix_suffix
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    elif prefix_mode == "raw":
        prefix_ids = get_marker_prefix_ids(tokenizer, system_prompt)
        if prefix_suffix:
            prefix_ids = prefix_ids + tokenizer.encode(prefix_suffix, add_special_tokens=False)
    else:
        raise ValueError(f"[Template] unsupported prefix_mode={prefix_mode!r}; expected 'template' or 'raw'")

    if max_prompt_length is not None and len(prefix_ids) > max_prompt_length:
        prefix_ids = prefix_ids[-max_prompt_length:]
    return prefix_ids


# ---------------------------------------------------------------------------
# Content filters for the self-generated queries
# ---------------------------------------------------------------------------

_MATH_KEYWORDS: list[str] = [
    "solve", "find", "compute", "prove", "determine", "evaluate",
    "calculate", "simplify", "show that", "how many", "what is the value",
    "what is", "let ", "given that", "suppose", "for all", "there exist",
    "integer", "positive", "triangle", "circle", "polynomial", "equation",
    "divisible", "modulo", "remainder", "sum of", "product of",
    "maximum", "minimum", "probability", "permutation", "combination",
    "求", "证明", "计算", "设", "解方程",
    "多少", "若", "已知",
]  # fmt: skip

_MATH_SYMBOL_RE = re.compile(r"[+\-=^_{}\\]|\d+|\\frac|\\sqrt|\\sum|\\int|\\lim|\$")

_CODE_KEYWORDS: list[str] = [
    "def ", "class ", "import ", "return ", "print(", "for ", "while ",
    "if ", "else:", "elif ", "try:", "except", "with ", "assert ",
    "input(", "sys.stdin", "stdout", "```", "solution",
    "function", "algorithm", "implement", "output", "test case",
]  # fmt: skip

_CODE_SYMBOL_RE = re.compile(r"[(){}\[\]:;]|==|!=|<=|>=|->|=>|\brange\b|\blen\b|\bint\b|\bstr\b|\blist\b")

#: Phrases that mark a reply *about* the conversation rather than a real query.
_META_PATTERNS: list[str] = [
    "seems like", "clarify", "incomplete", "provide more", "more details",
    "i'm an ai", "i am an ai", "deepseek", "language model", "语言模型",
    "how can i help", "what would you like", "feel free to",
    "i'd be happy to help", "could you clarify", "your message",
    "your question", "i'm here to help", "qwen",
]  # fmt: skip

#: Phrases that mark the model introducing itself instead of posing a task.
_SELF_INTRO_KEYWORDS: list[str] = [
    "deepseek", "语言模型", "智能助手", "qwen", "alibaba",
]  # fmt: skip


@dataclass
class FilterResult:
    """Outcome of filtering one batch of self-generated texts.

    Attributes:
        total_count: number of texts examined.
        valid_count: number that passed.
        rejected_count: number that were dropped.
        reject_reasons: reason -> count, e.g. ``{"too_short": 5}``.
        valid_indices: indices of the passing texts, in original order.
    """

    total_count: int = 0
    valid_count: int = 0
    rejected_count: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    valid_indices: list[int] = field(default_factory=list)

    @property
    def valid_ratio(self) -> float:
        return self.valid_count / max(self.total_count, 1)


def is_valid_math_content(text: str, min_char_length: int = 20) -> tuple[bool, str]:
    """Whether ``text`` looks like a genuine math query.

    Returns ``(is_valid, reject_reason)``; ``reject_reason`` is empty when valid.
    """
    if len(text) < min_char_length:
        return False, "too_short"

    lowered = text.lower()
    if any(p in lowered for p in _META_PATTERNS):
        return False, "meta_response"
    if any(k in lowered for k in _SELF_INTRO_KEYWORDS):
        return False, "self_intro"

    has_keyword = any(k in lowered for k in _MATH_KEYWORDS)
    has_symbol = bool(_MATH_SYMBOL_RE.search(text))
    has_digit = bool(re.search(r"\d", text))
    if not ((has_keyword or has_symbol) and has_digit):
        return False, "no_math_content"

    return True, ""


def is_valid_code_content(text: str, min_char_length: int = 20) -> tuple[bool, str]:
    """Whether ``text`` looks like a genuine coding query. See :func:`is_valid_math_content`."""
    if len(text) < min_char_length:
        return False, "too_short"

    lowered = text.lower()
    if any(p in lowered for p in _META_PATTERNS):
        return False, "meta_response"
    if any(k in lowered for k in _SELF_INTRO_KEYWORDS):
        return False, "self_intro"

    has_keyword = any(k in lowered for k in _CODE_KEYWORDS)
    has_symbol = bool(_CODE_SYMBOL_RE.search(text))
    if not (has_keyword or has_symbol):
        return False, "no_code_content"

    return True, ""


def get_content_checker(task: str):
    """Pick the content checker for a task (``"code"`` or anything else -> math)."""
    return is_valid_code_content if str(task).lower() == "code" else is_valid_math_content


def filter_valid_texts(texts: list[str], min_char_length: int = 20, task: str = "math") -> FilterResult:
    """Filter decoded rollout texts, dropping empty / too-short / meta responses.

    Pure string processing so it can be unit-tested with mock data.
    """
    checker = get_content_checker(task)

    valid_indices: list[int] = []
    reject_reasons: dict[str, int] = {}

    for i, text in enumerate(texts):
        stripped = (text or "").strip()
        if not stripped:
            reject_reasons["empty_response"] = reject_reasons.get("empty_response", 0) + 1
            continue
        is_valid, reason = checker(stripped, min_char_length)
        if is_valid:
            valid_indices.append(i)
        else:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    return FilterResult(
        total_count=len(texts),
        valid_count=len(valid_indices),
        rejected_count=len(texts) - len(valid_indices),
        reject_reasons=reject_reasons,
        valid_indices=valid_indices,
    )


def truncate_after_oversample(valid_indices: list[int], target_batch_size: int) -> list[int]:
    """Cut an oversampled batch back down to ``target_batch_size`` rows.

    Template mode generates ``ceil(target * oversample_ratio)`` rollouts so that
    enough survive filtering. This keeps the first ``target_batch_size`` valid
    rows; if too few survived, the remainder is filled by cycling through the
    valid rows (duplicating them) so the batch shape stays fixed. Returns an
    empty list when nothing at all survived -- the caller decides what to do.
    """
    if target_batch_size <= 0 or not valid_indices:
        return []
    if len(valid_indices) >= target_batch_size:
        return list(valid_indices[:target_batch_size])

    filled = list(valid_indices)
    while len(filled) < target_batch_size:
        filled.append(valid_indices[len(filled) % len(valid_indices)])
    return filled


def oversampled_batch_size(target_batch_size: int, oversample_ratio: float) -> int:
    """How many rollouts to request so ``target_batch_size`` survive filtering."""
    import math

    if oversample_ratio <= 1.0:
        return target_batch_size
    return int(math.ceil(target_batch_size * oversample_ratio))


def decode_responses(tokenizer, responses, response_mask) -> list[str]:
    """Decode a ``(bs, T)`` response tensor into stripped text, honouring the mask.

    Trailing EOS tokens are removed before decoding so they do not count toward
    the length filters.
    """
    eos_id = tokenizer.eos_token_id
    texts: list[str] = []
    for i in range(responses.shape[0]):
        valid_len = int(response_mask[i].sum().item())
        ids = responses[i, :valid_len].tolist()
        while ids and ids[-1] == eos_id:
            ids.pop()
        texts.append(tokenizer.decode(ids, skip_special_tokens=True).strip() if ids else "")
    return texts
