"""Answer extraction for MedDistractQA model responses.

Sixteen regular expressions are tried in a fixed order; the first pattern that matches wins and, within a pattern,
``re.search`` returns the FIRST occurrence (pattern 9 is the exception: it takes the LAST bracketed capital letter).
For compatibility, responses matched by that legacy cascade keep exactly the same result. If the cascade returns
invalid, a narrow recovery pass removes byte-level-tokenizer boundary glyphs emitted literally by some vLLM/model
combinations and accepts an explicit final-answer phrase with punctuation such as ``final answer is: C``. Only the
letters A-D are recognised. The result is ``"(X)"`` or ``"[invalid]"``; scoring is strict string equality of the
stripped letter with the key, so ``[invalid]`` (and a lower-case letter) counts as wrong.
"""
from __future__ import annotations

import re

INVALID = "[invalid]"

_CASCADE = [
    (re.compile(r"the answer is ([A-D])"), "first"),
    (re.compile(r"answer is ([A-D])\.?", re.IGNORECASE), "first"),
    (re.compile(r"Therefore, the final model answer is ([A-D])", re.IGNORECASE), "first"),
    (re.compile(r"boxed\{([A-D])\}"), "first"),
    (re.compile(r"The answer is \*\*\[([A-D])\]"), "first"),
    (re.compile(r"Answer: ([A-D])", re.IGNORECASE), "first"),
    (re.compile(r"Answer:\*\* ([A-D])", re.IGNORECASE), "first"),
    (re.compile(r"is ([A-D])"), "first"),
    (re.compile(r"[\(\[]([A-Z])[\)\]]"), "last"),
    (re.compile(r"\*\*([A-D])\*\*"), "first"),
    (re.compile(r"\[([A-D])\]"), "first"),
    (re.compile(r"Choice ([A-D])", re.IGNORECASE), "first"),
    (re.compile(r"\[\*\*([A-D])\]", re.IGNORECASE), "first"),
    (re.compile(r"\[([A-D])\*\*\]", re.IGNORECASE), "first"),
    (re.compile(r"My Answer\n\*\*([A-D])", re.IGNORECASE), "first"),
    (re.compile(r"\n\n\*\*Answer:\*\*\n([A-D])", re.IGNORECASE), "first"),
]

_RECOVERY_FINAL_ANSWER = re.compile(
    r"(?:therefore[, :]*\s*)?(?:the\s+)?(?:final\s+)(?:model\s+)?"
    r"answer\s*(?:is)?\s*[:=\-]?\s*[\"'`]*[\[(]*\**\s*([A-D])(?:\b|\**)",
    re.IGNORECASE,
)


def _cascade(response: str) -> str:
    """Apply the frozen legacy cascade to one response."""
    for regex, which in _CASCADE:
        if which == "first":
            match = regex.search(response)
            if match:
                return f"({match.group(1)})"
        else:
            found = regex.findall(response)
            if found:
                return f"({found[-1]})"
    return INVALID


def extract_answer(response: str) -> str:
    """Return ``"(X)"`` for an extracted answer, else ``"[invalid]"``."""
    if response is None:
        return INVALID
    parsed = _cascade(response)
    if parsed != INVALID:
        return parsed

    # GPT-2-style byte-level token boundary characters occasionally appeared
    # literally in vLLM output text (e.g. ``theĠfinalĠanswer``). This transform
    # only joins those boundaries back into ordinary whitespace.
    normalized = response.replace("Ġ", " ").replace("Ċ", "\n")
    parsed = _cascade(normalized)
    if parsed != INVALID:
        return parsed

    # The requested format is also commonly rendered with a colon after "is".
    # Use the last explicit final-answer phrase and do not infer from bare letters.
    matches = list(_RECOVERY_FINAL_ANSWER.finditer(normalized))
    return f"({matches[-1].group(1).upper()})" if matches else INVALID


def is_correct(parsed: str, correct_answer: str) -> bool:
    """Strict scoring used by the original runner: ``parsed.strip("()") == correct_answer``."""
    return str(parsed).strip("()") == str(correct_answer)


def is_invalid(parsed: str) -> bool:
    return str(parsed) == INVALID
