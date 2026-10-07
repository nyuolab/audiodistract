"""The MedDistractQA engines: result-directory name, display name, group, provenance.

``CANONICAL_ENGINES`` are the engines of the preprint minus Claude 3.5 Haiku and o1-mini, which were dropped in
September 2026 because the providers retired them before their clean runs could be repeated at a uniform token budget;
the endpoint-only Llama-3.3-70B run is also excluded because its exact inference configuration cannot be reproduced
(32 engines: 18 open-weight general, 3 open-weight medical, 11 proprietary); ``ADDED_ENGINES`` are frontier models
evaluated later with the same protocol (``run_eval``); ``ENGINES``
is their concatenation. Group rule (single source of truth, identical to the preprint figures): an engine name
containing gpt/o1/o3/o4/claude/gemini is Proprietary; med/meerkat/ultramedical is Open-weight, medical; anything else
is Open-weight, general. The lists are checked against the rule at import time.
"""
from __future__ import annotations

import pandas as pd

GROUP_GENERAL = "Open-weight, general"
GROUP_MEDICAL = "Open-weight, medical"
GROUP_PROPRIETARY = "Proprietary"
GROUPS = (GROUP_GENERAL, GROUP_MEDICAL, GROUP_PROPRIETARY)

# engine (results directory), display name, provider, open weights, params (B), model id passed to the backend
CANONICAL_ENGINES = [
    ("claude-3-7-sonnet-20250219", "Claude 3.7 Sonnet", "Anthropic", False, None, "claude-3-7-sonnet-20250219"),
    ("claude-haiku-4-5", "Claude Haiku 4.5", "Anthropic", False, None, "claude-haiku-4-5"),
    ("claude-opus-4-6", "Claude Opus 4.6", "Anthropic", False, None, "claude-opus-4-6"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6", "Anthropic", False, None, "claude-sonnet-4-6"),
    ("gemini-3.1-pro-preview", "Gemini 3.1 Pro (preview)", "Google", False, None, "gemini-3.1-pro-preview"),
    ("gpt-4o", "GPT-4o", "OpenAI", False, None, "gpt-4o"),
    ("gpt-4o-mini", "GPT-4o mini", "OpenAI", False, None, "gpt-4o-mini"),
    ("gpt-5.4", "GPT-5.4", "OpenAI", False, None, "gpt-5.4"),
    ("o3-mini", "o3-mini (medium)", "OpenAI", False, None, "o3-mini"),
    ("o3-mini-high", "o3-mini (high)", "OpenAI", False, None, "o3-mini-high"),
    ("o3-mini-low", "o3-mini (low)", "OpenAI", False, None, "o3-mini-low"),
    ("Llama-3-8B-UltraMedical", "Llama-3-8B-UltraMedical", "TsinghuaC3I", True, 8.0, "TsinghuaC3I/Llama-3-8B-UltraMedical"),
    ("MedMobile", "MedMobile-3.8B", "NYU (Phi-3-mini fine-tune)", True, 3.8, "KrithikV/MedMobile"),
    ("llama-3-meerkat-8b-v1.0", "Llama-3-Meerkat-8B", "DMIS Lab", True, 8.0, "dmis-lab/llama-3-meerkat-8b-v1.0"),
    ("DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Llama-8B", "DeepSeek", True, 8.0, "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    ("Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct", "Meta", True, 8.0, "meta-llama/Llama-3.1-8B-Instruct"),
    ("Llama-3.2-1B-Instruct", "Llama-3.2-1B-Instruct", "Meta", True, 1.2, "meta-llama/Llama-3.2-1B-Instruct"),
    ("Llama-3.2-3B-Instruct", "Llama-3.2-3B-Instruct", "Meta", True, 3.2, "meta-llama/Llama-3.2-3B-Instruct"),
    ("Meta-Llama-3-8B-Instruct", "Llama-3-8B-Instruct", "Meta", True, 8.0, "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("Ministral-8B-Instruct-2410", "Ministral-8B-Instruct", "Mistral AI", True, 8.0, "mistralai/Ministral-8B-Instruct-2410"),
    ("Mistral-7B-Instruct-v0.1", "Mistral-7B-Instruct-v0.1", "Mistral AI", True, 7.2, "mistralai/Mistral-7B-Instruct-v0.1"),
    ("Mistral-7B-Instruct-v0.2", "Mistral-7B-Instruct-v0.2", "Mistral AI", True, 7.2, "mistralai/Mistral-7B-Instruct-v0.2"),
    ("Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3", "Mistral AI", True, 7.2, "mistralai/Mistral-7B-Instruct-v0.3"),
    ("Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct", "Alibaba", True, 0.5, "Qwen/Qwen2.5-0.5B-Instruct"),
    ("Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct", "Alibaba", True, 1.5, "Qwen/Qwen2.5-1.5B-Instruct"),
    ("Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct", "Alibaba", True, 3.1, "Qwen/Qwen2.5-3B-Instruct"),
    ("Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", "Alibaba", True, 7.6, "Qwen/Qwen2.5-7B-Instruct"),
    ("Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-3B-Instruct", "Alibaba", True, 3.8, "Qwen/Qwen2.5-VL-3B-Instruct"),
    ("gemma-2-2b-it", "Gemma-2-2B-it", "Google", True, 2.6, "google/gemma-2-2b-it"),
    ("gemma-2-9b-it", "Gemma-2-9B-it", "Google", True, 9.2, "google/gemma-2-9b-it"),
    ("gemma-3-1b-it", "Gemma-3-1B-it", "Google", True, 1.0, "google/gemma-3-1b-it"),
    ("gemma-3-4b-it", "Gemma-3-4B-it", "Google", True, 4.3, "google/gemma-3-4b-it"),
]
# frontier models added after the preprint (September 2026 runs of run_eval; see docs/qa.md for their settings)
ADDED_ENGINES = [
    ("gpt-5.4-default", "GPT-5.4 (default effort)", "OpenAI", False, None, "gpt-5.4-2026-03-05"),
    ("gpt-5.6-sol", "GPT-5.6 Sol", "OpenAI", False, None, "gpt-5.6-sol"),
    ("gpt-5.6-luna", "GPT-5.6 Luna", "OpenAI", False, None, "gpt-5.6-luna"),
    ("gpt-5.6-terra", "GPT-5.6 Terra", "OpenAI", False, None, "gpt-5.6-terra"),
    ("claude-opus-5", "Claude Opus 5", "Anthropic", False, None, "claude-opus-5"),
    ("claude-fable-5-1", "Claude Fable 5.1", "Anthropic", False, None, "claude-fable-5-1"),
]
ENGINES = CANONICAL_ENGINES + ADDED_ENGINES
CANONICAL_ENGINE_NAMES = [e[0] for e in CANONICAL_ENGINES]
ENGINE_NAMES = [e[0] for e in ENGINES]
N_ENGINES = len(ENGINES)
DISPLAY_NAME = {e[0]: e[1] for e in ENGINES}
LLAMA3_8B_FAMILY = ["Meta-Llama-3-8B-Instruct", "Llama-3.1-8B-Instruct", "llama-3-meerkat-8b-v1.0",
                    "Llama-3-8B-UltraMedical", "DeepSeek-R1-Distill-Llama-8B"]


def group_of(engine: str) -> str:
    """Substring rule used for every grouped statistic and figure."""
    e = engine.lower()
    if any(k in e for k in ("gpt", "o1", "o3", "o4", "claude", "gemini")):
        return GROUP_PROPRIETARY
    if any(k in e for k in ("med", "meerkat", "ultramedical")):
        return GROUP_MEDICAL
    return GROUP_GENERAL


def engine_table() -> pd.DataFrame:
    """One row per canonical engine (the content of results/raw/qa_engines.csv)."""
    rows = [{"engine": e, "display_name": d, "group": group_of(e), "provider": p, "open_weights": ow,
             "params_b": pb, "api_model_id": mid} for e, d, p, ow, pb, mid in ENGINES]
    return pd.DataFrame(rows)


_counts = pd.Series([group_of(e) for e in CANONICAL_ENGINE_NAMES]).value_counts()
assert len(CANONICAL_ENGINES) == 32 and _counts[GROUP_GENERAL] == 18 and _counts[GROUP_MEDICAL] == 3 \
    and _counts[GROUP_PROPRIETARY] == 11, _counts.to_dict()
DROPPED_ENGINES = ["claude-3-5-haiku-20241022", "o1-mini"]  # preprint engines retired by their providers (see docstring)
assert len(set(ENGINE_NAMES)) == len(ENGINE_NAMES), "duplicate engine names"
