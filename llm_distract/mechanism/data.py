"""MedDistractQA pairs, the locked item split, MCQ prompts and tokenisation for the mechanism module.

Everything here is CPU-only and torch-free at import time so that analysis and export can run without GPU
dependencies. Prompts use the native chat template by default, or raw text with MECH_CHAT_TEMPLATE=0; their chat-templated
equivalent ending in the "Answer:" assistant prefill; the scored target is a single letter " A".." D".
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

CHOICE_LETTERS = ("A", "B", "C", "D")
DISTRACTORS = ("nonliteral", "bystander")
CONDITIONS = ("clean",) + DISTRACTORS
PAIR_COLUMNS = ["item_id", "correct_answer", "medical_system", "medical_competency",
                "choice_A", "choice_B", "choice_C", "choice_D",
                "clean_question", "nonliteral_question", "bystander_question",
                "nonliteral_sentence", "bystander_sentence"]

BASELINE_INSTRUCTION = ("You are a medical expert. Answer the following USMLE-style question by selecting the "
                        "correct option (A, B, C, or D).")
IGNORE_INSTRUCTION = (BASELINE_INSTRUCTION + " IMPORTANT: The question may contain irrelevant information that is "
                      "not clinically meaningful. Focus only on the medically relevant details when making your diagnosis.")
COT_SUFFIX = 'Think step by step, then provide your final answer in the format: "Therefore, the answer is [X]"'
PROMPT_STYLES = ("baseline", "ignore", "cot")


# ----------------------------------------------------------------------------- items and split
def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    return text.strip()


def reconstruct_clean_question(question: str, sentence: str) -> str:
    """Remove the inserted distractor sentence from a distracted question (same rule as the original prep)."""
    if sentence not in question:
        raise ValueError("distracting sentence not found in question")
    prefix, _, suffix = question.partition(sentence)
    return _tidy(f"{prefix.rstrip()} {suffix.lstrip()}".strip())


def pairs_from_json(json_dir: Path) -> pd.DataFrame:
    """Build the paired table from the public MedDistractQA_{Nonliteral,Bystander}.json files."""
    json_dir = Path(json_dir)
    nl = json.loads((json_dir / "MedDistractQA_Nonliteral.json").read_text(encoding="utf-8"))
    by = json.loads((json_dir / "MedDistractQA_Bystander.json").read_text(encoding="utf-8"))
    if len(nl) != len(by):
        raise ValueError("the two distractor files must have the same number of items")
    rows = []
    for item_id, (a, b) in enumerate(zip(nl, by)):
        clean = reconstruct_clean_question(a["question"], a["distracting_sentence"])
        clean_b = reconstruct_clean_question(b["question"], b["distracting_sentence"])
        if re.sub(r"\s+", " ", clean) != re.sub(r"\s+", " ", clean_b):
            raise ValueError(f"clean reconstruction mismatch for item {item_id}")
        if a["question_choices"] != b["question_choices"] or a["correct_answer"] != b["correct_answer"]:
            raise ValueError(f"choice/answer mismatch for item {item_id}")
        rows.append({
            "item_id": item_id, "correct_answer": a["correct_answer"],
            "medical_system": a["medical_system"], "medical_competency": a["medical_competency"],
            **{f"choice_{k}": a["question_choices"].get(k, "") for k in CHOICE_LETTERS},
            "clean_question": clean, "nonliteral_question": a["question"], "bystander_question": b["question"],
            "nonliteral_sentence": a["distracting_sentence"], "bystander_sentence": b["distracting_sentence"],
        })
    return pd.DataFrame.from_records(rows, columns=PAIR_COLUMNS)


def load_pairs(data_dir: Path = Path("data"), json_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load the 1,273 paired items.

    Prefers ``<data_dir>/meddistractqa/meddistractqa_v2.parquet`` (written by the meddistractqa module, one row per
    item with the PAIR_COLUMNS); otherwise rebuilds the table from the two public JSON files in ``json_dir``
    (default ``<data_dir>/meddistractqa``). Nothing is written.
    """
    parquet = Path(data_dir) / "meddistractqa" / "meddistractqa_v2.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet)
        missing = [c for c in PAIR_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{parquet} lacks columns {missing}")
        return df[PAIR_COLUMNS].sort_values("item_id").reset_index(drop=True)
    return pairs_from_json(json_dir or Path(data_dir) / "meddistractqa")


def write_parquet(df: pd.DataFrame, path: Path, provenance: dict) -> Path:
    """Write a parquet file with a JSON provenance record in its schema metadata (key ``provenance``)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"provenance"] = json.dumps(provenance, default=str).encode()
    pq.write_table(table.replace_schema_metadata(meta), path)
    return path


def read_provenance(path: Path) -> dict:
    """Return the provenance record stored by ``write_parquet`` (empty dict if absent)."""
    import pyarrow.parquet as pq

    meta = pq.read_schema(path).metadata or {}
    return json.loads(meta.get(b"provenance", b"{}"))


def assign_splits(item_ids: Iterable[int], fraction: float = 0.1, seed: int = 0) -> dict:
    """Deterministic item split: shuffle sorted unique ids with random.Random(seed); first int(N*fraction) -> heldout."""
    ids = sorted(set(int(i) for i in item_ids))
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * fraction))
    val = set(shuffled[:n_val])
    return {i: ("heldout" if i in val else "train") for i in ids}


def heldout_ids(item_ids: Iterable[int], fraction: float = 0.1, seed: int = 0) -> set:
    """The locked evaluation cohort (127 of 1,273 items for the published split), never used for CCHG training."""
    return {i for i, s in assign_splits(item_ids, fraction, seed).items() if s == "heldout"}


def select_split(pairs: pd.DataFrame, split: str, fraction: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Restrict a pairs table to 'heldout', 'train' or 'all'."""
    if split == "all":
        return pairs.reset_index(drop=True)
    splits = assign_splits(pairs["item_id"], fraction, seed)
    keep = pairs["item_id"].map(splits) == split
    return pairs[keep].reset_index(drop=True)


# ----------------------------------------------------------------------------- prompts and tokens
def question_for(row, condition: str) -> str:
    return row["clean_question"] if condition == "clean" else row[f"{condition}_question"]


def build_prompt(question: str, choices: dict, style: str = "baseline") -> str:
    """The exact MCQ prompt used for every log-prob evaluation (append ' X' as the scored target).

    baseline: instruction + question block; ignore: the explicit ignore-irrelevant instruction; cot: baseline
    instruction followed by the chain-of-thought suffix paragraph.
    """
    if style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style {style}")
    instruction = IGNORE_INSTRUCTION if style == "ignore" else BASELINE_INSTRUCTION
    parts = [instruction] + ([COT_SUFFIX] if style == "cot" else [])
    choice_block = "\n".join(f"{k}. {choices[k]}" for k in CHOICE_LETTERS if choices.get(k))
    parts.append("\n".join([f"Question: {question}", choice_block, "Answer:"]))
    return "\n\n".join(parts)


def prompt_for(row, condition: str, style: str = "baseline") -> str:
    text = build_prompt(question_for(row, condition), {k: row[f"choice_{k}"] for k in CHOICE_LETTERS}, style)
    return apply_chat(text) if CHAT_TEMPLATE else text


import os

# Legacy raw-prompt mechanism runs omitted the beginning-of-sequence token. Setting
# MECH_ADD_BOS=1 prepends it (after left truncation, so it stays first) for tokenizers that define one; every manifest
# records the setting. Gemma-2 in particular degrades without it (see the paper's limitations).
ADD_BOS = os.environ.get("MECH_ADD_BOS", "0") == "1"
# MECH_CHAT_TEMPLATE=1 wraps every prompt in the model's chat template (user turn + generation prompt) instead of the
# older raw-text protocol; the template supplies the BOS token, the trailing "Answer:" is moved into
# the assistant turn (apply_chat) and the answer target stays " X". The tokenizer is registered by models.setup().
CHAT_TEMPLATE = os.environ.get("MECH_CHAT_TEMPLATE", "1") == "1"
_TOKENIZER = None


def set_tokenizer(tokenizer) -> None:
    global _TOKENIZER
    _TOKENIZER = tokenizer


ANSWER_PREFIX = "Answer:"


def apply_chat(prompt: str) -> str:
    """The prompt wrapped in the registered tokenizer's chat template: the question and options form the single user
    turn and the trailing ``Answer:`` opens the assistant turn (after the generation prompt), so the scored target is
    the same `` X`` continuation as in the raw-text protocol. Scoring the bare letter as the first assistant token is
    unreliable for chat models (Gemma-2 put log-probabilities near -10 on every letter and answered near chance)."""
    if _TOKENIZER is None:
        raise RuntimeError("MECH_CHAT_TEMPLATE=1 but no tokenizer registered (models.setup() does this)")
    body = prompt.rstrip()
    if body.endswith(ANSWER_PREFIX):
        body = body[: -len(ANSWER_PREFIX)].rstrip()
    rendered = _TOKENIZER.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
    return rendered + ANSWER_PREFIX


def bos_prefix(tokenizer) -> list:
    """[bos_token_id] when BOS handling is on and the tokenizer defines a BOS token, else []."""
    return [tokenizer.bos_token_id] if ADD_BOS and getattr(tokenizer, "bos_token_id", None) is not None else []


def tokenize_prompt_targets(tokenizer, prompts: list, targets: list, max_length: Optional[int] = None):
    """Tokenise prompt+target pairs into right-padded ids and a boolean target mask.

    No special tokens except the optional BOS prefix (``MECH_ADD_BOS=1``) or the BOS that a chat template renders in
    front of the prompt. Long prompts are truncated from the left so that every target token and the BOS token are
    retained (until 2026-09-04 a template-rendered BOS was dropped by the truncation; this affected only prompts longer
    than ``max_length``, 1-4% of gate-training prompts). Returns two LongTensors [B, L] (ids) and [B, L] bool
    (True on target tokens).
    """
    import torch

    ids_list, mask_list = [], []
    bos = bos_prefix(tokenizer)
    for prompt, target in zip(prompts, targets):
        prompt = prompt.rstrip()  # raw text ends with "Answer:", chat text with the "Answer:" assistant prefill
        target = target if target.startswith(" ") else f" {target}"
        p_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        bos_id = getattr(tokenizer, "bos_token_id", None)
        if bos_id is not None and p_ids[:1] == [bos_id]:
            bos, p_ids = [bos_id], p_ids[1:]  # a BOS carried by the chat template is kept in front of any left truncation
        t_ids = list(tokenizer(target, add_special_tokens=False)["input_ids"])
        if not t_ids:
            raise ValueError("target tokenisation produced no tokens")
        if max_length is not None:
            if len(t_ids) + len(bos) >= max_length:
                raise ValueError("max_length too short to keep the target")
            p_ids = p_ids[-(max_length - len(t_ids) - len(bos)):]
        p_ids = bos + p_ids
        ids_list.append(torch.tensor(p_ids + t_ids, dtype=torch.long))
        mask_list.append(torch.tensor([False] * len(p_ids) + [True] * len(t_ids), dtype=torch.bool))
    pad = tokenizer.pad_token_id
    ids = torch.nn.utils.rnn.pad_sequence(ids_list, batch_first=True, padding_value=pad)
    masks = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True, padding_value=False)
    return ids, masks


def contrastive_tensors(tokenizer, pairs: pd.DataFrame, distractor: str, max_length: int = 512) -> dict:
    """Clean (positive) and distracted (negative) prompt tokens for CCHG training, same answer target."""
    if distractor not in DISTRACTORS:
        raise ValueError("contrastive pairs are defined for the distracted conditions only")
    rows = pairs.to_dict("records")
    targets = [r["correct_answer"] for r in rows]
    pos_ids, pos_masks = tokenize_prompt_targets(tokenizer, [prompt_for(r, "clean") for r in rows], targets, max_length)
    neg_ids, neg_masks = tokenize_prompt_targets(tokenizer, [prompt_for(r, distractor) for r in rows], targets, max_length)
    import torch
    return {"pos_ids": pos_ids, "pos_masks": pos_masks, "neg_ids": neg_ids, "neg_masks": neg_masks,
            "item_id": torch.tensor([int(r["item_id"]) for r in rows], dtype=torch.long)}
