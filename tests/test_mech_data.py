"""Locked split, exact prompt strings and target-preserving tokenisation (CPU, no model downloads)."""
from __future__ import annotations

import pytest

from llm_distract.mechanism import data as D


def test_split_is_deterministic_and_locked():
    held = D.heldout_ids(range(1273))
    assert len(held) == 127
    assert sorted(held)[:5] == [23, 24, 50, 57, 76]
    assert D.heldout_ids(reversed(range(1273))) == held  # order of the input ids does not matter
    splits = D.assign_splits(range(1273))
    assert sum(s == "train" for s in splits.values()) == 1146


def test_prompt_exact_string():
    choices = {"A": "Aspirin", "B": "Heparin", "C": "Warfarin", "D": "Observation"}
    prompt = D.build_prompt("A 45-year-old man presents with chest pain. What is the next step?", choices)
    assert prompt == (
        "You are a medical expert. Answer the following USMLE-style question by selecting the correct option "
        "(A, B, C, or D).\n\nQuestion: A 45-year-old man presents with chest pain. What is the next step?\n"
        "A. Aspirin\nB. Heparin\nC. Warfarin\nD. Observation\nAnswer:")
    assert D.build_prompt("Q", choices, "ignore").startswith(D.IGNORE_INSTRUCTION + "\n\nQuestion: Q")
    assert D.build_prompt("Q", choices, "cot").split("\n\n")[1] == D.COT_SUFFIX
    with pytest.raises(ValueError):
        D.build_prompt("Q", choices, "self_consistency")


def test_clean_reconstruction_removes_sentence():
    q = "First sentence. The patient joked about a surgical incision. Last sentence?"
    assert D.reconstruct_clean_question(q, "The patient joked about a surgical incision.") == "First sentence. Last sentence?"


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_tokenization_keeps_target_under_left_truncation():
    torch = pytest.importorskip("torch")
    ids, masks = D.tokenize_prompt_targets(FakeTokenizer(), ["abcdefgh", "xy"], ["A", "B"], max_length=5)
    assert ids.shape == (2, 5) and masks.shape == (2, 5)
    assert ids[0].tolist() == [ord(c) for c in "fgh A"]  # last 3 prompt chars + " A"
    assert masks[0].tolist() == [False, False, False, True, True]
    assert ids[1].tolist() == [ord("x"), ord("y"), ord(" "), ord("B"), 0]  # right padded
    assert masks[1].sum() == 2 and not masks[1][-1]
    assert torch.is_tensor(ids)
