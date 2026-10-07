"""CPU tests for the MedDistractQA module (insertion rule, parser, analysis on the shipped artifacts)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_distract.meddistractqa.data import insert_distractor
from llm_distract.meddistractqa.engines import ENGINE_NAMES, group_of, CANONICAL_ENGINE_NAMES
from llm_distract.meddistractqa.parse import extract_answer, is_correct

RAW = Path("results/raw")


def test_insertion_before_final_sentence():
    q = "A 4-year-old boy presents with fever. He has a rash. Which of the following is the diagnosis?"
    out = insert_distractor(q, "The patient's zodiac sign is Cancer.")
    assert out == ("A 4-year-old boy presents with fever. He has a rash. The patient's zodiac sign is Cancer. "
                   "Which of the following is the diagnosis?")


def test_insertion_single_sentence_question_prepends():
    assert insert_distractor("Which drug is first line?", "X.") == "X. Which drug is first line?"


@pytest.mark.parametrize("text,expected", [
    ("Therefore, the final answer is [B].", "(B)"),
    ("Therefore, the final answer is: C", "(C)"),
    ("Therefore,ĠtheĠfinalĠanswerĠisĠD.Ċ", "(D)"),
    ("... the answer is C.", "(C)"),
    ("\\boxed{D}", "(D)"),
    ("I think (a) is right", "[invalid]"),  # lower-case letters are not recognised
    ("no letter here", "[invalid]"),
])
def test_parser(text, expected):
    assert extract_answer(text) == expected


def test_strict_scoring():
    assert is_correct("(B)", "B") and not is_correct("[invalid]", "B")


def test_group_rule():
    assert group_of("gemini-3.1-pro-preview") == "Proprietary"
    assert group_of("llama-3-meerkat-8b-v1.0") == "Open-weight, medical"
    assert group_of("Llama-3.1-8B-Instruct") == "Open-weight, general"
    assert group_of("o4-mini") == "Proprietary" and group_of("claude-fable-5-1") == "Proprietary"
    groups = pd.Series([group_of(e) for e in ENGINE_NAMES]).value_counts()
    assert groups.to_dict() == {"Open-weight, general": 18, "Open-weight, medical": 3, "Proprietary": 17}
    assert len(ENGINE_NAMES) >= len(CANONICAL_ENGINE_NAMES)


@pytest.mark.skipif(not (RAW / "qa_per_item.parquet").exists(), reason="raw artifacts not present")
def test_headline_numbers_from_raw():
    per = pd.read_parquet(RAW / "qa_per_item.parquet")
    std = per[(per["prompt_variant"] == "standard") & per["engine"].isin(ENGINE_NAMES)]
    wide = std.pivot(index=["engine", "item_id"], columns="condition", values=["is_correct", "is_invalid"])
    evaluable = ~wide["is_invalid"]["clean"] & ~wide["is_invalid"]["nonliteral"]
    per_item_loss = 100 * (wide["is_correct"]["clean"].astype(int) - wide["is_correct"]["nonliteral"].astype(int))
    loss_nl = per_item_loss.where(evaluable).groupby(level="engine").mean()
    assert len(loss_nl) == len(ENGINE_NAMES) and set(std.groupby("engine")["item_id"].nunique()) == {1273}
    assert abs(loss_nl.min() - 0.3943) < 0.01 and abs(loss_nl.max() - 16.7324) < 0.01
    prop = [e for e in loss_nl.index if group_of(e) == "Proprietary"]
    assert abs(loss_nl.loc[prop].mean() - 2.6754) < 0.01
    assert np.isclose(len(prop), 17)
