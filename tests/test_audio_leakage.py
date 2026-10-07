"""CPU tests for llm_distract.audio.leakage on toy strings."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_distract.audio import leakage as lk


def test_content_words_drop_short_and_stopwords():
    assert lk.content_words("I have had chest pain and a cough for three days") == {"chest", "pain", "cough", "three"}


def test_leakage_fraction_on_toy_strings():
    seg = "Patient: I have had chest pain and a cough for three days"
    clean = "the patient has a headache and feels tired"
    mixed = "the patient has a headache chest pain and feels tired cough"
    r = lk.leakage(seg, mixed, clean)
    assert (r["n_content"], r["n_leaked"], r["leaked_words"]) == (5, 3, "chest cough pain")
    assert r["leakage"] == pytest.approx(0.6) and r["overlap_clean"] == pytest.approx(0.2)
    assert lk.leakage(seg, clean, clean)["leakage"] == 0.0          # nothing new in the mixed transcript
    assert np.isnan(lk.leakage("um, ok", mixed, clean)["leakage"])   # no content words


def test_wer():
    assert lk.wer("a b c d", "a b x d") == pytest.approx(0.25)
    assert lk.wer("a b c", "a b c") == 0.0 and lk.wer("a b", "a b c") == pytest.approx(0.5)
    assert lk.wer("a b c", "") == 1.0 and lk.wer("a b c", "b c") == pytest.approx(1 / 3)
    assert np.isnan(lk.wer("", "a"))


def test_leakage_table_columns():
    pairs = pd.DataFrame([{"item_id": "a", "distractor_type": "audio_-10dB_d0", "donor": 0, "donor_id": "b", "level_db": -10.0, "onset": 10.0,
                           "segment_start": 1.0, "segment_end": 8.0, "segment_seconds": 7.0, "segment_score": 3,
                           "distractor_summary": lk.SUMMARY_PREFIX + "any fever or cough", "clean_transcript": "hello there",
                           "distracted_transcript": "hello there fever", "human_transcript": "hello there"}])
    t = lk.leakage_table(pairs)
    assert t.iloc[0]["leakage"] == pytest.approx(0.5) and t.iloc[0]["wer_vs_clean"] == pytest.approx(0.5) and t.iloc[0]["wer_clean_vs_human"] == 0.0
    assert set(lk.DESIGN + ["n_content", "n_leaked", "leakage", "overlap_clean", "leaked_words", "wer_vs_clean", "wer_clean_vs_human"]) == set(t.columns)
