"""CPU tests for llm_distract.audio.data: TextGrid parsing, keyword scoring, donor segments, seeded donors.

``synthetic_corpus`` writes a tiny PriMock57-shaped corpus (WAVs, TextGrids, notes) and is reused by the other tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_distract.audio import data
from llm_distract.audio.mix import write_wav

REAL_SNIPPET = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 457.92
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "Doctor"
        xmin = 0
        xmax = 457.92
        intervals: size = 4
        intervals [1]:
            xmin = 0
            xmax = 2.5334561157322537
            text = ""
        intervals [2]:
            xmin = 2.5334561157322537
            xmax = 12.499861706065632
            text = "Hello? Hi. Um, should we start? Yeah, okay. <UNSURE>Hello how</UNSURE> um. Good morning sir, how can I help you this morning?"
        intervals [3]:
            xmin = 39.36552045421459
            xmax = 40.15008379175207
            text = "<UNIN/>"
        intervals [4]:
            xmin = 40.15008379175207
            xmax = 45.0
            text = "He said ""fine"" and left."
'''
DOCTOR = ["how can I help you today", "any fever or cough with that", "I will prescribe {drug} and an inhaler for the asthma",
          "let me listen to your chest and lungs", "come back in a week"]
PATIENT = ["I have had {complaint} and a cough for three days", "the headache is worse at night", "no vomiting or diarrhoea",
           "thank you doctor", "okay"]
CASES = [{"complaint": "chest pain", "drug": "paracetamol"}, {"complaint": "a headache", "drug": "ibuprofen"},
         {"complaint": "a rash", "drug": "an ointment"}, {"complaint": "sciatica", "drug": "naproxen"}]


def write_textgrid(path: Path, name: str, intervals: list, xmax: float) -> None:
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0 ", f"xmax = {xmax} ", "tiers? <exists> ",
             "size = 1 ", "item []: ", "    item [1]:", '        class = "IntervalTier" ', f'        name = "{name}" ',
             "        xmin = 0 ", f"        xmax = {xmax} ", f"        intervals: size = {len(intervals)} "]
    for k, (a, b, t) in enumerate(intervals, 1):
        lines += [f"        intervals [{k}]:", f"            xmin = {a} ", f"            xmax = {b} ", f'            text = "{t}" ']
    path.write_text("\n".join(lines) + "\n")


def synthetic_corpus(root: Path, n: int = 3, seconds: float = 40.0, sr: int = 16000, seed: int = 0) -> Path:
    """n (<= 4) consultations of ``seconds`` s with distinct complaint/drug words: noise channels, 3-s utterances every
    8 s (patient offset 4 s), JSON notes."""
    rng = np.random.default_rng(seed)
    for d in ("audio", "transcripts", "notes"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        item = f"day1_consultation{i + 1:02d}"
        for k, (sp, texts) in enumerate((("doctor", DOCTOR), ("patient", PATIENT))):
            write_wav(root / "audio" / f"{item}_{sp}.wav", rng.normal(0, 0.05 + 0.03 * k, int(seconds * sr)).astype(np.float32), sr)
            intervals, t = [], 0.0
            for u, text in enumerate(texts):
                start = 8.0 * u + 4.0 * k
                intervals += [(t, start, ""), (start, start + 3.0, text.format(**CASES[i]))]
                t = start + 3.0
            intervals.append((t, seconds, ""))
            write_textgrid(root / "transcripts" / f"{item}_{sp}.TextGrid", sp.capitalize(), intervals, seconds)
        (root / "notes" / f"{item}.json").write_text(json.dumps({"day": 1, "consultation": i + 1, "presenting_complaint": f"complaint {i + 1}",
                                                                 "note": f"Note for {item}."}))
    return root


def test_parse_real_textgrid_snippet(tmp_path):
    p = tmp_path / "x.TextGrid"
    p.write_text(REAL_SNIPPET)
    got = data.parse_textgrid(p)
    assert len(got) == 2  # empty and tag-only intervals are dropped
    assert got[0]["start"] == pytest.approx(2.5334561157322537) and got[0]["end"] == pytest.approx(12.499861706065632)
    assert got[0]["text"] == "Hello? Hi. Um, should we start? Yeah, okay. Hello how um. Good morning sir, how can I help you this morning?"
    assert got[1]["text"] == 'He said "fine" and left.'


def test_clinical_score_counts_keywords_and_plurals():
    assert data.clinical_score("Chest pain and a cough, headaches, no tablets, thank you") == 5
    assert data.clinical_score("thank you, see you next week") == 0


def test_select_segment_maximises_keywords_with_tie_breaks():
    units = pd.DataFrame({"speaker": ["Doctor", "Patient"] * 3, "start": [0, 4, 8, 12, 16, 20], "end": [3, 7, 11, 15, 19, 23],
                          "text": ["hello there", "chest pain and cough", "fever vomiting diarrhoea rash", "thank you", "asthma", "bye"]})
    seg = data.select_segment(units, 5, 8)          # only pairs of consecutive utterances (span 7 s) fit
    assert (seg["start"], seg["end"], seg["score"], seg["n_units"]) == (4, 11, 7, 2)
    assert seg["lines"] == "Patient: chest pain and cough\nDoctor: fever vomiting diarrhoea rash"
    assert seg["text"] == "chest pain and cough fever vomiting diarrhoea rash"
    seg12 = data.select_segment(units, 5, 12)       # triples (score 7, span 11 s) tie with the pair -> shorter span wins
    assert (seg12["start"], seg12["end"], seg12["score"]) == (4, 11, 7)
    with pytest.raises(ValueError):
        data.select_segment(units, 30, 40)


def test_assign_donors_seeded_distinct_and_never_self():
    ids = [f"c{i}" for i in range(6)]
    a = data.assign_donors(ids, 2, seed=0)
    assert len(a) == 12 and (a["item_id"] != a["donor_id"]).all()
    assert a.groupby("item_id")["donor_id"].nunique().eq(2).all() and list(a["donor"].unique()) == [0, 1]
    pd.testing.assert_frame_equal(a, data.assign_donors(ids, 2, seed=0))
    assert not a["donor_id"].equals(data.assign_donors(ids, 2, seed=1)["donor_id"])


def test_load_corpus_synthetic(tmp_path):
    cons, utts = data.load_corpus(synthetic_corpus(tmp_path / "primock57"))
    assert list(cons["item_id"]) == ["day1_consultation01", "day1_consultation02", "day1_consultation03"]
    assert len(utts) == 30 and utts.groupby("item_id")["start"].apply(lambda s: s.is_monotonic_increasing).all()
    assert cons["human_transcript"].iloc[0].startswith("Doctor: how can I help you today\nPatient: I have had chest pain")
    assert cons["note"].iloc[1] == "Note for day1_consultation02." and cons["doctor_wav"].iloc[0].endswith("day1_consultation01_doctor.wav")
    assert not data.audio_present(tmp_path)          # no audio dir -> False; real WAVs -> True
    assert data.audio_present(tmp_path / "primock57")


def test_whisper_units_from_clean_chunks():
    tr = pd.DataFrame({"item_id": ["a", "a"], "distractor_type": ["clean", "audio_-5dB_d0"],
                       "segments": [json.dumps([{"start": 0.0, "end": 2.0, "text": " Hello."}, {"start": 2.0, "end": None, "text": "open"}]), "[]"]})
    u = data.whisper_units(tr)
    assert len(u) == 1 and u.iloc[0]["text"] == "Hello." and u.iloc[0]["speaker"] == ""
    assert data.segment_lines(u) == "Hello."


def test_load_corpus_skips_missing_textgrid(tmp_path):
    root = synthetic_corpus(tmp_path / "primock57")
    (root / "transcripts" / "day1_consultation02_doctor.TextGrid").unlink()
    (root / "transcripts" / "day1_consultation02_patient.TextGrid").unlink()
    cons, utts = data.load_corpus(root)
    assert len(cons) == 3 and set(utts["item_id"]) == {"day1_consultation01", "day1_consultation03"} and len(utts) == 20
    assert cons.set_index("item_id").loc["day1_consultation02", "human_transcript"] == ""
    with pytest.raises(ValueError, match="no timestamped units"):
        data.select_segment(utts[utts["item_id"] == "day1_consultation02"], 5, 15)
    assert data.assign_donors(cons["item_id"], 0, seed=0).empty and list(data.assign_donors(cons["item_id"], 0, seed=0).columns) == ["item_id", "donor", "donor_id"]
