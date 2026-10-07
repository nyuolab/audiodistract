"""The audio pairs table feeds the ambient note generator layout and the unchanged ambient paired judge."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from llm_distract.ambient import judge
from llm_distract.audio import build_pairs, generate_notes, leakage, mix
from llm_distract.audio.data import load_corpus
from test_audio_data import synthetic_corpus

AMBIENT_PAIR_COLUMNS = ["source_dataset", "family", "item_id", "distractor_type", "clean_transcript", "distracted_transcript",
                        "reference_note", "distractor_conversation", "distractor_summary", "distractor_topic"]


@pytest.fixture(scope="module")
def pairs(tmp_path_factory):
    root = synthetic_corpus(tmp_path_factory.mktemp("primock57"))
    cons, utts = load_corpus(root)
    cons["seconds"] = [mix.wav_seconds(d) for d in cons["doctor_wav"]]
    manifest = mix.render(mix.plan(cons, utts, [-20, -10], 2, (5, 15), (0.2, 0.8), seed=0), cons, tmp_path_factory.mktemp("mixes"), 16000, 0.95)
    human = cons.set_index("item_id")["human_transcript"]
    transcripts = pd.DataFrame({"item_id": manifest["item_id"], "distractor_type": manifest["distractor_type"],
                                "text": [human[i] + ("" if c == "clean" else " " + s) for i, c, s in
                                         zip(manifest["item_id"], manifest["condition"], manifest["segment_text"])],
                                "segments": json.dumps([])})
    return build_pairs.build_pairs(manifest, transcripts, cons)


def test_pairs_layout(pairs):
    assert len(pairs) == 3 * 2 * 2 and list(pairs.columns[:10]) == AMBIENT_PAIR_COLUMNS
    assert (pairs["source_dataset"] == "primock57").all() and (pairs["family"] == "PriMock57").all()
    assert not pairs.duplicated(["source_dataset", "item_id", "distractor_type"]).any()
    r = pairs.iloc[0]
    assert r["distractor_summary"].startswith(build_pairs.SUMMARY_PREFIX) and r["distractor_conversation"].startswith(("Doctor: ", "Patient: "))
    assert r["distractor_summary"][len(build_pairs.SUMMARY_PREFIX):] in r["distracted_transcript"]
    assert r["reference_note"] == f"Note for {r['item_id']}." and r["distractor_topic"].startswith("complaint ")
    assert r["clean_transcript"] == r["human_transcript"] and r["wav"].endswith(f"{r['item_id']}__{r['distractor_type']}.wav")


def test_note_rows_match_ambient_layout(pairs):
    n = generate_notes.note_rows(pairs)
    assert len(n) == 2 * len(pairs) and list(n.columns) == ["source_dataset", "family", "item_id", "distractor_type", "condition",
                                                            "transcript", "reference_note", "distractor_summary"]
    assert n["transcript"].nunique() == 3 + 3 * 2       # one clean transcript per A; the fake ASR is level-independent


def test_ambient_paired_judge_runs_unchanged(pairs, tmp_path, monkeypatch):
    notes = generate_notes.note_rows(pairs)
    notes["note"] = "S: " + notes["transcript"].str[:40]
    notes["model"] = "meta-llama/Llama-3.1-8B-Instruct"
    calls = []
    monkeypatch.setattr(judge, "_call", lambda provider, model, system, user, **kw: calls.append(user) or
                        '{"note_a_contamination": 1, "note_a_severity": 2, "note_b_contamination": 0, "note_b_severity": 0, "reasoning": "x"}')
    out = judge.judge_paired(notes, pairs.assign(item_id=pairs["item_id"].astype(str)), "anthropic", "claude-sonnet-5", 0, tmp_path / "j.parquet")
    assert len(out) == 2 * len(pairs) and len(calls) == len(pairs) and not out["parse_error"].any()
    assert build_pairs.SUMMARY_PREFIX in calls[0] and pairs.iloc[0]["clean_transcript"] in calls[0]
    assert set(out["contamination"]) == {0, 1} and out.groupby(["item_id", "distractor_type"])["contamination"].sum().eq(1).all()


def test_leakage_on_perfectly_transcribed_overlay(pairs):
    t = leakage.leakage_table(pairs)
    assert len(t) == len(pairs) and (t["leakage"] + t["overlap_clean"]).round(9).eq(1.0).all() and (t["n_leaked"] > 0).all()


def test_build_pairs_requires_every_transcript(pairs, tmp_path_factory):
    root = synthetic_corpus(tmp_path_factory.mktemp("primock57b"))
    cons, utts = load_corpus(root)
    cons["seconds"] = 40.0
    manifest = mix.plan(cons, utts, [-10], 1, (5, 15), (0.2, 0.8), seed=0)
    transcripts = pd.DataFrame({"item_id": ["day1_consultation01"], "distractor_type": ["clean"], "text": ["x"], "segments": ["[]"]})
    with pytest.raises(RuntimeError, match="5 manifest recordings have no transcript"):
        build_pairs.build_pairs(manifest, transcripts, cons)
