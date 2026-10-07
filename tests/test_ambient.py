"""CPU tests for the ambient module (splice rule, cohort loader on an inline CSV, analysis on shipped artifacts)."""
from pathlib import Path

import pandas as pd
import pytest

from llm_distract.ambient.data import insert_conversation, speaker_tags

RAW = Path("results/raw")


def test_insert_after_turn():
    t = "Doctor: Hi.\nPatient: Hello.\nDoctor: What brings you in?\nPatient: Cough."
    ins = "Patient: My coworker had pneumonia.\nDoctor: Sorry to hear."
    out = insert_conversation(t, ins, 2)
    assert out.splitlines()[2:4] == ins.splitlines()
    assert out.replace(ins + "\n", "") == t


def test_insert_clamps_and_relabels_bracketed_transcripts():
    t = "Doctor: Hi.\nPatient: Hello."
    assert insert_conversation(t, "Doctor: aside.", 99).endswith("Doctor: aside.")
    aci = "[doctor] hi there\n[patient] hello\n[doctor] what brings you in"
    out = insert_conversation(aci, "Doctor: my neighbor has a cat.\nPatient: that's funny.", 1)
    assert out.splitlines() == ["[doctor] hi there", "[doctor] my neighbor has a cat.", "[patient] that's funny.",
                                "[patient] hello", "[doctor] what brings you in"]
    assert insert_conversation("no labels here", "Doctor: aside.", 1) == "no labels here\nDoctor: aside."
    assert speaker_tags(aci) == ["Doctor", "Patient"]   # what the insertion generator was shown
    assert speaker_tags(t) == ["Doctor", "Patient"]


@pytest.mark.skipif(not (RAW / "amb_judgments.parquet").exists(), reason="raw artifacts not present")
def test_headline_numbers_from_raw():
    from llm_distract.common.style import MECHANISM_MODELS

    j = pd.read_parquet(RAW / "amb_judgments.parquet")
    j = j[j["model"].isin(MECHANISM_MODELS)]  # the prespecified open-weight cohort (the frontier cohort shares the file)
    wide = j.pivot_table(index=["source_dataset", "item_id", "model", "distractor_type"], columns="condition", values="contamination_v3")
    assert len(wide) == 4616
    delta = wide["distracted"] - wide["clean"]
    assert abs(delta.mean() * 100 - 41.8) < 0.1
    strata = delta.groupby(level=["model", "source_dataset", "distractor_type"]).mean() * 100
    assert len(strata) == 40 and abs(strata.mean() - 33.1) < 0.1 and (strata > 0).sum() == 39
    assert wide["clean"].sum() == 2


def test_export_reruns_replaces_matching_rows(tmp_path):
    """Folding a rerun replaces rows with the same keys and leaves the rest untouched."""
    from llm_distract.ambient.export_reruns import collect, fold

    raw = tmp_path / "raw"
    raw.mkdir()
    keys = dict(source_dataset="aci_bench_test", item_id="D2N001", model="m", distractor_type="bystander")
    base_notes = pd.DataFrame([{**keys, "family": "ACI-Bench", "condition": c, "note": "old", "note_chars": 3, "transcript_chars": 10}
                               for c in ("clean", "distracted")] +
                              [{**keys, "source_dataset": "mts_dialog_valid", "family": "MTS-Dialog", "condition": "clean", "note": "keep",
                                "note_chars": 4, "transcript_chars": 5}])
    base_notes.to_parquet(raw / "amb_notes.parquet", index=False)
    run = tmp_path / "runs" / "m" / "aci_bench_test" / "bystander"
    run.mkdir(parents=True)
    notes = pd.DataFrame([{**keys, "condition": c, "transcript": "t" * 20, "note": "new"} for c in ("clean", "distracted")])
    notes.to_parquet(run / "notes.parquet", index=False)
    pd.DataFrame([{**keys, "condition": c, "contamination": int(c == "distracted"), "severity": 2 * int(c == "distracted"),
                   "judge_reasoning": "r", "judge_protocol": "v3_paired_symmetric"} for c in ("clean", "distracted")]
                 ).to_parquet(run / "judge_contamination_paired.parquet", index=False)
    pd.DataFrame([{**keys, "condition": c, "clinical_correctness": 4, "completeness": 4, "succinctness": 4, "hallucination": 0,
                   "overall_quality": 4} for c in ("clean", "distracted")]).to_parquet(run / "judge_quality.parquet", index=False)
    new_notes, new_judg = collect(tmp_path / "runs")
    assert list(new_notes["note_chars"]) == [3, 3] and list(new_notes["transcript_chars"]) == [20, 20]
    assert new_judg["contamination_v2_gpt"].isna().all() and list(new_judg["contamination_v3"]) == [0, 1]
    merged = fold(raw, "amb_notes", new_notes, {})
    assert (raw / "amb_notes_original.parquet").exists()
    assert len(merged) == 3 and set(merged["note"]) == {"new", "keep"}


def test_single_note_judge_uses_target_only_for_distracted(tmp_path, monkeypatch):
    from llm_distract.ambient import judge

    calls = []
    monkeypatch.setattr(judge, "_call", lambda provider, model, system, user, **kw: calls.append(user) or
                        '{"distractor_contamination": 1, "contamination_severity": 2, "reasoning": "x"}')
    notes = pd.DataFrame([{"source_dataset": "aci_bench_test", "item_id": "D2N001", "distractor_type": "bystander", "condition": c,
                           "model": "m", "transcript": "[doctor] hi", "note": "S: ...", "distractor_summary": "coworker pneumonia"}
                          for c in ("clean", "distracted")])
    out = judge.judge_single(notes, "openai", "gpt-5.4", tmp_path / "j.parquet")
    assert len(out) == 2 and list(out["contamination"]) == [1, 1] and (out["judge_protocol"] == "v2_single_note").all()
    assert "coworker pneumonia" in calls[1] and "coworker pneumonia" not in calls[0]
    again = judge.judge_single(notes, "openai", "gpt-5.4", tmp_path / "j.parquet")   # resumable: no new calls
    assert len(calls) == 2 and len(again) == 2
