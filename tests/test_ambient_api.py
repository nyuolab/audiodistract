"""API backend of the note generator and the concurrent judge, with the network calls stubbed."""
from pathlib import Path

import pandas as pd

from llm_distract.ambient import generate_notes, judge


def test_generate_api_checkpoints_and_resumes(tmp_path, monkeypatch):
    calls = []

    def fake_call(provider, engine, prompt, temperature, max_tokens, force_temperature, reasoning_effort=None, thinking="default"):
        calls.append(prompt)
        if "fail" in prompt:
            raise RuntimeError("rate limit")
        return f"NOTE for {prompt}", {"input_tokens": 10, "output_tokens": 5, "finish_reason": "stop"}

    monkeypatch.setattr("llm_distract.meddistractqa.run_eval.call_api", fake_call)
    monkeypatch.setattr("time.sleep", lambda s: None)
    keys = [("1", "clean"), ("1", "distracted"), ("2", "clean")]
    prompts = ["p1", "p2", "fail"]
    ck = tmp_path / "ck.jsonl"
    df = generate_notes.generate_api("openai", "m", keys, prompts, 64, 0.0, 2, ck, attempts=2)
    assert list(df["note"]) == ["NOTE for p1", "NOTE for p2", ""]
    assert list(df["finish_reason"]) == ["stop", "stop", "error"] and df["api_error"].iloc[2].startswith("RuntimeError")
    assert len(ck.read_text().splitlines()) == 2  # only successful notes are checkpointed
    n_before = len(calls)
    df2 = generate_notes.generate_api("openai", "m", keys, ["p1", "p2", "p3"], 64, 0.0, 2, ck, attempts=2)
    assert list(df2["note"]) == ["NOTE for p1", "NOTE for p2", "NOTE for p3"]
    assert len(calls) == n_before + 1  # the two checkpointed notes were not requested again


def _notes():
    rows = []
    for iid in ("a", "b", "c"):
        for cond in ("clean", "distracted"):
            rows.append({"source_dataset": "mts_dialog_test1", "item_id": iid, "distractor_type": "bystander", "condition": cond,
                         "model": "m", "note": f"note {iid} {cond}", "transcript": "t", "reference_note": "r"})
    return pd.DataFrame(rows)


def _pairs():
    return pd.DataFrame([{"source_dataset": "mts_dialog_test1", "item_id": iid, "distractor_type": "bystander",
                          "clean_transcript": "t", "distractor_summary": "s"} for iid in ("a", "b", "c")])


def test_judges_concurrent_match_sequential(tmp_path, monkeypatch):
    def fake_call(provider, model, system, user, max_tokens=800, retries=4):
        if "note_a" in system.lower() or "Note A" in user:
            return '{"note_a_contamination": 1, "note_a_severity": 2, "note_b_contamination": 0, "note_b_severity": 0, "reasoning": "x"}'
        return '{"clinical_correctness": 4, "completeness": 3, "succinctness": 4, "hallucination": 0, "overall_quality": 4, "reasoning": "y"}'

    monkeypatch.setattr(judge, "_call", fake_call)
    seq = judge.judge_paired(_notes(), _pairs(), "anthropic", "j", 0, tmp_path / "seq.parquet", concurrency=1)
    par = judge.judge_paired(_notes(), _pairs(), "anthropic", "j", 0, tmp_path / "par.parquet", concurrency=4)
    cols = ["source_dataset", "item_id", "distractor_type", "condition", "contamination", "severity", "shown_as"]
    pd.testing.assert_frame_equal(seq[cols].reset_index(drop=True), par[cols].reset_index(drop=True))
    assert len(par) == 6 and not par["parse_error"].any()
    qs = judge.judge_quality(_notes(), "openai", "q", tmp_path / "qs.parquet", concurrency=1)
    qp = judge.judge_quality(_notes(), "openai", "q", tmp_path / "qp.parquet", concurrency=3)
    qcols = ["item_id", "condition", "clinical_correctness", "overall_quality", "hallucination"]
    pd.testing.assert_frame_equal(qs[qcols].reset_index(drop=True), qp[qcols].reset_index(drop=True))
    assert Path(tmp_path / "qp.parquet").exists()


def test_chat_template_bos_survives_left_truncation(monkeypatch):
    """A BOS rendered by the chat template stays first when a long prompt is truncated from the left."""
    import os

    monkeypatch.setenv("MECH_CHAT_TEMPLATE", "1")
    import importlib
    from llm_distract.mechanism import data
    importlib.reload(data)

    class Tok:
        bos_token_id = 2
        pad_token_id = 0

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "<bos>" + msgs[0]["content"] + "\n"

        def __call__(self, text, add_special_tokens=False):
            ids = []
            for part in text.replace("<bos>", "\x00").split():
                ids.extend(2 if ch == "\x00" else ord(ch) for ch in part)
            return {"input_ids": ids}

    data.set_tokenizer(Tok())
    prompt = data.apply_chat("word " * 300 + "Answer:")
    ids, masks = data.tokenize_prompt_targets(Tok(), [prompt], [" B"], max_length=64)
    row = ids[0].tolist()
    assert row[0] == 2 and row.count(2) == 1 and len(row) == 64 and int(masks[0].sum()) == 1 and row[-1] == ord("B")
    monkeypatch.delenv("MECH_CHAT_TEMPLATE"); importlib.reload(data)


def test_hf_chat_template_adds_special_tokens_once(monkeypatch):
    """--chat-template renders the template once and tokenises it without adding a second BOS."""
    import torch

    class Tok:
        bos_token_id = 2
        pad_token_id = 0
        padding_side = "left"

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "<bos>[U]" + msgs[0]["content"] + "[/U]"

        def __call__(self, texts, return_tensors=None, padding=True, add_special_tokens=True):
            rows = []
            for t in texts:
                ids = [2 if tok == "<bos>" else len(tok) for tok in t.replace("<bos>", " <bos> ").split()]
                if add_special_tokens:
                    ids = [2] + ids
                rows.append(ids)
            width = max(map(len, rows))
            rows = [[0] * (width - len(r)) + r for r in rows]
            class Enc(dict):
                def to(self, device):
                    return self
            return Enc(input_ids=torch.tensor(rows), attention_mask=torch.ones(len(rows), width, dtype=torch.long))

        def decode(self, ids, skip_special_tokens=True):
            return "note"

    class Mdl:
        device = "cpu"

        def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, pad_token_id):
            assert (input_ids == 2).sum(dim=1).tolist() == [1] * input_ids.shape[0], "BOS must appear exactly once"
            return torch.cat([input_ids, torch.full((input_ids.shape[0], 1), 7)], dim=1)

    out = generate_notes.generate(Tok(), Mdl(), ["hello world", "hi"], 4, 2, chat_template=True)
    assert out == ["note", "note"]


def test_attribution_judge_runs_only_on_contaminated_distracted_notes(tmp_path, monkeypatch):
    calls = []

    def fake_call(provider, model, system, user, max_tokens=800, retries=4):
        calls.append(user)
        return '{"attribution": "third_party", "used_for_patient": 0, "reasoning": "recorded as the coworker\'s illness"}'

    monkeypatch.setattr(judge, "_call", fake_call)
    notes = _notes()
    paired = pd.DataFrame([{"source_dataset": "mts_dialog_test1", "item_id": iid, "distractor_type": "bystander", "condition": cond,
                            "contamination": int(cond == "distracted" and iid != "c")} for iid in ("a", "b", "c") for cond in ("clean", "distracted")])
    notes["distractor_summary"] = "coworker had pneumonia"
    out = judge.judge_attribution(notes, paired, _pairs(), "anthropic", "j", tmp_path / "attr.parquet", concurrency=2)
    assert sorted(out["item_id"]) == ["a", "b"] and (out["attribution"] == "correct").all() and len(calls) == 2
    again = judge.judge_attribution(notes, paired, _pairs(), "anthropic", "j", tmp_path / "attr.parquet", concurrency=2)
    assert len(again) == 2 and len(calls) == 2  # resumable: nothing re-judged
