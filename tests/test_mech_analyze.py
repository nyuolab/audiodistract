"""analyze.py reproduces the verified mechanism numbers from results/raw."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llm_distract.mechanism import analyze

RAW = Path(__file__).resolve().parents[1] / "results" / "raw"
pytestmark = pytest.mark.skipif(not (RAW / "mech_masks.parquet").exists(), reason="results/raw/mech_*.parquet missing")


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    return analyze.run(RAW, tmp_path_factory.mktemp("results"))


def _row(df, model_short, distractor=None):
    sel = df["model_short"] == model_short
    if distractor:
        sel &= df["distractor"] == distractor
    return df[sel].iloc[0]


def test_sparsity_counts(tables):
    sp = tables["sparsity"]
    got = {(r.model_short, r.distractor): r.heads_selected for r in sp.itertuples()}
    assert got == {("Llama-3.1-8B", "nonliteral"): 17, ("Llama-3.1-8B", "bystander"): 15,
                   ("Qwen2.5-7B", "nonliteral"): 58, ("Qwen2.5-7B", "bystander"): 56,
                   ("Mistral-7B", "nonliteral"): 143, ("Mistral-7B", "bystander"): 160,
                   ("Gemma-2-9B", "nonliteral"): 146, ("Gemma-2-9B", "bystander"): 159}
    assert dict(zip(sp["model_short"], sp["total_heads"])) == {"Llama-3.1-8B": 1024, "Qwen2.5-7B": 784, "Mistral-7B": 1024, "Gemma-2-9B": 672}
    assert abs(_row(sp, "Gemma-2-9B", "bystander").pct_selected - 23.661) < 0.01


def test_jaccard(tables):
    j = tables["jaccard"].set_index("model_short")["jaccard"]
    assert j.round(3).to_dict() == {"Llama-3.1-8B": 0.778, "Qwen2.5-7B": 0.390, "Mistral-7B": 0.702, "Gemma-2-9B": 0.622}
    assert tables["jaccard"].set_index("model_short").loc["Llama-3.1-8B", "bystander_subset_of_nonliteral"] == False


def test_layer_thirds(tables):
    lay = tables["layers"]
    q = lay[(lay["model_short"] == "Qwen2.5-7B") & (lay["distractor"] == "nonliteral")].groupby("third")["n_selected"].sum()
    assert q.to_dict() == {"early": 29, "middle": 18, "late": 11}


def test_baseline_heldout_cohort(tables):
    b = tables["baseline"]
    assert (b["n"] == 127).all()
    llama = _row(b, "Llama-3.1-8B", "nonliteral")
    assert (round(llama.clean_acc, 1), round(llama.distracted_acc, 1), round(llama.drop_pp, 1)) == (66.1, 55.1, 11.0)
    gemma = _row(b, "Gemma-2-9B", "nonliteral")
    assert round(gemma.drop_pp, 1) == 11.0  # under the chat-template protocol every model is evaluated on the 127 held-out items only
    assert "n_all" not in b.columns or pd.isna(llama.n_all)


def test_patching_and_suppression(tables):
    p = tables["patching"]
    assert (p["n"] == 127).all() and (p["interference_delta_pp"] > 0).sum() == 7
    assert round(p["interference_delta_pp"].min(), 1) == 0.0 and round(p["interference_delta_pp"].max(), 1) == 16.5
    assert (p["interference_minus_random_pp"] > 0).sum() == 7 and (p["random_delta_pp"].abs() <= 3.2).all()
    assert round(_row(p, "Mistral-7B", "nonliteral").all_delta_pp, 1) == 14.2
    s = tables["suppression"]
    s0 = s.query("scale == 0.0 and head_set == 'interference'")
    drops = -s0["clean_delta_pp"]
    assert len(s0) == 8 and round(drops.min(), 1) == 24.4 and round(drops.max(), 1) == 37.8
    assert (s0["distracted_delta_pp"] < -14.9).all()
    r0 = s.query("scale == 0.0 and head_set == 'random'")
    assert len(r0) == 8 and (-r0["clean_delta_pp"]).max() < 9.9


def test_prompting_and_attention(tables):
    pr = tables["prompting"]
    assert (pr["n"] == 127).all()
    assert round(_row(pr, "Llama-3.1-8B", "nonliteral").ignore_delta_pp, 1) == 2.4
    assert round(_row(pr, "Qwen2.5-7B", "nonliteral").cot_delta_pp, 1) == -5.5
    at = tables["attention"].set_index(["model_short", "distractor"])["ratio"].round(2)
    assert at[("Llama-3.1-8B", "nonliteral")] == 2.26 and at[("Gemma-2-9B", "nonliteral")] == 1.36
