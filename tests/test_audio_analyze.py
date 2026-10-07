"""analyze/plot on synthetic judge and leakage artifacts with a known effect structure."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from llm_distract.audio import analyze, plot  # noqa: E402

MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
LEVELS = [-20, -15, -10, -5]
ITEMS = [f"day1_consultation{i:02d}" for i in range(1, 7)]


def _runs(root):
    """outputs/<model>/judge_*.parquet: donor 0 contaminated at >= -10 dB, donor 1 never; clean notes never."""
    rng = np.random.default_rng(0)
    for model in MODELS:
        c, q = [], []
        for item in ITEMS:
            for donor in (0, 1):
                for level in LEVELS:
                    dt = f"audio_{level}dB_d{donor}"
                    for cond in ("clean", "distracted"):
                        hit = int(cond == "distracted" and donor == 0 and level >= -10)
                        c.append({"source_dataset": "primock57", "item_id": item, "distractor_type": dt, "model": model, "condition": cond,
                                  "contamination": hit, "severity": 2 * hit, "shown_as": "A", "judge_reasoning": "", "judge_raw": "",
                                  "parse_error": False, "judge_model": "claude-sonnet-5", "judge_provider": "anthropic",
                                  "judge_protocol": "v3_paired_symmetric", "seed": 0, "timestamp": ""})
                        q.append({"source_dataset": "primock57", "item_id": item, "distractor_type": dt, "model": model, "condition": cond,
                                  "clinical_correctness": 4, "completeness": 3 + hit, "succinctness": 4, "hallucination": int(rng.uniform() < 0.9),
                                  "overall_quality": 4, "judge_reasoning": "", "parse_error": False})
        d = root / model.replace("/", "__")
        d.mkdir(parents=True)
        pd.DataFrame(c).to_parquet(d / "judge_contamination_paired.parquet", index=False)
        pd.DataFrame(q).to_parquet(d / "judge_quality.parquet", index=False)
    (root / "other").mkdir()   # a run directory without judge outputs is ignored


def _leakage():
    rows = []
    for item in ITEMS:
        for donor in (0, 1):
            for level in LEVELS:
                leak = 1.0 if level >= -10 else 0.0
                rows.append({"item_id": item, "distractor_type": f"audio_{level}dB_d{donor}", "donor": donor, "donor_id": "x", "level_db": float(level),
                             "onset": 100.0, "segment_start": 10.0, "segment_end": 20.0, "segment_seconds": 10.0, "segment_score": 6,
                             "n_content": 20, "n_leaked": int(20 * leak), "leakage": leak, "overlap_clean": 0.1, "leaked_words": "",
                             "wer_vs_clean": 0.05 - level / 1000, "wer_clean_vs_human": 0.2})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    raw, runs, out = (tmp_path_factory.mktemp(n) for n in ("raw", "runs", "results"))
    _runs(runs)
    j = analyze.collect(runs)
    assert len(j) == 2 * 6 * 2 * 4 * 2 and set(j["family"]) == {"PriMock57"} and "overall_quality" in j
    j.to_parquet(raw / "audio_judgments.parquet", index=False)
    _leakage().to_parquet(raw / "audio_leakage.parquet", index=False)
    return analyze.run(raw, out), out


def _row(df: pd.DataFrame, **keys) -> pd.Series:
    """The single row matching all column == value pairs (level_db mixes ints and 'all')."""
    mask = np.ones(len(df), bool)
    for k, v in keys.items():
        mask &= np.array([x == v for x in df[k]])
    assert mask.sum() == 1, keys
    return df[mask].iloc[0]


def test_contamination_table_recovers_design(tables):
    t, _ = tables
    c = t["contamination_by_level_model"]
    p = _row(c, level_db=-10, model="all")
    assert p["delta_pp"] == pytest.approx(50.0) and p["n_pairs"] == 24 and p["n_encounters"] == 6 and p["rate_clean"] == 0.0
    assert p["ci_low"] == pytest.approx(50.0) and p["ci_high"] == pytest.approx(50.0)   # balanced design: no encounter variance
    llama = _row(c, level_db=-20, model="Llama-3.1-8B")
    assert llama["delta_pp"] == 0.0 and llama["ci_high"] == 0.0
    assert _row(c, level_db="all", model="all")["rate_mixed"] == pytest.approx(25.0)
    assert _row(c, level_db=-5, model="Qwen2.5-7B")["severity_mixed"] == pytest.approx(1.0)
    assert list(c["model"].unique()) == ["all", "Llama-3.1-8B", "Qwen2.5-7B"] and len(c) == 5 * 3


def test_leakage_and_conditional_tables(tables):
    t, _ = tables
    lk = t["leakage_by_level"].set_index("level_db")
    assert lk.loc[-20, "leakage_pct"] == 0.0 and lk.loc[-5, "leakage_pct"] == 100.0 and lk.loc[-10, "any_leak_pct"] == 100.0
    assert lk.loc[-20, "n_mixes"] == 12 and lk.loc[-20, "overlap_clean_pct"] == pytest.approx(10.0)
    cg = t["contamination_given_leakage"]
    assert _row(cg, level_db=-10, leaked=True)["rate_mixed"] == pytest.approx(50.0) and _row(cg, level_db=-20, leaked=False)["rate_mixed"] == 0.0
    assert _row(cg, level_db=-10, leaked=True)["n_pairs"] == 24 and len(cg) == 4   # one stratum per level in this design


def test_quality_table_and_summary(tables):
    t, _ = tables
    q = t["quality"]
    assert _row(q, level_db="all", metric="completeness")["delta"] == pytest.approx(0.25)
    assert _row(q, level_db=-20, metric="completeness")["delta"] == 0.0
    assert _row(q, level_db="all", metric="hallucination")["scale"] == "0/1" and _row(q, level_db="all", metric="clinical_correctness")["delta"] == 0.0
    s = analyze.summarize(t)
    assert s.startswith("primary (-10 dB, pooled over models, 6 encounters, 24 pairs): clean 0.0% -> mixed 50.0%; delta 50.0 pp")


def test_parse_distractor():
    assert analyze.parse_distractor("audio_-10dB_d1") == (-10, 1) and analyze.parse_distractor("audio_5dB_d0") == (5, 0)
    with pytest.raises(ValueError):
        analyze.parse_distractor("bystander")


def test_plot_writes_figure(tables, tmp_path):
    _, out = tables
    paths = plot.ed_audio(out, tmp_path)
    assert [p.suffix for p in paths] == [".pdf", ".png"] and all(p.stat().st_size > 1000 for p in paths)


def test_analyze_without_quality_judge(tmp_path):
    """Contamination judged, quality judge not yet run: tables and summary still work."""
    raw, runs, out = tmp_path / "raw", tmp_path / "runs", tmp_path / "results"
    _runs(runs)
    for q in runs.glob("*/judge_quality.parquet"):
        q.unlink()
    j = analyze.collect(runs)
    assert "overall_quality" not in j
    j.to_parquet(raw.mkdir() or raw / "audio_judgments.parquet", index=False)
    _leakage().to_parquet(raw / "audio_leakage.parquet", index=False)
    t = analyze.run(raw, out)
    assert t["quality"].empty and "level_db" in t["quality"].columns and (out / "audio_quality.csv").exists()
    assert "quality deltas" not in analyze.summarize(t) and _row(t["contamination_by_level_model"], level_db=-10, model="all")["delta_pp"] == pytest.approx(50.0)
    with pytest.raises(RuntimeError, match="no encounter"):
        j.assign(contamination=None).to_parquet(raw / "audio_judgments.parquet", index=False)
        analyze.run(raw, out)
