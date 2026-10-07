"""CPU tests for llm_distract.audio.mix: RMS-relative gain, overlay, peak normalisation, WAV I/O, seeded plan, render."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_distract.audio import mix
from llm_distract.audio.data import load_corpus
from test_audio_data import synthetic_corpus

LEVELS = [-20, -15, -10, -5]


def _db(bg: np.ndarray, fg: np.ndarray) -> float:
    return 20 * np.log10(mix.rms(bg) / mix.rms(fg))


def test_gain_achieves_requested_level():
    rng = np.random.default_rng(0)
    fg, bg = rng.normal(0, 0.1, 16000).astype(np.float32), rng.normal(0, 0.3, 4000).astype(np.float32)
    for level in LEVELS:
        g = mix.gain_for_level(fg, bg, level)
        assert _db(g * bg, fg) == pytest.approx(level, abs=1e-6)
    with pytest.raises(ValueError):
        mix.gain_for_level(fg, np.zeros(10, np.float32), -10)


def test_overlay_truncates_and_peak_normalises():
    fg, bg = np.zeros(100, np.float32), np.ones(10, np.float32)
    out = mix.overlay(fg, bg, 95, 0.5)
    assert len(out) == 100 and np.allclose(out[95:], 0.5) and np.allclose(out[:95], 0)
    assert np.max(np.abs(mix.peak_normalize(out, 0.95))) == pytest.approx(0.95)
    assert np.array_equal(mix.peak_normalize(fg, 0.95), fg)


def test_wav_roundtrip_and_resample(tmp_path):
    x = np.linspace(-1, 1, 44100).astype(np.float32)
    mix.write_wav(tmp_path / "a.wav", x, 44100)
    y, sr = mix.read_wav(tmp_path / "a.wav")
    assert sr == 44100 and np.max(np.abs(x - y)) <= 1 / 32768 and mix.wav_seconds(tmp_path / "a.wav") == pytest.approx(1.0)
    assert len(mix.resample(y, 44100, 16000)) == 16000
    assert mix.resample(y, 16000, 16000) is y


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = synthetic_corpus(tmp_path_factory.mktemp("primock57"))
    cons, utts = load_corpus(root)
    cons["seconds"] = [max(mix.wav_seconds(d), mix.wav_seconds(p)) for d, p in zip(cons["doctor_wav"], cons["patient_wav"])]
    return cons, utts


def test_plan_layout_and_determinism(corpus):
    cons, utts = corpus
    m = mix.plan(cons, utts, LEVELS, 2, (5, 15), (0.2, 0.8), seed=0)
    assert len(m) == 3 + 3 * 2 * 4 and (m["condition"] == "clean").sum() == 3
    mixed = m[m["condition"] == "distracted"]
    assert not mixed.duplicated(["item_id", "distractor_type"]).any()
    assert set(mixed["distractor_type"]) == {f"audio_{lv}dB_d{k}" for lv in LEVELS for k in (0, 1)}
    assert (mixed["donor_id"] != mixed["item_id"]).all() and mixed["segment_seconds"].between(5, 15).all()
    assert mixed["onset"].between(0.2 * 40, 0.8 * 40).all() and (mixed["onset"] + mixed["segment_seconds"] <= 40).all()
    assert mixed.groupby(["item_id", "donor"])["onset"].nunique().eq(1).all()      # one onset per (A, donor), shared by levels
    pd.testing.assert_frame_equal(m, mix.plan(cons, utts, LEVELS, 2, (5, 15), (0.2, 0.8), seed=0))
    other = mix.plan(cons, utts, LEVELS, 2, (5, 15), (0.2, 0.8), seed=1)
    assert not np.allclose(mixed["onset"].values, other[other["condition"] == "distracted"]["onset"].values)


def test_render_reproduces_mix_and_level(corpus, tmp_path):
    cons, utts = corpus
    m = mix.render(mix.plan(cons, utts, LEVELS, 2, (5, 15), (0.2, 0.8), seed=0), cons, tmp_path, 16000, 0.95)
    assert all(mix.wav_seconds(p) == pytest.approx(40.0) for p in m["wav"])
    r = m[m["condition"] == "distracted"].iloc[5]
    assert _db(r["gain"] * np.ones(1), np.ones(1)) + _db(np.ones(1) * r["rms_bg"], np.ones(1) * r["rms_fg"]) == pytest.approx(r["level_db"], abs=1e-6)
    clean, _ = mix.read_wav(m[(m["item_id"] == r["item_id"]) & (m["condition"] == "clean")]["wav"].iloc[0])
    donor = cons.set_index("item_id").loc[r["donor_id"]]
    seg = mix.two_speaker_track(donor["doctor_wav"], donor["patient_wav"], 16000)[int(round(r["segment_start"] * 16000)):int(round(r["segment_end"] * 16000))]
    expected = mix.peak_normalize(mix.overlay(clean, seg, int(round(r["onset"] * 16000)), r["gain"]), 0.95)
    got, _ = mix.read_wav(r["wav"])
    assert np.max(np.abs(got - expected)) < 3e-4 and np.max(np.abs(got)) == pytest.approx(0.95, abs=1e-4)


def test_silent_foreground_raises():
    with pytest.raises(ValueError, match="silent foreground"):
        mix.gain_for_level(np.zeros(10, np.float32), np.ones(10, np.float32), -10)


def test_plan_clean_only_short_foreground_and_missing_donor_units(corpus):
    cons, utts = corpus
    clean = mix.plan(cons, utts, LEVELS, 0, (5, 15), (0.2, 0.8), seed=0)            # --clean-only
    assert len(clean) == 3 and (clean["condition"] == "clean").all() and list(clean.columns) == mix.MANIFEST
    short = cons.copy()
    short.loc[short["item_id"] == "day1_consultation01", "seconds"] = 12.0          # shorter than the 15-s segment
    m = mix.plan(short, utts, LEVELS, 2, (5, 15), (0.2, 0.8), seed=0)
    r = m[(m["item_id"] == "day1_consultation01") & (m["condition"] == "distracted")]
    assert np.allclose(r["onset"], 0.2 * 12.0)                                        # as far left as allowed; overlay truncates the rest
    with pytest.raises(ValueError, match="donor day1_consultation0. of day1_consultation01: no timestamped units"):
        mix.plan(cons, utts[utts["item_id"] == "day1_consultation01"], LEVELS, 2, (5, 15), (0.2, 0.8), seed=0)
