"""Overlay a clinically dense segment of donor consultation B on foreground consultation A at fixed levels (CPU, numpy).

For every consultation A: ``donors_per_consultation`` seeded donors B != A, one 5-15 s segment per donor
(``data.select_segment``), one seeded onset per (A, donor) drawn uniformly within the middle 60% of A (shifted left only
when the segment would run past the end), and one mix per level in ``levels_db`` (background RMS relative to the RMS of
A's whole track). Every mix is peak-normalised to ``peak``; clean = A's two-speaker track unmodified. The two-speaker
track is the mean of the doctor and patient channels (the repository's ``sox -m``), resampled to 16 kHz mono if needed.

    python -m llm_distract.audio.mix [--root data/audio/primock57] [--out outputs/audio/mixes] [--seed 0]
        [--whisper-units outputs/audio/transcripts.parquet]   # use Whisper timestamps of clean B instead of the TextGrids

Fallback without human timestamps: ``mix --clean-only`` (clean tracks + manifest only) -> ``transcribe`` ->
``mix --whisper-units transcripts.parquet`` -> ``transcribe`` again (resumes; the clean rows are already done).
"""
from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from llm_distract.ambient.common import load_config, write_parquet
from llm_distract.audio.data import assign_donors, load_corpus, select_segment, whisper_units

MANIFEST = ["item_id", "distractor_type", "condition", "donor", "donor_id", "level_db", "onset", "segment_start",
            "segment_end", "segment_seconds", "segment_score", "segment_text", "segment_lines", "seed"]


def read_wav(path: str | Path) -> tuple:
    """16-bit PCM WAV -> (float32 mono in [-1, 1], sample rate); multi-channel files are averaged to mono."""
    with wave.open(str(path), "rb") as f:
        n_ch, width, sr, frames = f.getnchannels(), f.getsampwidth(), f.getframerate(), f.readframes(f.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported (got {8 * width}-bit)")
    x = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return (x.reshape(-1, n_ch).mean(axis=1) if n_ch > 1 else x), sr


def write_wav(path: str | Path, x: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())
    return path


def wav_seconds(path: str | Path) -> float:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x
    g = math.gcd(sr, target)
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def two_speaker_track(doctor_wav: str | Path, patient_wav: str | Path, sr: int) -> np.ndarray:
    """Mean of the two speaker channels at ``sr`` (the shorter channel is zero-padded)."""
    d, p = (resample(*read_wav(w), sr) for w in (doctor_wav, patient_wav))
    n = max(len(d), len(p))
    return (np.pad(d, (0, n - len(d))) + np.pad(p, (0, n - len(p)))) / 2.0


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def gain_for_level(foreground: np.ndarray, background: np.ndarray, level_db: float) -> float:
    """Scalar gain that puts the background ``level_db`` dB relative to the foreground RMS."""
    f, r = rms(foreground), rms(background)
    if f == 0.0 or r == 0.0:
        raise ValueError("silent " + ("foreground" if f == 0.0 else "background segment"))
    return 10 ** (level_db / 20.0) * f / r


def overlay(foreground: np.ndarray, background: np.ndarray, onset: int, gain: float) -> np.ndarray:
    """foreground + gain * background starting at sample ``onset`` (truncated at the end of the foreground)."""
    out = foreground.astype(np.float32).copy()
    n = max(0, min(len(background), len(out) - onset))
    out[onset:onset + n] += gain * background[:n]
    return out


def peak_normalize(x: np.ndarray, peak: float) -> np.ndarray:
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    return x * (peak / m) if m > 0 else x


def plan(consultations: pd.DataFrame, units: pd.DataFrame, levels_db, n_donors: int, segment_seconds, onset_fraction,
         seed: int) -> pd.DataFrame:
    """Manifest rows (no audio written): one clean row per A and one row per A x donor x level (clean rows only for n_donors 0).

    ``consultations`` needs a ``seconds`` column (duration of A); ``units`` are timestamped utterances of every
    consultation. The onset and the donor segment are shared by the levels of one (A, donor).
    """
    donors = assign_donors(consultations["item_id"], n_donors, seed)
    rng = np.random.default_rng([seed, 2])
    lo_frac, hi_frac = onset_fraction
    seconds = consultations.set_index("item_id")["seconds"]
    rows = []
    for a in sorted(consultations["item_id"].astype(str)):
        rows.append({"item_id": a, "distractor_type": "clean", "condition": "clean", "donor": -1, "donor_id": "",
                     "level_db": np.nan, "onset": np.nan, "segment_start": np.nan, "segment_end": np.nan, "segment_seconds": np.nan,
                     "segment_score": np.nan, "segment_text": "", "segment_lines": "", "seed": seed})
        for d in donors[donors["item_id"] == a].itertuples(index=False):
            try:
                seg = select_segment(units[units["item_id"] == d.donor_id], *segment_seconds)
            except ValueError as exc:
                raise ValueError(f"donor {d.donor_id} of {a}: {exc}") from None
            dur = float(seconds[a])
            lo, hi = lo_frac * dur, max(lo_frac * dur, min(hi_frac * dur, dur - seg["seconds"]))
            onset = lo + float(rng.uniform()) * (hi - lo)
            for level in levels_db:
                rows.append({"item_id": a, "distractor_type": f"audio_{int(level)}dB_d{d.donor}", "condition": "distracted",
                             "donor": int(d.donor), "donor_id": d.donor_id, "level_db": float(level), "onset": onset,
                             "segment_start": seg["start"], "segment_end": seg["end"], "segment_seconds": seg["seconds"],
                             "segment_score": seg["score"], "segment_text": seg["text"], "segment_lines": seg["lines"], "seed": seed})
    return pd.DataFrame(rows, columns=MANIFEST)


def render(manifest: pd.DataFrame, consultations: pd.DataFrame, out_dir: str | Path, sr: int, peak: float) -> pd.DataFrame:
    """Write clean/<A>.wav and mixed/<A>__<distractor_type>.wav; returns the manifest with wav paths and achieved gains."""
    out_dir = Path(out_dir)
    cons = consultations.set_index("item_id")
    track = lambda i: two_speaker_track(cons.loc[i, "doctor_wav"], cons.loc[i, "patient_wav"], sr)  # noqa: E731
    rows = []
    for item_id, group in manifest.groupby("item_id", sort=False):
        fg = track(item_id)
        for r in group.to_dict("records"):
            if r["condition"] == "clean":
                path = out_dir / "clean" / f"{item_id}.wav"
                write_wav(path, fg, sr)
                r.update({"rms_fg": rms(fg), "rms_bg": np.nan, "gain": np.nan, "peak_scale": np.nan})
            else:
                seg = track(r["donor_id"])[int(round(r["segment_start"] * sr)):int(round(r["segment_end"] * sr))]
                gain = gain_for_level(fg, seg, r["level_db"])
                raw = overlay(fg, seg, int(round(r["onset"] * sr)), gain)
                path = out_dir / "mixed" / f"{item_id}__{r['distractor_type']}.wav"
                write_wav(path, peak_normalize(raw, peak), sr)
                r.update({"rms_fg": rms(fg), "rms_bg": rms(seg), "gain": gain, "peak_scale": peak / float(np.max(np.abs(raw)))})
            r["wav"] = str(path)
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/audio/primock57")
    ap.add_argument("--out", default="outputs/audio/mixes")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=None, help="default: audio.seed in the config")
    ap.add_argument("--whisper-units", default=None, help="transcripts.parquet: use Whisper chunk timestamps of clean B")
    ap.add_argument("--clean-only", action="store_true", help="write the clean tracks and manifest only (before --whisper-units)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N consultations (smoke test)")
    args = ap.parse_args()
    cfg = load_config(args.config)["audio"]
    seed = cfg["seed"] if args.seed is None else args.seed
    cons, utts = load_corpus(args.root)
    if args.limit:
        cons = cons.head(args.limit)
        utts = utts[utts["item_id"].isin(cons["item_id"])]
    if args.whisper_units:
        utts = whisper_units(pd.read_parquet(args.whisper_units))
    cons["seconds"] = [max(wav_seconds(d), wav_seconds(p)) for d, p in zip(cons["doctor_wav"], cons["patient_wav"])]
    n_donors = 0 if args.clean_only else cfg["donors_per_consultation"]
    manifest = plan(cons, utts, cfg["levels_db"], n_donors, cfg["segment_seconds"], cfg["onset_fraction"], seed)
    manifest = render(manifest, cons, args.out, cfg["sample_rate"], cfg["peak"])
    meta = {"seed": seed, "levels_db": cfg["levels_db"], "donors_per_consultation": n_donors,
            "segment_seconds": cfg["segment_seconds"], "onset_fraction": cfg["onset_fraction"], "peak": cfg["peak"],
            "timestamps": "none" if args.clean_only else "whisper" if args.whisper_units else "human_textgrid", "root": args.root}
    path = write_parquet(manifest, Path(args.out) / "manifest.parquet", meta=meta)
    n_mixed = int((manifest["condition"] == "distracted").sum())
    print(f"wrote {n_mixed} mixes + {len(manifest) - n_mixed} clean tracks to {args.out}; manifest {path}")


if __name__ == "__main__":
    main()
