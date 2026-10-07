"""PriMock57 corpus (Babylon Health, CC BY 4.0): download, load consultations, select clinically dense donor segments.

Layout of https://github.com/babylonhealth/primock57 (57 mock GP consultations, days 1-5):
  audio/day{d}_consultation{NN}_{doctor,patient}.wav        16 kHz 16-bit mono, one channel per speaker (Git LFS)
  transcripts/day{d}_consultation{NN}_{doctor,patient}.TextGrid   manual utterance intervals (xmin, xmax, text)
  notes/day{d}_consultation{NN}.json                         clinician note (day, consultation, presenting_complaint, note)

Utterances of both speakers are merged in start-time order (the repository's ``textgrid_to_transcript.py`` rule) and
``<UNSURE>``/``<UNIN/>`` tags are stripped. A donor segment is the contiguous run of utterances spanning 5-15 s whose
text contains the most medical keywords (symptoms, diagnoses, medications, body parts; ties -> shorter, then earlier).
A consultation whose TextGrids are absent keeps its audio and note but has no utterances (human_transcript ''); the
timestamped units then come from Whisper (``whisper_units``; ``mix --clean-only`` -> ``transcribe`` -> ``mix --whisper-units``).

    python -m llm_distract.audio.data [--root data/audio/primock57] [--download]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from llm_distract.ambient.common import load_config

REPO_URL = "https://github.com/babylonhealth/primock57.git"
SPEAKERS = ("doctor", "patient")
UNIT_COLUMNS = ["item_id", "speaker", "start", "end", "text"]
INTERVAL_RE = re.compile(r'intervals \[\d+\]:\s*xmin = (\S+)\s*xmax = (\S+)\s*text = "((?:[^"]|"")*)"')
TAG_RE = re.compile(r"</?UNSURE>|<UNIN/>|<INAUDIBLE_SPEECH/>")
WORD_RE = re.compile(r"[a-z]+")
KEYWORDS = frozenset("""
pain ache aching headache migraine fever feverish chill cough coughing wheeze wheezing wheezy breath breathless breathing
shortness dyspnoea nausea nauseous vomit vomiting sick diarrhoea diarrhea constipation constipated bleeding blood bloody
dizzy dizziness faint fainting fatigue tired tiredness lethargic lethargy weakness numb numbness tingling swelling swollen
rash itchy itching sweat sweating palpitation cramp cramping discharge burning stiff stiffness sore tender tenderness
seizure tremor insomnia anxiety anxious depressed depression appetite sputum phlegm mucus hives lump bruise bruising
dehydrated dehydration infection infected infectious asthma asthmatic pneumonia bronchitis diabetes diabetic hypertension
angina stroke cancer tumour tumor anaemia anemia anaemic eczema psoriasis arthritis gastroenteritis reflux heartburn
ulcer allergy allergic allergies cystitis thrush tonsillitis sinusitis flu influenza covid coronavirus virus viral bacterial
thyroid hernia appendicitis gout epilepsy concussion fracture fractured sprain sprained sciatica dermatitis conjunctivitis
paracetamol acetaminophen ibuprofen aspirin antibiotic antibiotics amoxicillin penicillin inhaler salbutamol ventolin
steroid steroids prednisolone antihistamine cetirizine loratadine omeprazole lansoprazole metformin insulin statin
tablet tablets medication medications medicine medicines prescription prescribed dose dosage cream ointment
codeine morphine naproxen diclofenac gaviscon laxative loperamide imodium dioralyte contraceptive
chest stomach abdomen abdominal tummy belly spine neck throat ear ears eye eyes nose sinus sinuses mouth tooth teeth gum
gums tongue lung lungs heart liver kidney kidneys bladder bowel bowels stool stools urine urinary skin joint joints knee
knees hip hips shoulder shoulders elbow wrist finger fingers leg legs ankle ankles foot feet toe toes groin pelvis muscle
muscles bone bones gland glands lymph breast testicle pregnancy pregnant menstrual
temperature pulse symptom symptoms diagnosis examination scan xray ultrasound referral specialist hospital emergency
""".split())
"""Compact clinical keyword list used only to rank candidate donor windows (plurals are matched by stripping a final s)."""


def clean_text(text: str) -> str:
    """Strip transcriber tags and collapse whitespace (the repository's ``strip_transcript_tags``)."""
    return re.sub(r"\s+", " ", TAG_RE.sub("", str(text))).strip()


def clinical_score(text: str) -> int:
    """Number of keyword tokens in ``text``."""
    return sum(1 for t in WORD_RE.findall(str(text).lower()) if t in KEYWORDS or (t.endswith("s") and t[:-1] in KEYWORDS))


def parse_textgrid(path: str | Path) -> list:
    """Non-empty intervals of a Praat TextGrid (long format) as dicts with start, end, text (tags stripped)."""
    out = []
    for m in INTERVAL_RE.finditer(Path(path).read_text(encoding="utf-8")):
        text = clean_text(m.group(3).replace('""', '"'))
        if text:
            out.append({"start": float(m.group(1)), "end": float(m.group(2)), "text": text})
    return out


def load_corpus(root: str | Path) -> tuple:
    """(consultations, utterances): one row per consultation / one row per timestamped utterance of either speaker.

    consultations: item_id, doctor_wav, patient_wav, note, presenting_complaint, human_transcript (``Speaker: text`` lines).
    utterances: item_id, speaker, start, end, text, sorted by start time within consultation (missing TextGrids are skipped).
    """
    root = Path(root)
    cons, utts = [], []
    for note_path in sorted((root / "notes").glob("day*_consultation*.json")):
        item_id = note_path.stem
        meta = json.loads(note_path.read_text(encoding="utf-8"))
        row = {"item_id": item_id, "note": str(meta["note"]), "presenting_complaint": str(meta.get("presenting_complaint", ""))}
        for sp in SPEAKERS:
            row[f"{sp}_wav"] = str(root / "audio" / f"{item_id}_{sp}.wav")
            grid = root / "transcripts" / f"{item_id}_{sp}.TextGrid"
            for u in (parse_textgrid(grid) if grid.exists() else []):
                utts.append({"item_id": item_id, "speaker": sp.capitalize(), **u})
        cons.append(row)
    if not cons:
        raise RuntimeError(f"no consultations under {root}; run with --download")
    utterances = pd.DataFrame(utts, columns=UNIT_COLUMNS).sort_values(["item_id", "start", "end"], kind="stable").reset_index(drop=True)
    consultations = pd.DataFrame(cons)
    consultations["human_transcript"] = [segment_lines(utterances[utterances["item_id"] == i]) for i in consultations["item_id"]]
    return consultations, utterances


def segment_lines(units: pd.DataFrame) -> str:
    """``Speaker: text`` lines (plain text lines when the units carry no speaker, e.g. Whisper chunks)."""
    if "speaker" in units and units["speaker"].astype(str).str.len().gt(0).all():
        return "\n".join(f"{s}: {t}" for s, t in zip(units["speaker"], units["text"]))
    return "\n".join(units["text"])


def select_segment(units: pd.DataFrame, min_seconds: float, max_seconds: float) -> dict:
    """Contiguous run of units (sorted by start) spanning [min_seconds, max_seconds] with the most keywords.

    Span = max end - first start (utterances of the two speakers may overlap). Ties: shorter span, then earlier start.
    """
    units = units.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
    starts, ends = units["start"].to_numpy(float), units["end"].to_numpy(float)
    scores = np.array([clinical_score(t) for t in units["text"]], dtype=int)
    best = None
    for i in range(len(units)):
        end = 0.0
        for j in range(i, len(units)):
            end = max(end, ends[j])
            span = end - starts[i]
            if span > max_seconds:
                break
            if span < min_seconds:
                continue
            key = (int(scores[i:j + 1].sum()), -span, -starts[i])
            if best is None or key > best[0]:
                best = (key, i, j, end)
    if best is None:
        raise ValueError("no timestamped units" if units.empty else f"no run of utterances spans {min_seconds}-{max_seconds} s")
    (score, _, _), i, j, end = best
    run = units.iloc[i:j + 1]
    return {"start": float(starts[i]), "end": float(end), "seconds": float(end - starts[i]), "score": score,
            "n_units": len(run), "text": " ".join(run["text"]), "lines": segment_lines(run)}


def assign_donors(item_ids, n_donors: int, seed: int) -> pd.DataFrame:
    """Seeded donors B != A for every A: columns item_id, donor (0..n_donors-1), donor_id (no rows for n_donors 0)."""
    ids = sorted(str(i) for i in item_ids)
    if n_donors and len(ids) <= n_donors:
        raise ValueError(f"{n_donors} donors per consultation need at least {n_donors + 1} consultations, got {len(ids)}")
    rng = np.random.default_rng([seed, 1])
    rows = []
    for a in ids:
        for k, b in enumerate(rng.choice([b for b in ids if b != a], size=n_donors, replace=False)):
            rows.append({"item_id": a, "donor": k, "donor_id": str(b)})
    return pd.DataFrame(rows, columns=["item_id", "donor", "donor_id"])


def whisper_units(transcripts: pd.DataFrame) -> pd.DataFrame:
    """Timestamped Whisper chunks of the clean recordings (``transcribe.py`` output) in the utterance-table layout."""
    rows = []
    for r in transcripts[transcripts["distractor_type"] == "clean"].itertuples(index=False):
        for c in json.loads(r.segments):
            if c.get("end") is not None and str(c.get("text", "")).strip():
                rows.append({"item_id": r.item_id, "speaker": "", "start": float(c["start"]), "end": float(c["end"]),
                             "text": clean_text(c["text"])})
    return pd.DataFrame(rows, columns=UNIT_COLUMNS).sort_values(["item_id", "start", "end"], kind="stable").reset_index(drop=True)


def audio_present(root: Path) -> bool:
    """True when the first audio file is a real WAV rather than a Git LFS pointer."""
    wavs = sorted((root / "audio").glob("*.wav"))
    return bool(wavs) and wavs[0].open("rb").read(4) == b"RIFF"


def download(root: str | Path, url: str = REPO_URL) -> Path:
    """Clone PriMock57 into ``root`` (skipped when present) and fetch the LFS audio; raises if audio stays unavailable."""
    root = Path(root)
    if not (root / "notes").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, str(root)], check=True)
    if not audio_present(root):
        subprocess.run(["git", "-C", str(root), "lfs", "pull"], check=False)
    if not audio_present(root):
        raise RuntimeError(f"{root}/audio holds Git LFS pointers; install git-lfs and run: git -C {root} lfs pull")
    return root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/audio/primock57")
    ap.add_argument("--download", action="store_true", help="git clone (+ git lfs pull) the corpus if absent")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)["audio"]
    if args.download:
        download(args.root, cfg.get("repo_url", REPO_URL))
    cons, utts = load_corpus(args.root)
    lo, hi = cfg["segment_seconds"]
    timed = utts["item_id"].nunique()
    print(f"{len(cons)} consultations, {timed} with human TextGrids, {len(utts)} utterances")
    if timed < len(cons):
        print(f"{len(cons) - timed} without timestamps: mix --clean-only, transcribe, then mix --whisper-units transcripts.parquet")
    if timed:
        segs = pd.DataFrame([select_segment(g, lo, hi) for _, g in utts.groupby("item_id")])
        dur = utts.groupby("item_id")["end"].max()
        print(f"{dur.sum() / 3600:.2f} h transcribed (mean {dur.mean() / 60:.1f} min); best {lo}-{hi} s window: mean "
              f"{segs['seconds'].mean():.1f} s, {segs['score'].mean():.1f} keywords (min {segs['score'].min()}, max {segs['score'].max()})")


if __name__ == "__main__":
    main()
