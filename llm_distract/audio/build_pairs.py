"""Assemble the ambient-compatible transcript pairs: one row per A x donor x level (CPU).

The columns are those of data/ambient/pairs.parquet, so ``llm_distract.audio.generate_notes`` and
``llm_distract.ambient.judge`` consume the file unchanged: source_dataset='primock57', family='PriMock57', item_id (A),
distractor_type='audio_{level}dB_d{donor}', clean_transcript (Whisper on clean A), distracted_transcript (Whisper on
the mix), reference_note (A's clinician note), distractor_conversation (human transcript lines of the overlaid B
segment), distractor_summary (judge target: the same text prefixed by SUMMARY_PREFIX), distractor_topic (B's presenting
complaint); plus the design columns (donor, donor_id, level_db, onset, segment_*, wav) and A's human transcript.

    python -m llm_distract.audio.build_pairs [--manifest outputs/audio/mixes/manifest.parquet]
        [--transcripts outputs/audio/transcripts.parquet] [--root data/audio/primock57] [--out data/audio/pairs.parquet]
"""
from __future__ import annotations

import argparse

import pandas as pd

from llm_distract.ambient.common import write_parquet
from llm_distract.audio.data import load_corpus

SOURCE_DATASET, FAMILY = "primock57", "PriMock57"
SUMMARY_PREFIX = "Content from a different patient's consultation: "
COLUMNS = ["source_dataset", "family", "item_id", "distractor_type", "clean_transcript", "distracted_transcript",
           "reference_note", "distractor_conversation", "distractor_summary", "distractor_topic", "donor", "donor_id",
           "level_db", "onset", "segment_start", "segment_end", "segment_seconds", "segment_score", "wav", "human_transcript"]


def build_pairs(manifest: pd.DataFrame, transcripts: pd.DataFrame, consultations: pd.DataFrame) -> pd.DataFrame:
    text = transcripts.set_index(["item_id", "distractor_type"])["text"]
    missing = [k for k in zip(manifest["item_id"], manifest["distractor_type"]) if k not in text.index]
    if missing:
        raise RuntimeError(f"{len(missing)} manifest recordings have no transcript (first: {missing[0]}); rerun transcribe on the full manifest")
    cons = consultations.set_index("item_id")
    rows = []
    for r in manifest[manifest["condition"] == "distracted"].itertuples(index=False):
        rows.append({"source_dataset": SOURCE_DATASET, "family": FAMILY, "item_id": r.item_id, "distractor_type": r.distractor_type,
                     "clean_transcript": text[(r.item_id, "clean")], "distracted_transcript": text[(r.item_id, r.distractor_type)],
                     "reference_note": cons.loc[r.item_id, "note"], "distractor_conversation": r.segment_lines,
                     "distractor_summary": SUMMARY_PREFIX + r.segment_text, "distractor_topic": cons.loc[r.donor_id, "presenting_complaint"],
                     "donor": int(r.donor), "donor_id": r.donor_id, "level_db": float(r.level_db), "onset": float(r.onset),
                     "segment_start": float(r.segment_start), "segment_end": float(r.segment_end), "segment_seconds": float(r.segment_seconds),
                     "segment_score": int(r.segment_score), "wav": r.wav, "human_transcript": cons.loc[r.item_id, "human_transcript"]})
    pairs = pd.DataFrame(rows, columns=COLUMNS)
    if pairs.duplicated(["source_dataset", "item_id", "distractor_type"]).any():
        raise RuntimeError("pair keys are not unique")
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="outputs/audio/mixes/manifest.parquet")
    ap.add_argument("--transcripts", default="outputs/audio/transcripts.parquet")
    ap.add_argument("--root", default="data/audio/primock57")
    ap.add_argument("--out", default="data/audio/pairs.parquet")
    args = ap.parse_args()
    cons, _ = load_corpus(args.root)
    pairs = build_pairs(pd.read_parquet(args.manifest), pd.read_parquet(args.transcripts), cons)
    write_parquet(pairs, args.out, meta={"manifest": args.manifest, "transcripts": args.transcripts})
    print(f"wrote {len(pairs)} transcript pairs for {pairs['item_id'].nunique()} consultations to {args.out}")


if __name__ == "__main__":
    main()
