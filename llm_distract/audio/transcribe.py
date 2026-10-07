"""Transcribe the clean and mixed recordings with Whisper large-v3 (GPU; open weights; greedy; English; no diarization).

transformers ``automatic-speech-recognition`` pipeline at ``chunk_length_s`` 30 with ``return_timestamps=True``
(segment timestamps; ``--word-timestamps`` requests word-level ones), identical settings for clean and mixed audio.
Audio is decoded with the stdlib WAV reader (no ffmpeg). Resumable: rows already in ``--out`` are skipped.

    python -m llm_distract.audio.transcribe --manifest outputs/audio/mixes/manifest.parquet --out outputs/audio/transcripts.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from llm_distract.ambient.common import load_config, write_parquet
from llm_distract.audio.mix import read_wav, resample

GENERATE_KWARGS = {"language": "english", "task": "transcribe", "num_beams": 1, "do_sample": False}
COLUMNS = ["item_id", "distractor_type", "condition", "wav", "text", "segments", "asr_model", "timestamps", "seconds"]


def load_asr(model_id: str, chunk_length_s: int):
    """Whisper ASR pipeline on the first GPU (fp16) or CPU (fp32)."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline("automatic-speech-recognition", model=model_id, chunk_length_s=chunk_length_s, device=device,
                    torch_dtype=torch.float16 if device == 0 else torch.float32)


def transcribe_one(asr, wav: str | Path, sr: int, word_timestamps: bool) -> tuple:
    """(text, chunks) for one recording; chunks are dicts with start, end (None when open-ended) and text."""
    x, file_sr = read_wav(wav)
    out = asr({"raw": resample(x, file_sr, sr), "sampling_rate": sr}, return_timestamps="word" if word_timestamps else True,
              generate_kwargs=GENERATE_KWARGS)
    chunks = [{"start": c["timestamp"][0], "end": c["timestamp"][1], "text": str(c["text"]).strip()} for c in out.get("chunks", [])]
    return str(out["text"]).strip(), chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="outputs/audio/mixes/manifest.parquet")
    ap.add_argument("--out", default="outputs/audio/transcripts.parquet")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None, help="default: audio.whisper_model in the config")
    ap.add_argument("--word-timestamps", action="store_true", help="return_timestamps='word' instead of segment timestamps")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)["audio"]
    model_id = args.model or cfg["whisper_model"]
    manifest = pd.read_parquet(args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit)
    out = Path(args.out)
    rows = pd.read_parquet(out).to_dict("records") if out.exists() else []
    done = {(r["item_id"], r["distractor_type"]) for r in rows}
    todo = manifest[[(i, d) not in done for i, d in zip(manifest["item_id"], manifest["distractor_type"])]]
    meta = {"asr_model": model_id, "chunk_length_s": cfg["whisper_chunk_length_s"], "generate_kwargs": GENERATE_KWARGS,
            "timestamps": "word" if args.word_timestamps else "segment", "manifest": args.manifest}
    asr = load_asr(model_id, cfg["whisper_chunk_length_s"]) if len(todo) else None
    for k, r in enumerate(todo.itertuples(index=False), 1):
        t0 = time.time()
        text, chunks = transcribe_one(asr, r.wav, cfg["sample_rate"], args.word_timestamps)
        rows.append({"item_id": r.item_id, "distractor_type": r.distractor_type, "condition": r.condition, "wav": r.wav,
                     "text": text, "segments": json.dumps(chunks), "asr_model": model_id, "timestamps": meta["timestamps"],
                     "seconds": round(time.time() - t0, 2)})
        if k % 20 == 0 or k == len(todo):
            write_parquet(pd.DataFrame(rows, columns=COLUMNS), out, meta=meta)
    print(f"{len(rows)} transcripts ({len(todo)} new) -> {out}")


if __name__ == "__main__":
    main()
