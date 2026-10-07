"""SOAP notes for the audio pairs with the four open-weight models (GPU), reusing the ambient generator.

``llm_distract.ambient.generate_notes`` restricts ``--distractor`` to nonliteral|bystander, so this driver calls its
``load_model``/``generate`` (pinned revision, raw-text SOAP prompt without chat template, greedy, 512 new tokens, batch 2)
on the audio pairs table. Each distinct transcript is generated once (the clean transcript of A is shared by its
mixes) and the note is broadcast to every row. notes.parquet has the ambient layout, so ``llm_distract.ambient.judge``
runs unchanged.

    python -m llm_distract.audio.generate_notes --model meta-llama/Llama-3.1-8B-Instruct [--pairs data/audio/pairs.parquet] [--out outputs/audio]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from llm_distract.ambient.generate_notes import generate, git_commit, load_config, load_model
from llm_distract.ambient.prompts import build_note_prompt


def note_rows(pairs: pd.DataFrame) -> pd.DataFrame:
    """Long table (clean and distracted row per pair) in the layout written by the ambient generator."""
    rows = []
    for r in pairs.itertuples(index=False):
        for cond, col in (("clean", "clean_transcript"), ("distracted", "distracted_transcript")):
            rows.append({"source_dataset": r.source_dataset, "family": r.family, "item_id": r.item_id,
                         "distractor_type": r.distractor_type, "condition": cond, "transcript": getattr(r, col),
                         "reference_note": r.reference_note, "distractor_summary": r.distractor_summary})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", default="data/audio/pairs.parquet", help="output of llm_distract.audio.build_pairs")
    ap.add_argument("--out", default="outputs/audio")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--chat-template", action="store_true", help="apply the model's chat template")
    ap.add_argument("--attn-implementation", default=None, choices=[None, "eager", "sdpa"])
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    max_new_tokens = args.max_new_tokens or cfg["audio"].get("note_max_new_tokens", 512)
    revision = cfg["mechanism"]["revisions"].get(args.model)
    pairs = pd.read_parquet(args.pairs)
    if args.limit:
        pairs = pairs.head(args.limit)
    df = note_rows(pairs)
    unique = list(dict.fromkeys(df["transcript"]))
    tok, mdl = load_model(args.model, revision, args.attn_implementation)
    t0 = time.time()
    notes = generate(tok, mdl, [build_note_prompt(t) for t in unique], max_new_tokens,
                     cfg["ambient"]["note_batch_size"], args.chat_template)
    df["note"] = df["transcript"].map(dict(zip(unique, notes)))
    df["model"] = args.model
    out_dir = Path(args.out) / args.model.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "notes.parquet", index=False)
    manifest = {"model": args.model, "revision": revision, "chat_template": args.chat_template, "dtype": "bfloat16",
                "max_new_tokens": max_new_tokens, "batch_size": cfg["ambient"]["note_batch_size"],
                "decoding": "greedy", "n_notes": len(df), "n_unique_transcripts": len(unique), "pairs": args.pairs,
                "seconds": round(time.time() - t0, 1), "code": git_commit(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(df)} notes ({len(unique)} generations) to {out_dir}")


if __name__ == "__main__":
    main()
