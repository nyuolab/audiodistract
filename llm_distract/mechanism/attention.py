"""Distractor attention mass: attention from the answer-prediction position (final prompt token) onto the inserted
distractor sentence, per (layer, head), for the held-out distracted prompts; heads are labelled as selected or not
from a CCHG mask. Loads the model with eager attention so that attention weights are materialised.

Writes ``<out>/<model_label>/<distractor>/attention_items.parquet`` (item x layer x head) and
``attention_heads.parquet`` (per-head mean, the schema of results/raw/mech_attention.parquet).

    python -m llm_distract.mechanism.attention --model meta-llama/Llama-3.1-8B-Instruct --distractor nonliteral \
        --mask outputs/mech/cchg/meta-llama__Llama-3.1-8B-Instruct/nonliteral/seed0/mask.pt --out outputs/mech/attention
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from llm_distract.mechanism.data import DISTRACTORS, prompt_for, select_split, write_parquet
from llm_distract.mechanism.models import add_common_args, label, setup


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--distractor", required=True, choices=DISTRACTORS)
    ap.add_argument("--mask", required=True, type=Path, help="mask.pt with sigmoid gates [layers, heads]")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--split", default="heldout", choices=["heldout", "train", "all"])
    ap.add_argument("--out", type=Path, default=None, help="output root (default outputs/mech/attention)")
    args = ap.parse_args(argv)

    import torch
    from llm_distract.mechanism.chg import distractor_attention_mass, select_heads

    cfg, pairs, model, tokenizer, prov = setup(args, attn_implementation="eager")
    mech = cfg["mechanism"]
    subset = select_split(pairs, args.split, mech["validation_fraction"], mech["split_seed"])
    selected = select_heads(torch.load(args.mask, map_location="cpu"), args.threshold)
    layers, heads = selected.shape
    records, skipped = [], 0
    for r in subset.to_dict("records"):
        mass = distractor_attention_mass(model, tokenizer, prompt_for(r, args.distractor), r[f"{args.distractor}_sentence"])
        if mass is None:
            skipped += 1
            continue
        for layer in range(layers):
            for head in range(heads):
                records.append({"item_id": int(r["item_id"]), "layer": layer, "head": head,
                                "is_interference": bool(selected[layer, head]), "mass": float(mass[layer, head])})
    items = pd.DataFrame.from_records(records)
    per_head = (items.groupby(["layer", "head", "is_interference"], as_index=False)
                .agg(mean_mass=("mass", "mean"), n_items=("item_id", "nunique"))
                .assign(model=args.model, distractor=args.distractor))
    out_dir = (args.out or cfg["paths"]["work_dir"] / "mech" / "attention") / label(args.model) / args.distractor
    meta = {**prov, "distractor": args.distractor, "mask": str(args.mask), "threshold": args.threshold,
            "split": args.split, "n_items": int(items["item_id"].nunique()), "n_skipped": skipped}
    write_parquet(items, out_dir / "attention_items.parquet", meta)
    write_parquet(per_head, out_dir / "attention_heads.parquet", meta)
    by_group = per_head.groupby("is_interference")["mean_mass"].mean()
    print(f"mean mass selected {by_group.get(True, float('nan')):.4f} vs other {by_group.get(False, float('nan')):.4f}; wrote {out_dir}")


if __name__ == "__main__":
    main()
