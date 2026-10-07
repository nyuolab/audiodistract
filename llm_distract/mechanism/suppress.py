"""Static suppression frontier: scale the selected heads' outputs by s in {1, .75, .5, .25, 0} on clean and
distracted prompts of the held-out cohort (s = 0 is hard suppression, s = 1 the unmodified model).

Writes ``<out>/<model_label>/<distractor>/frontier_items.parquet`` (item x condition x scale) and
``frontier.parquet`` (one summary row per scale, the schema of results/raw/mech_frontier.parquet).

    python -m llm_distract.mechanism.suppress --model meta-llama/Llama-3.1-8B-Instruct --distractor nonliteral \
        --mask outputs/mech/cchg/meta-llama__Llama-3.1-8B-Instruct/nonliteral/seed0/mask.pt --out outputs/mech/frontier
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from llm_distract.mechanism.data import DISTRACTORS, select_split, write_parquet
from llm_distract.mechanism.evaluate import score_mcq
from llm_distract.mechanism.models import add_common_args, label, setup


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--distractor", required=True, choices=DISTRACTORS)
    ap.add_argument("--mask", required=True, type=Path, help="mask.pt with sigmoid gates [layers, heads]")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--scales", nargs="+", type=float, default=None, help="default config static_frontier_scales")
    ap.add_argument("--split", default="heldout", choices=["heldout", "train", "all"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="output root (default outputs/mech/frontier)")
    ap.add_argument("--random-control", action="store_true",
                    help="also scale a size-matched random head set (drawn with --control-seed, excluding selected heads)")
    ap.add_argument("--control-seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    from llm_distract.mechanism.chg import random_head_set, scaled_gates, select_heads

    cfg, pairs, model, tokenizer, prov = setup(args)
    mech = cfg["mechanism"]
    scales = args.scales or mech["static_frontier_scales"]
    subset = select_split(pairs, args.split, mech["validation_fraction"], mech["split_seed"])
    selected = select_heads(torch.load(args.mask, map_location="cpu"), args.threshold)
    head_sets = {"interference": selected}
    if args.random_control:
        head_sets["random"] = random_head_set(selected, args.control_seed)
    frames, summary_rows = [], []
    for set_name, heads in head_sets.items():
        for scale in scales:
            if set_name == "random" and scale == 1.0:
                continue  # identical to the unmodified model
            gates = scaled_gates(heads, scale)
            per_cond = {c: score_mcq(model, tokenizer, subset, c, "baseline", args.batch_size, gates) for c in ("clean", args.distractor)}
            for c, df in per_cond.items():
                frames.append(df.assign(scale=scale, head_set=set_name))
            clean, dist = per_cond["clean"], per_cond[args.distractor]
            summary_rows.append({"model": args.model, "distractor": args.distractor, "threshold": args.threshold, "scale": scale,
                                 "head_set": set_name, "num_scaled_heads": int(heads.sum()),
                                 "clean_accuracy": float(clean["is_correct"].mean()), "distracted_accuracy": float(dist["is_correct"].mean()),
                                 "clean_margin": float(clean["margin"].mean()), "distracted_margin": float(dist["margin"].mean()),
                                 "n": int(len(clean))})
            print(f"{set_name} scale {scale}: clean {summary_rows[-1]['clean_accuracy']:.3f} distracted {summary_rows[-1]['distracted_accuracy']:.3f}")
    out_dir = (args.out or cfg["paths"]["work_dir"] / "mech" / "frontier") / label(args.model) / args.distractor
    meta = {**prov, "distractor": args.distractor, "mask": str(args.mask), "threshold": args.threshold, "scales": scales,
            "split": args.split, "n_items": int(len(subset)), "random_control": args.random_control, "control_seed": args.control_seed}
    write_parquet(pd.concat(frames, ignore_index=True), out_dir / "frontier_items.parquet", meta)
    write_parquet(pd.DataFrame.from_records(summary_rows), out_dir / "frontier.parquet", meta)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
