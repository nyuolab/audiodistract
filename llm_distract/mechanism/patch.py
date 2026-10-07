"""Clean-activation patching on the held-out cohort: replace selected heads' outputs with their clean-prompt outputs.

Interventions: ``none`` (unpatched distracted prompt), ``interference`` (gate < threshold heads), ``random``
(size-matched heads drawn with a seeded generator from the non-selected heads) and ``all`` (every head).
Writes ``<out>/<model_label>/<distractor>/patching.parquet`` (one row per item x intervention) and ``summary.json``
with the selected and random head sets.

    python -m llm_distract.mechanism.patch --model meta-llama/Llama-3.1-8B-Instruct --distractor nonliteral \
        --mask outputs/mech/cchg/meta-llama__Llama-3.1-8B-Instruct/nonliteral/seed0/mask.pt --split heldout \
        --out outputs/mech/patching
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from llm_distract.mechanism.data import CHOICE_LETTERS, DISTRACTORS, prompt_for, select_split, \
    tokenize_prompt_targets, write_parquet
from llm_distract.mechanism.models import add_common_args, label, setup


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--distractor", required=True, choices=DISTRACTORS)
    ap.add_argument("--mask", required=True, type=Path, help="mask.pt with sigmoid gates [layers, heads]")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--split", default="heldout", choices=["heldout", "train", "all"])
    ap.add_argument("--seed", type=int, default=0, help="seed of the random-head control")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None, help="output root (default outputs/mech/patching)")
    args = ap.parse_args(argv)

    import torch
    from llm_distract.mechanism.chg import patched_target_logprob, random_head_set, select_heads, target_logprob

    cfg, pairs, model, tokenizer, prov = setup(args)
    mech = cfg["mechanism"]
    subset = select_split(pairs, args.split, mech["validation_fraction"], mech["split_seed"])
    gates = torch.load(args.mask, map_location="cpu")
    selected = select_heads(gates, args.threshold)
    head_sets = {"interference": selected, "random": random_head_set(selected, args.seed),
                 "all": torch.ones_like(selected)}
    device = next(model.parameters()).device
    rows = subset.to_dict("records")
    records = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            scores = {name: [dict() for _ in batch] for name in ["none"] + list(head_sets)}
            for letter in CHOICE_LETTERS:
                targets = [letter] * len(batch)
                clean_ids, _ = tokenize_prompt_targets(tokenizer, [prompt_for(r, "clean") for r in batch], targets)
                dis_ids, dis_masks = tokenize_prompt_targets(tokenizer, [prompt_for(r, args.distractor) for r in batch], targets)
                plain = target_logprob(model, dis_ids.to(device), dis_masks.to(device), "sum").cpu()
                for j in range(len(batch)):
                    scores["none"][j][letter] = float(plain[j])
                for name, head_mask in head_sets.items():
                    logp = patched_target_logprob(model, clean_ids, dis_ids, dis_masks, head_mask, tokenizer.pad_token_id)
                    for j in range(len(batch)):
                        scores[name][j][letter] = float(logp[j])
            for name, per_item in scores.items():
                for j, r in enumerate(batch):
                    s, correct = per_item[j], r["correct_answer"]
                    pred = max(s, key=s.get)
                    best_wrong = max(v for k, v in s.items() if k != correct)
                    records.append({"item_id": int(r["item_id"]), "distractor": args.distractor, "intervention": name,
                                    "correct_answer": correct, "pred_answer": pred, "is_correct": pred == correct,
                                    "logp_correct": s[correct], "margin": s[correct] - best_wrong})
    df = pd.DataFrame.from_records(records)
    out_dir = (args.out or cfg["paths"]["work_dir"] / "mech" / "patching") / label(args.model) / args.distractor
    heads = {name: [[int(l), int(h)] for l, h in torch.nonzero(m).tolist()] for name, m in head_sets.items() if name != "all"}
    summary = {**prov, "distractor": args.distractor, "mask": str(args.mask), "threshold": args.threshold,
               "random_seed": args.seed, "split": args.split, "n_items": len(rows), "heads": heads,
               "accuracy": {k: float(v) for k, v in df.groupby("intervention")["is_correct"].mean().items()}}
    write_parquet(df, out_dir / "patching.parquet", summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["accuracy"], indent=1), f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
