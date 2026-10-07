"""Train a contrastive causal head gating (CCHG) mask for one model and one distractor type.

Training items = the 1,146 non-held-out items of the locked split; the 127 held-out items are never seen.
Writes ``<out>/<model_label>/<distractor>/seed<seed>/{mask.pt, heads.parquet, train_log.parquet, summary.json}``.
Hyperparameters default to ``mechanism.cchg`` in configs/default.yaml (500 updates, batch 4 x accumulation 4,
Adam 1e-2 -> 1e-3 linear, lambda 5, grad clip 1, logit clamp 6, max 512 tokens, selection gate < 0.5).

    python -m llm_distract.mechanism.train_cchg --model meta-llama/Llama-3.1-8B-Instruct --distractor nonliteral \
        --seed 0 --lambda 5 --out outputs/mech/cchg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from llm_distract.mechanism.data import DISTRACTORS, contrastive_tensors, select_split, write_parquet
from llm_distract.mechanism.models import add_common_args, label, seed_everything, setup


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--distractor", required=True, choices=DISTRACTORS)
    ap.add_argument("--seed", type=int, default=None, help="optimisation seed (batch order); default config")
    ap.add_argument("--lambda", dest="l1_weight", type=float, default=None, help="weight of the mean(1 - g) penalty")
    ap.add_argument("--out", type=Path, default=None, help="output root (default outputs/mech/cchg)")
    for name in ("num_updates", "batch_size", "grad_accum", "max_seq_length"):
        ap.add_argument(f"--{name.replace('_', '-')}", type=int, default=None)
    for name in ("lr", "lr_end", "grad_clip", "logit_clamp", "gate_threshold"):
        ap.add_argument(f"--{name.replace('_', '-')}", type=float, default=None)
    ap.add_argument("--no-pbar", action="store_true")
    args = ap.parse_args(argv)

    import torch
    from llm_distract.mechanism.chg import ContrastiveMaskTrainer, select_heads

    cfg, pairs, model, tokenizer, prov = setup(args)
    mech = cfg["mechanism"]
    hp = {k: (getattr(args, k) if getattr(args, k) is not None else v) for k, v in mech["cchg"].items()}
    train = select_split(pairs, "train", mech["validation_fraction"], mech["split_seed"])
    tensors = contrastive_tensors(tokenizer, train, args.distractor, hp["max_seq_length"])

    seed_everything(hp["seed"])
    trainer = ContrastiveMaskTrainer(model, tensors, tokenizer.pad_token_id, batch_size=hp["batch_size"],
                                     grad_accum=hp["grad_accum"], lr=hp["lr"], lr_end=hp["lr_end"],
                                     l1_weight=hp["l1_weight"], grad_clip=hp["grad_clip"],
                                     logit_clamp=hp["logit_clamp"], seed=hp["seed"])
    gates, log = trainer.fit(num_updates=hp["num_updates"], show_pbar=not args.no_pbar)
    import torch
    if torch.cuda.is_available():
        print(f"peak GPU memory during training: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
              f"(batch {hp['batch_size']} x accumulation {hp['grad_accum']}, pos/neg tensors {tuple(tensors['pos_ids'].shape)})")
    selected = select_heads(gates, hp["gate_threshold"])

    out_dir = (args.out or cfg["paths"]["work_dir"] / "mech" / "cchg") / label(args.model) / args.distractor / f"seed{hp['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gates, out_dir / "mask.pt")
    layers, heads = gates.shape
    heads_df = pd.DataFrame({"layer": [l for l in range(layers) for _ in range(heads)],
                             "head": [h for _ in range(layers) for h in range(heads)],
                             "gate": gates.flatten().tolist(), "selected": selected.flatten().tolist()})
    summary = {**prov, "distractor": args.distractor, "hyperparameters": hp, "split_seed": mech["split_seed"],
               "validation_fraction": mech["validation_fraction"], "n_train_items": int(len(train)),
               "n_layers": layers, "heads_per_layer": heads, "n_selected": int(selected.sum()),
               "fraction_selected": float(selected.float().mean()), "final_loss": float(log["loss"].iloc[-1])}
    write_parquet(heads_df.assign(model=args.model, distractor=args.distractor), out_dir / "heads.parquet", summary)
    write_parquet(log, out_dir / "train_log.parquet", summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"{args.model} {args.distractor}: {summary['n_selected']} heads selected "
          f"({100 * summary['fraction_selected']:.2f}%); artifacts in {out_dir}")


if __name__ == "__main__":
    main()
