"""Score MedDistractQA items by the conditional log-probability of the four answer letters.

Used for the baseline evaluation (clean / nonliteral / bystander, prompt style ``baseline``) and for the prompting
mitigations (``ignore`` instruction, ``cot`` suffix), optionally under fixed head gates. Prediction = argmax letter.

    python -m llm_distract.mechanism.evaluate --model meta-llama/Llama-3.1-8B-Instruct \
        --conditions clean nonliteral bystander --prompt-style baseline --split heldout \
        --out outputs/mech/baseline/meta-llama__Llama-3.1-8B-Instruct.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from llm_distract.mechanism.data import CHOICE_LETTERS, CONDITIONS, PROMPT_STYLES, prompt_for, select_split, \
    tokenize_prompt_targets, write_parquet
from llm_distract.mechanism.models import add_common_args, label, setup


def score_mcq(model, tokenizer, pairs: pd.DataFrame, condition: str, style: str = "baseline", batch_size: int = 4,
              gates=None) -> pd.DataFrame:
    """Per-item predictions for one condition: pred/is_correct/logp_correct/margin and all four letter log-probs."""
    import torch
    from contextlib import nullcontext
    from llm_distract.mechanism.chg import HeadGates, target_logprob

    rows = pairs.to_dict("records")
    prompts = [prompt_for(r, condition, style) for r in rows]
    context = HeadGates(model).apply(gates) if gates is not None else nullcontext()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad(), context:
        for start in range(0, len(rows), batch_size):
            chunk = list(range(start, min(start + batch_size, len(rows))))
            batch_prompts = [prompts[i] for i in chunk for _ in CHOICE_LETTERS]
            targets = [letter for _ in chunk for letter in CHOICE_LETTERS]
            ids, masks = tokenize_prompt_targets(tokenizer, batch_prompts, targets)
            logp = target_logprob(model, ids.to(device), masks.to(device), "sum").cpu().view(len(chunk), 4)
            for j, i in enumerate(chunk):
                scores = {letter: float(logp[j, k]) for k, letter in enumerate(CHOICE_LETTERS)}
                correct = rows[i]["correct_answer"]
                pred = max(scores, key=scores.get)
                best_wrong = max(v for k, v in scores.items() if k != correct)
                out.append({"item_id": int(rows[i]["item_id"]), "condition": condition, "prompt_style": style,
                            "correct_answer": correct, "pred_answer": pred, "is_correct": pred == correct,
                            "logp_correct": scores[correct], "margin": scores[correct] - best_wrong,
                            "choice_logps": json.dumps(scores, sort_keys=True)})
    return pd.DataFrame.from_records(out)


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--prompt-style", default="baseline", choices=PROMPT_STYLES)
    ap.add_argument("--split", default="heldout", choices=["heldout", "train", "all"])
    ap.add_argument("--batch-size", type=int, default=4, help="items per forward pass (x4 answer letters)")
    ap.add_argument("--out", type=Path, default=None, help="output parquet (default outputs/mech/<style>/<label>.parquet)")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N items of the split (smoke tests)")
    args = ap.parse_args(argv)

    cfg, pairs, model, tokenizer, prov = setup(args)
    mech = cfg["mechanism"]
    subset = select_split(pairs, args.split, mech["validation_fraction"], mech["split_seed"])
    if args.limit is not None:
        subset = subset.head(args.limit)
    frames = [score_mcq(model, tokenizer, subset, c, args.prompt_style, args.batch_size) for c in args.conditions]
    df = pd.concat(frames, ignore_index=True)
    out = args.out or cfg["paths"]["work_dir"] / "mech" / args.prompt_style / f"{label(args.model)}.parquet"
    write_parquet(df, out, {**prov, "split": args.split, "n_items": int(subset.shape[0])})
    for c, g in df.groupby("condition"):
        print(f"{args.model} {args.prompt_style} {c}: accuracy {g['is_correct'].mean():.3f} (n={len(g)})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
