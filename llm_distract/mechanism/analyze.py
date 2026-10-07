"""Regenerate every mechanism table from results/raw/mech_*.parquet (CPU only, no models needed).

All mechanistic comparisons use the locked cohort of 127 held-out items (seed-0 split, never used for CCHG training).
Where an artifact was stored on all 1,273 items (Mistral/Gemma baseline, patching, prompting) the same table also
carries ``*_all`` sensitivity columns computed on every stored item. Accuracies are in percent, differences in
percentage points (pp) with 2,000-sample paired percentile-bootstrap CIs (seed 0) and exact McNemar p-values.

    python -m llm_distract.mechanism.analyze [--raw results/raw] [--out results]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llm_distract.common.stats import jaccard, mcnemar_exact, paired_accuracy_diff, paired_bootstrap_ci
from llm_distract.common.style import MECHANISM_MODELS, MODEL_SHORT
from llm_distract.mechanism.data import DISTRACTORS

MODELS = list(MECHANISM_MODELS)  # the four open-weight models of the mechanism arm (MODEL_SHORT also names the API models)
INTERVENTIONS = ("interference", "random", "all")


def _paired(ref: np.ndarray, cmp: np.ndarray, prefix: str) -> dict:
    """mean(cmp - ref) in pp with bootstrap CI, binomial-covariance SE and exact McNemar p."""
    delta, low, high = paired_bootstrap_ci(ref, cmp)
    return {f"{prefix}_pp": delta, f"{prefix}_ci_low": low, f"{prefix}_ci_high": high,
            f"{prefix}_se": paired_accuracy_diff(ref, cmp)[1], f"{prefix}_p": mcnemar_exact(ref, cmp)}


def _item_vectors(df: pd.DataFrame, key: str, values: list, item_col: str = "item_id") -> pd.DataFrame:
    """Wide item x value table of is_correct (inner join over items present in every value)."""
    wide = df.pivot_table(index=item_col, columns=key, values="is_correct", aggfunc="first")[values]
    return wide.dropna().astype(int)


def baseline_table(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        sub = base[base["model"] == model]
        for d in DISTRACTORS:
            row = {"model": model, "model_short": MODEL_SHORT[model], "distractor": d}
            for suffix, part in (("", sub[sub["heldout"]]), ("_all", sub)):
                if suffix == "_all" and part["item_id"].nunique() == sub[sub["heldout"]]["item_id"].nunique():
                    continue  # stored on the held-out cohort only: no separate sensitivity set
                wide = _item_vectors(part, "condition", ["clean", d])
                row.update({f"n{suffix}": len(wide), f"clean_acc{suffix}": 100 * wide["clean"].mean(),
                            f"distracted_acc{suffix}": 100 * wide[d].mean(),
                            **{k + suffix: v for k, v in _paired(wide[d].values, wide["clean"].values, "drop").items()}})
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def sparsity_table(masks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, d), g in masks.groupby(["model", "distractor"], sort=False):
        rows.append({"model": model, "model_short": MODEL_SHORT[model], "distractor": d,
                     "n_layers": int(g["layer"].max() + 1), "heads_per_layer": int(g["head"].max() + 1),
                     "total_heads": len(g), "heads_selected": int(g["selected"].sum()),
                     "pct_selected": 100 * g["selected"].mean(), "threshold": float(g["threshold"].iloc[0]),
                     "gate_min": g["gate"].min(), "gate_median": g["gate"].median(), "gate_mean": g["gate"].mean(),
                     "gate_max": g["gate"].max()})
    return pd.DataFrame.from_records(rows)


def jaccard_table(masks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        sets = {d: set(map(tuple, masks[(masks["model"] == model) & (masks["distractor"] == d) & masks["selected"]]
                           [["layer", "head"]].values.tolist())) for d in DISTRACTORS}
        nl, by = sets["nonliteral"], sets["bystander"]
        rows.append({"model": model, "model_short": MODEL_SHORT[model], "nonliteral_selected": len(nl),
                     "bystander_selected": len(by), "overlap": len(nl & by), "union": len(nl | by),
                     "jaccard": jaccard(nl, by), "bystander_subset_of_nonliteral": by <= nl})
    return pd.DataFrame.from_records(rows)


def layers_table(masks: pd.DataFrame) -> pd.DataFrame:
    """Selected heads per layer with relative depth layer/(L-1) and thirds (early/middle/late)."""
    rows = []
    for (model, d), g in masks.groupby(["model", "distractor"], sort=False):
        n_layers = int(g["layer"].max() + 1)
        counts = g.groupby("layer")["selected"].sum()
        for layer in range(n_layers):
            rel = layer / (n_layers - 1)
            rows.append({"model": model, "model_short": MODEL_SHORT[model], "distractor": d, "layer": layer,
                         "n_layers": n_layers, "rel_depth": rel,
                         "third": "early" if rel <= 1 / 3 else ("middle" if rel <= 2 / 3 else "late"),
                         "n_selected": int(counts.get(layer, 0)), "heads_per_layer": int(g["head"].max() + 1)})
    return pd.DataFrame.from_records(rows)


def patching_table(pat: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for d in DISTRACTORS:
            p = pat[(pat["model"] == model) & (pat["distractor"] == d)]
            b = base[(base["model"] == model) & (base["condition"] == d)].set_index("item_id")["is_correct"]
            row = {"model": model, "model_short": MODEL_SHORT[model], "distractor": d}
            for suffix, part in (("", p[p["heldout"]]), ("_all", p)):
                if suffix == "_all" and part["item_id"].nunique() == p[p["heldout"]]["item_id"].nunique():
                    continue
                has_none = "none" in set(part["intervention"])
                wide = _item_vectors(part, "intervention", list(INTERVENTIONS) + (["none"] if has_none else []))
                # comparator: the run's own un-patched distracted pass when available, else the baseline evaluation
                wide["baseline"] = wide["none"].astype(int) if has_none else b.reindex(wide.index).astype(int)
                row[f"n{suffix}"] = len(wide)
                row[f"baseline_distracted_acc{suffix}"] = 100 * wide["baseline"].mean()
                for iv in INTERVENTIONS:
                    row[f"{iv}_acc{suffix}"] = 100 * wide[iv].mean()
                    row.update({k + suffix: v for k, v in _paired(wide["baseline"].values, wide[iv].values, f"{iv}_delta").items()})
                row.update({k + suffix: v for k, v in
                            _paired(wide["random"].values, wide["interference"].values, "interference_minus_random").items()})
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def suppression_table(frontier: pd.DataFrame) -> pd.DataFrame:
    """Accuracy at each scale for the selected heads and, when present, for a size-matched random head set
    (``head_set`` column); deltas are relative to the unmodified model (selected heads at scale 1)."""
    if "head_set" not in frontier.columns:
        frontier = frontier.assign(head_set="interference")
    rows = []
    for (model, d), g in frontier.groupby(["model", "distractor"], sort=False):
        ref = g[(g["scale"] == 1.0) & (g["head_set"] == "interference")].iloc[0]
        for _, r in g.sort_values(["head_set", "scale"], ascending=[True, False]).iterrows():
            rows.append({"model": model, "model_short": MODEL_SHORT[model], "distractor": d, "head_set": r["head_set"], "scale": r["scale"],
                         "n": int(r["n"]), "num_scaled_heads": int(r["num_scaled_heads"]),
                         "clean_acc": 100 * r["clean_accuracy"], "distracted_acc": 100 * r["distracted_accuracy"],
                         "clean_delta_pp": 100 * (r["clean_accuracy"] - ref["clean_accuracy"]),
                         "distracted_delta_pp": 100 * (r["distracted_accuracy"] - ref["distracted_accuracy"]),
                         "clean_margin": r["clean_margin"], "distracted_margin": r["distracted_margin"]})
    return pd.DataFrame.from_records(rows)


def prompting_table(pro: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for d in DISTRACTORS:
            p = pro[(pro["model"] == model) & (pro["distractor"] == d)]
            row = {"model": model, "model_short": MODEL_SHORT[model], "distractor": d}
            for suffix, part in (("", p[p["heldout"]]), ("_all", p)):
                if suffix == "_all" and part["item_id"].nunique() == p[p["heldout"]]["item_id"].nunique():
                    continue
                wide = _item_vectors(part, "prompt_style", ["baseline", "ignore", "cot"])
                row[f"n{suffix}"] = len(wide)
                for style in ("baseline", "ignore", "cot"):
                    row[f"{style}_acc{suffix}"] = 100 * wide[style].mean()
                for style in ("ignore", "cot"):
                    row.update({k + suffix: v for k, v in _paired(wide["baseline"].values, wide[style].values, f"{style}_delta").items()})
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def attention_table(att: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, d), g in att.groupby(["model", "distractor"], sort=False):
        sel, other = g[g["is_interference"]], g[~g["is_interference"]]
        rows.append({"model": model, "model_short": MODEL_SHORT[model], "distractor": d,
                     "n_items": int(g["n_items"].max()), "n_interference_heads": len(sel), "n_other_heads": len(other),
                     "interference_mass": sel["mean_mass"].mean(), "other_mass": other["mean_mass"].mean(),
                     "ratio": sel["mean_mass"].mean() / other["mean_mass"].mean()})
    return pd.DataFrame.from_records(rows)


def run(raw: Path, out: Path) -> dict:
    """Build all tables, write results/mech_*.csv and return them keyed by name."""
    load = {n: pd.read_parquet(raw / f"mech_{n}.parquet") for n in ("baseline", "masks", "patching", "frontier", "prompting", "attention")}
    tables = {"baseline": baseline_table(load["baseline"]), "sparsity": sparsity_table(load["masks"]),
              "jaccard": jaccard_table(load["masks"]), "layers": layers_table(load["masks"]),
              "patching": patching_table(load["patching"], load["baseline"]),
              "suppression": suppression_table(load["frontier"]), "prompting": prompting_table(load["prompting"]),
              "attention": attention_table(load["attention"])}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"mech_{name}.csv", index=False, float_format="%.6g")
    return tables


def summarize(t: dict) -> str:
    b, p, s = t["baseline"], t["patching"], t["suppression"]
    s0 = s[s["scale"] == 0.0]
    lines = ["baseline (127 held-out): " + "; ".join(
        f"{r.model_short} {r.distractor[:2].upper()} {r.clean_acc:.1f}->{r.distracted_acc:.1f} (drop {r.drop_pp:+.1f} pp, p={r.drop_p:.3g})"
        for r in b.itertuples()),
        "sparsity: " + ", ".join(f"{r.model_short} {r.distractor[:2].upper()} {r.heads_selected}/{r.total_heads} ({r.pct_selected:.1f}%)"
                                 for r in t["sparsity"].itertuples()),
        "jaccard NL vs BY: " + ", ".join(f"{r.model_short} {r.jaccard:.2f}" for r in t["jaccard"].itertuples()),
        f"patching: {(p['interference_delta_pp'] > 0).sum()}/8 conditions improve; interference delta range "
        f"{p['interference_delta_pp'].min():+.1f} to {p['interference_delta_pp'].max():+.1f} pp; beats random in "
        f"{(p['interference_minus_random_pp'] > 0).sum()}/8",
        f"hard suppression (scale 0 vs 1): clean drops {(-s0['clean_delta_pp']).min():.1f}-{(-s0['clean_delta_pp']).max():.1f} pp; "
        f"distracted changes {s0['distracted_delta_pp'].min():+.1f} to {s0['distracted_delta_pp'].max():+.1f} pp",
        "prompting (127): " + "; ".join(f"{r.model_short} {r.distractor[:2].upper()} ignore {r.ignore_delta_pp:+.1f} cot {r.cot_delta_pp:+.1f}"
                                        for r in t["prompting"].itertuples()),
        "attention ratio: " + ", ".join(f"{r.model_short} {r.distractor[:2].upper()} {r.ratio:.2f}" for r in t["attention"].itertuples())]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("results/raw"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    print(summarize(run(args.raw, args.out)))


if __name__ == "__main__":
    main()
