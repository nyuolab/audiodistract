"""Stability of the CCHG interference-head sets across optimisation seeds (CPU only, no models needed).

Collects ``<cchg-dir>/<model_label>/<distractor>/seed<k>/heads.parquet`` (written by ``train_cchg``), compares the
selected sets between seeds and against the published seed-0 masks, and writes ``results/mech_seed_stability.csv``
(one row per model x distractor), the tidy gates ``results/raw/mech_masks_seeds.parquet`` and ``figures/ed_mech_seeds``.

    python -m llm_distract.mechanism.seed_stability [--cchg-dir outputs/mech/cchg] [--published results/raw/mech_masks.parquet]
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from llm_distract.common import style
from llm_distract.common.stats import jaccard
from llm_distract.common.style import DISTRACTOR_COLORS, DISTRACTOR_LABEL, MECHANISM_MODELS, MM, MODEL_SHORT, PALETTE
from llm_distract.mechanism.data import DISTRACTORS, read_provenance
from llm_distract.mechanism.provenance import write_with_provenance
from llm_distract.mechanism.models import code_version

COLUMNS = ["model", "distractor", "seed", "layer", "head", "gate", "selected", "threshold"]
MIN_SEEDS = 2  # a published head counts as recovered when at least this many seeds select it
RECOVERED = f"published_recovered_ge{MIN_SEEDS}"


def collect(cchg_dir: Path) -> tuple:
    """Tidy gates of every ``<label>/<distractor>/seed<k>/heads.parquet`` under ``cchg_dir``; seed from the directory name.

    ``model``/``distractor``/``selected`` come from the file (as written by train_cchg), the threshold from its
    provenance record. Returns (table sorted by model, distractor, seed, layer, head; list of source paths).
    """
    frames, sources = [], []
    for path in sorted(Path(cchg_dir).glob("*/*/seed*/heads.parquet")):
        threshold = read_provenance(path).get("hyperparameters", {}).get("gate_threshold", np.nan)
        df = pd.read_parquet(path).assign(seed=int(path.parent.name[len("seed"):]), threshold=float(threshold))
        frames.append(df[COLUMNS])
        sources.append(path)
    if not frames:
        raise FileNotFoundError(f"no <model>/<distractor>/seed*/heads.parquet under {cchg_dir}")
    tidy = pd.concat(frames, ignore_index=True).sort_values(["model", "distractor", "seed", "layer", "head"])
    return tidy.reset_index(drop=True), sources


def selected_sets(masks: pd.DataFrame, keys: list) -> dict:
    """{key tuple: set of selected (layer, head)} for every group of ``keys`` (groups without selected heads kept)."""
    return {k: set(map(tuple, g.loc[g["selected"], ["layer", "head"]].values.tolist())) for k, g in masks.groupby(keys)}


def _mmm(prefix: str, values: list) -> dict:
    """``<prefix>_mean/min/max`` of a list (NaN when empty; a NaN entry propagates)."""
    if not values:
        return {f"{prefix}_mean": np.nan, f"{prefix}_min": np.nan, f"{prefix}_max": np.nan}
    return {f"{prefix}_mean": float(np.mean(values)), f"{prefix}_min": np.min(values).item(), f"{prefix}_max": np.max(values).item()}


def _order(model: str, distractor: str) -> tuple:
    models = list(MECHANISM_MODELS)
    return (models.index(model) if model in models else len(models), model, DISTRACTORS.index(distractor))


def stability_table(tidy: pd.DataFrame, published: pd.DataFrame) -> pd.DataFrame:
    """One row per model x distractor: seeds found, heads selected per seed, pairwise Jaccard between seeds, Jaccard
    of each seed with the published mask, intersection/union across seeds, published heads recovered by >= MIN_SEEDS seeds
    (NaN, like the pairwise Jaccard, when fewer than MIN_SEEDS seeds were found)."""
    per_seed = selected_sets(tidy, ["model", "distractor", "seed"])
    pub = selected_sets(published, ["model", "distractor"])
    rows = []
    for (model, d), g in tidy.groupby(["model", "distractor"]):
        seeds = sorted(g["seed"].unique().tolist())
        sets = [per_seed[(model, d, s)] for s in seeds]
        hits = Counter(h for s in sets for h in s)
        p = pub.get((model, d))
        vs_pub = [jaccard(s, p) for s in sets] if p is not None else []
        recovered = sum(hits[h] >= MIN_SEEDS for h in p) if p is not None and len(seeds) >= MIN_SEEDS else np.nan
        rows.append({"model": model, "model_short": MODEL_SHORT.get(model, model), "distractor": d,
                     "seeds": ",".join(map(str, seeds)), "n_seeds": len(seeds), "total_heads": len(g) // len(seeds),
                     **_mmm("heads", [len(s) for s in sets]),
                     **_mmm("pairwise_jaccard", [jaccard(a, b) for a, b in combinations(sets, 2)]),
                     "intersection": len(set.intersection(*sets)), "union": len(set.union(*sets)),
                     "published_heads": len(p) if p is not None else np.nan, **_mmm("jaccard_published", vs_pub),
                     **{f"jaccard_published_seed{s}": j for s, j in zip(seeds, vs_pub)},
                     RECOVERED: recovered, f"{RECOVERED}_frac": recovered / len(p) if p else np.nan})
    rows.sort(key=lambda r: _order(r["model"], r["distractor"]))
    return pd.DataFrame.from_records(rows)


def figure(table: pd.DataFrame, tidy: pd.DataFrame, path: Path) -> None:
    """ed_mech_seeds: a) heads selected per seed (open) and in the published mask (filled); b) Jaccard between seeds
    (mean, range) and mean Jaccard of the seeds with the published mask (triangle)."""
    style.set_style()
    counts = tidy.groupby(["model", "distractor", "seed"])["selected"].sum()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(style.NATURE_SINGLE, 55 * MM), sharey=True, gridspec_kw={"wspace": 0.2})
    for i, r in enumerate(table.itertuples()):
        y, c = -i, DISTRACTOR_COLORS.get(r.distractor, PALETTE["slate"])
        per_seed = counts.loc[(r.model, r.distractor)]
        ax_a.plot(per_seed.values, [y] * len(per_seed), "o", mfc="white", mec=c, ms=3.2, mew=0.8)
        ax_a.plot(r.published_heads, y, "o", color=c, ms=3.2, mec="white", mew=0.4)
        ax_b.plot([r.pairwise_jaccard_min, r.pairwise_jaccard_max], [y, y], color=c, lw=0.8, solid_capstyle="round")
        ax_b.plot(r.pairwise_jaccard_mean, y, "o", color=c, ms=3.2, mec="white", mew=0.4)
        ax_b.plot(r.jaccard_published_mean, y, "^", mfc="white", mec=c, ms=3.2, mew=0.8)
    ticks = table.assign(y=-np.arange(len(table))).groupby("model_short", sort=False)["y"].mean()
    ax_a.set_yticks(ticks.values)
    ax_a.set_yticklabels(ticks.index)
    ax_a.tick_params(axis="y", length=0)
    ax_a.set_xlim(left=0)
    ax_a.set_xlabel("Heads selected")
    ax_b.set_xlim(0, 1)
    ax_b.set_xticks([0, 0.5, 1])
    ax_b.set_xticklabels(["0", "0.5", "1"])
    ax_b.set_xlabel("Jaccard similarity")
    grey = PALETTE["grey"]
    legends = ((ax_a, [Line2D([], [], marker="o", ls="", mfc="white", mec=grey, label="Per seed"),
                       Line2D([], [], marker="o", ls="", color=grey, label="Published (seed 0)")]),
               (ax_b, [Line2D([], [], marker="o", ls="-", color=grey, label="Between seeds"),
                       Line2D([], [], marker="^", ls="", mfc="white", mec=grey, label="vs published")]))
    for ax, handles in legends:
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.5, columnspacing=0.8,
                  handletextpad=0.3, handlelength=1.4, borderaxespad=0.2)
    fig.legend(handles=[Line2D([], [], marker="s", ls="", color=DISTRACTOR_COLORS[d], label=DISTRACTOR_LABEL[d]) for d in DISTRACTORS],
               loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=5.5, handletextpad=0.3, columnspacing=1.0)
    style.place_panel_labels(fig, [("a", ax_a, 0, 0), ("b", ax_b, 1, 0)])
    style.save(fig, path)


def run(cchg_dir: Path, published: Path, out: Path, raw: Path, figures: Path) -> pd.DataFrame:
    """Collect the seeds and write mech_seed_stability.csv, mech_masks_seeds.parquet and ed_mech_seeds; return the table."""
    tidy, sources = collect(cchg_dir)
    table = stability_table(tidy, pd.read_parquet(published))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "mech_seed_stability.csv", index=False, float_format="%.6g")
    write_with_provenance(tidy, raw / "mech_masks_seeds.parquet", sources,
                          {"exporter": "llm_distract.mechanism.seed_stability", "code_version": code_version(),
                           "published": str(published)})
    figure(table, tidy, figures / "ed_mech_seeds")
    return table


def summarize(table: pd.DataFrame) -> str:
    return "\n".join(
        f"{r.model_short} {r.distractor[:2].upper()}: seeds {r.seeds}; heads {r.heads_mean:.1f} ({r.heads_min:g}-{r.heads_max:g}); "
        f"pairwise J {r.pairwise_jaccard_mean:.2f} ({r.pairwise_jaccard_min:.2f}-{r.pairwise_jaccard_max:.2f}); "
        f"J vs published {r.jaccard_published_mean:.2f}; published heads in >={MIN_SEEDS} seeds "
        f"{getattr(r, RECOVERED):g}/{r.published_heads:g} ({100 * getattr(r, RECOVERED + '_frac'):.0f}%); "
        f"intersection {r.intersection}, union {r.union}" for r in table.itertuples())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cchg-dir", type=Path, default=Path("outputs/mech/cchg"),
                    help="root holding <model_label>/<distractor>/seed<k>/heads.parquet")
    ap.add_argument("--published", type=Path, default=Path("results/raw/mech_masks.parquet"), help="published seed-0 masks")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--raw", type=Path, default=Path("results/raw"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    args = ap.parse_args()
    print(summarize(run(args.cchg_dir, args.published, args.out, args.raw, args.figures)))


if __name__ == "__main__":
    main()
