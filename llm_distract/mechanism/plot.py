"""Figures for the mechanism section from results/mech_*.csv (plus results/raw/mech_masks.parquet for gate ECDFs).

    python -m llm_distract.mechanism.plot [--results results] [--figures figures]

fig4 (double column): a) paired accuracy decrement on the 127 held-out items, b) heads selected with Jaccard overlap,
c) clean-activation patching vs size-matched random heads, d) suppression frontier, e) layer depth of selected heads.
Extended data: ed_mech_prompting, ed_mech_attention, ed_mech_gates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from llm_distract.common import style
from llm_distract.common.style import DISTRACTOR_COLORS, DISTRACTOR_LABEL, MECHANISM_MODELS, MM, MODEL_COLORS, MODEL_SHORT, PALETTE

MODELS = [MODEL_SHORT[m] for m in MECHANISM_MODELS]
DIST = ("nonliteral", "bystander")
ROWS = [(m, d) for m in MODELS for d in DIST]  # top-to-bottom order of the 8 model x distractor rows


def _ypos(model: str, distractor: str) -> float:
    return -(ROWS.index((model, distractor)) + 0.35 * (DIST.index(distractor) - 0.5))


def _model_axis(ax) -> None:
    ax.set_yticks([-(i * 2 + 0.5) for i in range(len(MODELS))])
    ax.set_yticklabels(MODELS)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-(2 * len(MODELS)) + 0.35, 0.65)
    for i in range(len(MODELS) - 1):
        ax.axhline(-(2 * i + 1.5), color=PALETTE["light"], lw=0.5, zorder=0)


def _distractor_legend(ax, loc="lower right", **kw) -> None:
    handles = [Line2D([], [], marker="o", ls="", color=DISTRACTOR_COLORS[d], label=DISTRACTOR_LABEL[d]) for d in DIST]
    ax.legend(handles=handles, loc=loc, handletextpad=0.3, borderaxespad=0.2, **kw)


def panel_decrement(ax, base: pd.DataFrame) -> None:
    for r in base.itertuples():
        y, c = _ypos(r.model_short, r.distractor), DISTRACTOR_COLORS[r.distractor]
        ax.plot([r.drop_ci_low, r.drop_ci_high], [y, y], color=c, lw=0.8, solid_capstyle="round")
        ax.plot(r.drop_pp, y, "o", color=c, ms=3.5, mec="white", mew=0.4)
    ax.axvline(0, color=PALETTE["grey"], lw=0.6, ls=":")
    _model_axis(ax)
    ax.set_xlabel("Accuracy decrement, clean minus distracted (pp)")
    _distractor_legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.0)


def panel_sparsity(ax, sparsity: pd.DataFrame, jac: pd.DataFrame) -> None:
    for r in sparsity.itertuples():
        y = -(MODELS.index(r.model_short) * 2 + 0.5)
        ax.plot(r.pct_selected, y, "o" if r.distractor == "nonliteral" else "s", color=DISTRACTOR_COLORS[r.distractor],
                ms=4, mec="white", mew=0.4, zorder=3)
        ax.annotate(f"{r.heads_selected}", (r.pct_selected, y), xytext=(0, 4 if r.distractor == "nonliteral" else -4),
                    textcoords="offset points", ha="center", va="bottom" if r.distractor == "nonliteral" else "top",
                    fontsize=5.5, color=DISTRACTOR_COLORS[r.distractor])
    xmax = sparsity["pct_selected"].max() * 1.35
    for r in jac.itertuples():
        y = -(MODELS.index(r.model_short) * 2 + 0.5)
        ax.text(xmax, y, f"J = {r.jaccard:.2f}", ha="right", va="center", fontsize=6, color=PALETTE["slate"])
    ax.set_xlim(-0.5, xmax * 1.02)
    _model_axis(ax)
    ax.set_xlabel("Attention heads selected (%)")
    handles = [Line2D([], [], marker="o", ls="", color=DISTRACTOR_COLORS["nonliteral"], label="Nonliteral heads"),
               Line2D([], [], marker="s", ls="", color=DISTRACTOR_COLORS["bystander"], label="Bystander heads")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.0, handletextpad=0.3, borderaxespad=0.2)


def panel_patching(ax, pat: pd.DataFrame) -> None:
    h = 0.3
    for r in pat.itertuples():
        y = _ypos(r.model_short, r.distractor)
        for off, iv, color in ((h / 2, "interference", DISTRACTOR_COLORS[r.distractor]), (-h / 2, "random", PALETTE["random"])):
            delta = getattr(r, f"{iv}_delta_pp")
            ax.barh(y + off, delta, height=h, color=color, lw=0)
            ax.plot([getattr(r, f"{iv}_delta_ci_low"), getattr(r, f"{iv}_delta_ci_high")], [y + off, y + off],
                    color=PALETTE["slate"], lw=0.5)
    ax.axvline(0, color=PALETTE["grey"], lw=0.6, ls=":")
    _model_axis(ax)
    ax.set_xlabel("Accuracy change from distracted (pp)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=0.8, handletextpad=0.4,
              handles=[Patch(color=DISTRACTOR_COLORS["nonliteral"], label="Nonliteral"), Patch(color=DISTRACTOR_COLORS["bystander"], label="Bystander"),
                       Patch(color=PALETTE["random"], label="Random, size-matched")],
              handlelength=1.2, borderaxespad=0.2)


def panel_frontier(axes, sup: pd.DataFrame) -> None:
    scales = sorted(sup["scale"].unique(), reverse=True)
    if "head_set" not in sup.columns:
        sup = sup.assign(head_set="interference")
    has_random = (sup["head_set"] == "random").any()
    for ax, d in zip(axes, DIST):
        for m in MODELS:
            sel = sup[(sup["model_short"] == m) & (sup["distractor"] == d)]
            g = sel[sel["head_set"] == "interference"].set_index("scale").loc[scales]
            ax.plot(scales, g["clean_acc"], "-o", color=MODEL_COLORS[m], ms=2.8, lw=0.9)
            ax.plot(scales, g["distracted_acc"], "--o", color=MODEL_COLORS[m], ms=2.8, lw=0.9, mfc="white", mew=0.7)
            rnd = sel[sel["head_set"] == "random"]
            if len(rnd):
                rs = [1.0] + sorted(rnd["scale"].unique(), reverse=True)
                rc = [g.loc[1.0, "clean_acc"]] + [rnd.set_index("scale").loc[x, "clean_acc"] for x in rs[1:]]
                ax.plot(rs, rc, ":s", color=MODEL_COLORS[m], ms=2.2, lw=0.7, alpha=0.55)
        ax.axhline(25, color=PALETTE["grey"], lw=0.5, ls=":")
        ax.text(1.0, 25.8, "chance", fontsize=5.5, color=PALETTE["grey"], ha="left", va="bottom")
        ax.axvspan(-0.06, 0.06, color=PALETTE["light"], zorder=0)
        ax.invert_xaxis()
        ax.set_xticks(scales)
        ax.set_xticklabels([f"{s:g}" for s in scales])
        ax.set_xlabel("Scale applied to selected heads")
        ax.set_title(f"{DISTRACTOR_LABEL[d]} heads", loc="left")
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].tick_params(labelleft=False)
    handles = [Line2D([], [], color=MODEL_COLORS[m], lw=1, label=m) for m in MODELS] + [
        Line2D([], [], color=PALETTE["slate"], lw=0.9, marker="o", ms=2.8, label="Clean input"),
        Line2D([], [], color=PALETTE["slate"], lw=0.9, ls="--", marker="o", ms=2.8, mfc="white", label="Distracted input")]
    if has_random:
        handles.append(Line2D([], [], color=PALETTE["slate"], lw=0.7, ls=":", marker="s", ms=2.2, alpha=0.55, label="Random heads, clean input"))
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.24), ncol=4, columnspacing=1.2, handlelength=1.6,
                   borderaxespad=0.0, handletextpad=0.5)


def panel_layers(axes, layers: pd.DataFrame) -> None:
    edges = np.linspace(0, 1, 6)
    width = 0.2 / 2.2
    for ax, m in zip(axes, MODELS):
        for k, d in enumerate(DIST):
            g = layers[(layers["model_short"] == m) & (layers["distractor"] == d)]
            counts, _ = np.histogram(g["rel_depth"], bins=edges, weights=g["n_selected"])
            frac = 100 * counts / max(counts.sum(), 1)
            centers = (edges[:-1] + edges[1:]) / 2 + (k - 0.5) * width
            ax.bar(centers, frac, width=width, color=DISTRACTOR_COLORS[d], lw=0)
        ax.text(0.98, 0.98, f"{m} ({int(g['n_layers'].iloc[0])} layers)", transform=ax.transAxes, ha="right",
                va="top", fontsize=5.8, color=PALETTE["slate"])
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 30, 60])
        ax.tick_params(labelbottom=ax is axes[-1], labelsize=6)
    axes[-1].set_xticks([0, 0.5, 1])
    axes[-1].set_xlabel("Relative depth of selected heads")
    axes[0].set_title("Layer distribution", loc="left")
    axes[1].set_ylabel("Selected heads (%)", y=0, labelpad=7)


def fig4(results: Path, figures: Path) -> None:
    base = pd.read_csv(results / "mech_baseline.csv")
    sparsity = pd.read_csv(results / "mech_sparsity.csv")
    jac = pd.read_csv(results / "mech_jaccard.csv")
    pat = pd.read_csv(results / "mech_patching.csv")
    sup = pd.read_csv(results / "mech_suppression.csv")
    layers = pd.read_csv(results / "mech_layers.csv")

    fig = plt.figure(figsize=(style.NATURE_DOUBLE, 128 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.42, wspace=0.5, left=0.09, right=0.99, top=0.95, bottom=0.10)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
    gs_d = gs[1, :2].subgridspec(1, 2, wspace=0.1)
    axes_d = [fig.add_subplot(gs_d[0, 0])]
    axes_d.append(fig.add_subplot(gs_d[0, 1], sharey=axes_d[0]))
    gs_e = gs[1, 2].subgridspec(4, 1, hspace=0.25)
    axes_e = [fig.add_subplot(gs_e[i, 0]) for i in range(4)]

    panel_decrement(ax_a, base)
    panel_sparsity(ax_b, sparsity, jac)
    panel_patching(ax_c, pat)
    panel_frontier(axes_d, sup)
    panel_layers(axes_e, layers)
    style.place_panel_labels(fig, [("a", ax_a, 0, 0), ("b", ax_b, 1, 0), ("c", ax_c, 2, 0), ("d", axes_d, 0, 1), ("e", axes_e, 2, 1)])
    style.save(fig, figures / "fig4")


def ed_prompting(results: Path, figures: Path) -> None:
    pro = pd.read_csv(results / "mech_prompting.csv")
    fig, ax = plt.subplots(figsize=(style.NATURE_SINGLE, 62 * MM))
    colors = {"ignore": PALETTE["clean"], "cot": PALETTE["slate"]}
    for r in pro.itertuples():
        y = _ypos(r.model_short, r.distractor)
        for off, st in ((0.14, "ignore"), (-0.14, "cot")):
            ax.plot([getattr(r, f"{st}_delta_ci_low"), getattr(r, f"{st}_delta_ci_high")], [y + off, y + off], color=colors[st], lw=0.8)
            ax.plot(getattr(r, f"{st}_delta_pp"), y + off, "o" if r.distractor == "nonliteral" else "s", color=colors[st],
                    ms=3.2, mec="white", mew=0.4)
    ax.axvline(0, color=PALETTE["grey"], lw=0.6, ls=":")
    _model_axis(ax)
    ax.set_xlabel("Accuracy change vs baseline prompt on distracted items (pp)")
    handles = [Line2D([], [], marker="o", ls="", color=colors["ignore"], label="Ignore-irrelevant instruction"),
               Line2D([], [], marker="o", ls="", color=colors["cot"], label="Chain-of-thought suffix"),
               Line2D([], [], marker="o", ls="", color=PALETTE["grey"], label="Nonliteral"),
               Line2D([], [], marker="s", ls="", color=PALETTE["grey"], label="Bystander")]
    ax.legend(handles=handles, loc="lower left", fontsize=5.5, handletextpad=0.3, borderaxespad=0.2)
    style.save(fig, figures / "ed_mech_prompting")


def ed_attention(results: Path, figures: Path) -> None:
    att = pd.read_csv(results / "mech_attention.csv")
    fig, ax = plt.subplots(figsize=(style.NATURE_SINGLE, 62 * MM))
    for r in att.itertuples():
        y = _ypos(r.model_short, r.distractor)
        ax.plot([r.other_mass, r.interference_mass], [y, y], color=PALETTE["grey"], lw=0.8, zorder=1)
        ax.plot(r.other_mass, y, "o", color=PALETTE["random"], ms=3.5, mec=PALETTE["grey"], mew=0.5, zorder=2)
        ax.plot(r.interference_mass, y, "o", color=DISTRACTOR_COLORS[r.distractor], ms=3.5, mec="white", mew=0.4, zorder=3)
        ax.annotate(f"x{r.ratio:.2f}", (max(r.other_mass, r.interference_mass), y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=5.5, color=PALETTE["slate"])
    _model_axis(ax)
    ax.set_xlim(0, att[["other_mass", "interference_mass"]].max().max() * 1.35)
    ax.set_xlabel("Mean attention mass on the distractor sentence")
    handles = [Line2D([], [], marker="o", ls="", color=PALETTE["random"], mec=PALETTE["grey"], label="Other heads"),
               Line2D([], [], marker="o", ls="", color=DISTRACTOR_COLORS["nonliteral"], label="Selected heads, nonliteral"),
               Line2D([], [], marker="o", ls="", color=DISTRACTOR_COLORS["bystander"], label="Selected heads, bystander")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=5.5, handletextpad=0.3,
              borderaxespad=0.0, columnspacing=1.0)
    style.save(fig, figures / "ed_mech_attention")


def ed_gates(raw: Path, figures: Path) -> None:
    masks = pd.read_parquet(raw / "mech_masks.parquet")
    fig, ax = plt.subplots(figsize=(style.NATURE_SINGLE, 58 * MM))
    for (model, d), g in masks.groupby(["model", "distractor"]):
        x = np.sort(g["gate"].values)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=MODEL_COLORS[MODEL_SHORT[model]],
                ls="-" if d == "nonliteral" else "--", lw=0.9)
    ax.axvline(0.5, color=PALETTE["grey"], lw=0.6, ls=":")
    ax.text(0.5, 1.02, "selection threshold", ha="center", va="bottom", fontsize=5.5, color=PALETTE["grey"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Learned gate value")
    ax.set_ylabel("Fraction of heads (ECDF)")
    handles = [Line2D([], [], color=MODEL_COLORS[m], lw=1, label=m) for m in MODELS] + [
        Line2D([], [], color=PALETTE["slate"], lw=0.9, label="Nonliteral"), Line2D([], [], color=PALETTE["slate"], lw=0.9, ls="--", label="Bystander")]
    ax.legend(handles=handles, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.6, fontsize=5.5)
    style.save(fig, figures / "ed_mech_gates")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    args = ap.parse_args()
    style.set_style()
    fig4(args.results, args.figures)
    ed_prompting(args.results, args.figures)
    ed_attention(args.results, args.figures)
    ed_gates(args.results / "raw", args.figures)
    print(f"figures written to {args.figures}")


if __name__ == "__main__":
    main()
