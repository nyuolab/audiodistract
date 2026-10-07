"""Figure 2 and the MedDistractQA Extended Data figures from results/qa_*.csv.

    python -m llm_distract.meddistractqa.plot [--results results] [--figures figures]
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from llm_distract.common import style
from llm_distract.common.style import GROUP_COLORS, GROUP_MARKERS, NATURE_DOUBLE, PALETTE, place_panel_labels, save

DISTRACTORS = ("nonliteral", "bystander")
TITLES = {"nonliteral": "Nonliteral perturbation", "bystander": "Bystander perturbation"}
EXAMPLES = {
    "nonliteral": ["“The patient joked that their favorite mystery novel had a plot twist "
                   "that felt like a surgical incision into the storyline.”",
                   "“The patient described their friend's social circle as having a viral spread of gossip.”"],
    "bystander": ["“The patient's niece mentioned that her classmate's hamster was diagnosed "
                  "with a staph infection last month.”",
                  "“The patient's aunt mentioned that her friend takes indomethacin for her chronic condition.”"],
}
GROUP_ORDER = ["Open-weight, general", "Open-weight, medical", "Proprietary"]


def _bubble(ax, text: str, x: float, y: float, width: int = 46) -> None:
    """Speech-bubble style example sentence in axes coordinates (manually wrapped)."""
    ax.text(x, y, textwrap.fill(text, width), transform=ax.transAxes, fontsize=5.6, style="italic", ha="right", va="top",
            multialignment="left", bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.6", fc="white", ec=PALETTE["grey"], lw=0.6),
            linespacing=1.15)  # anchored to the right edge of the axes so the bubble never leaves the panel


def panel_bars(ax, mt: pd.DataFrame, distractor: str) -> None:
    d = mt.sort_values(f"loss_{distractor}_pp", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))[::-1]  # smallest loss at the top
    colors = [GROUP_COLORS[g] for g in d["group"]]
    ax.barh(y, d[f"loss_{distractor}_pp"], xerr=d[f"se_{distractor}_pp"], color=colors, edgecolor="black",
            linewidth=0.4, height=0.72, error_kw=dict(elinewidth=0.5, capsize=1.2, ecolor="black"))
    for yi, v, se in zip(y, d[f"loss_{distractor}_pp"], d[f"se_{distractor}_pp"]):
        ax.text(v + se + 0.25, yi, f"{v:.1f}", va="center", ha="left", fontsize=5.2)
    ax.set_yticks(y)
    ax.set_yticklabels(d["display_name"], fontsize=5.4)
    ax.set_xlim(0, 22)
    ax.set_xlabel("Accuracy loss (percentage points)")
    ax.set_title(TITLES[distractor], fontsize=8, loc="left")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E6E9EC", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.axvline(0, color="black", linewidth=0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=GROUP_COLORS[g], ec="black", lw=0.4) for g in GROUP_ORDER]
    ax.legend(handles, GROUP_ORDER, loc="upper left", bbox_to_anchor=(0.56, 0.66), fontsize=5.8, frameon=False,
              borderpad=0.3, handlelength=1.4)  # right of the longest value label at this height


def _fmt_p(p: float) -> str:
    """P value as mathtext, e.g. 2.1e-10 -> $P = 2.1 \\times 10^{-10}$."""
    mant, exp = f"{p:.1e}".split("e")
    return rf"$P = {mant} \times 10^{{{int(exp)}}}$"


def panel_groups(ax, gs: pd.DataFrame) -> None:
    g = gs[gs["kind"] == "group"]
    w = 0.36
    xs = np.arange(len(GROUP_ORDER))
    for i, dist in enumerate(DISTRACTORS):
        sub = g[g["distractor"] == dist].set_index("name").loc[GROUP_ORDER]
        offs = (i - 0.5) * w
        bars = ax.bar(xs + offs, sub["mean_loss_pp"], width=w, yerr=sub["sem_pp"], color=[GROUP_COLORS[n] for n in GROUP_ORDER],
                      edgecolor="black", linewidth=0.5, alpha=1.0 if dist == "nonliteral" else 0.55,
                      error_kw=dict(elinewidth=0.6, capsize=2), hatch=None if dist == "nonliteral" else "////")
        for b, m, se in zip(bars, sub["mean_loss_pp"], sub["sem_pp"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + se + 0.35, f"{m:.1f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n.replace(', ', chr(10))}\n(n = {int(g[(g['name'] == n) & (g['distractor'] == 'nonliteral')]['n'].iloc[0])})"
                        for n in GROUP_ORDER], fontsize=6.2)
    ax.set_ylabel("Mean accuracy loss (percentage points)")
    ax.set_ylim(0, 16)
    # significance annotations (Welch one-tailed, proprietary < others)
    w_ = gs[gs["kind"] == "welch"]
    p_nl = w_[(w_["distractor"] == "nonliteral") & (w_["name"] == "Proprietary < Open-weight, general")]["p_one_tailed"].iloc[0]
    p_by = w_[(w_["distractor"] == "bystander") & (w_["name"] == "Proprietary < Open-weight, general")]["p_one_tailed"].iloc[0]
    ax.text(0.98, 0.97, f"Proprietary < general:\n{_fmt_p(p_nl)} (nonliteral)\n{_fmt_p(p_by)} (bystander)",
            transform=ax.transAxes, fontsize=5.6, va="top", ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", lw=0.5),
               plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", lw=0.5, hatch="////")]
    ax.legend(handles, ["Nonliteral", "Bystander"], loc="upper left", fontsize=6, frameon=False)


def panel_scatter(ax, mt: pd.DataFrame, reg: pd.DataFrame) -> None:
    for dist, alpha in zip(DISTRACTORS, (1.0, 0.45)):
        for grp in GROUP_ORDER:
            sub = mt[mt["group"] == grp]
            ax.scatter(sub["acc_clean"], sub[f"loss_{dist}_pp"], marker=GROUP_MARKERS[grp], s=16,
                       facecolor=GROUP_COLORS[grp], edgecolor="black", linewidth=0.4, alpha=alpha,
                       label=grp if dist == "nonliteral" else None, zorder=3)
        r = reg[reg["distractor"] == dist].iloc[0]
        xx = np.linspace(mt["acc_clean"].min() - 2, mt["acc_clean"].max() + 2, 50)
        ax.plot(xx, r["intercept"] + r["slope"] * xx, ls="--" if dist == "nonliteral" else ":", color="#555555", lw=0.9,
                label=f"{TITLES[dist].split()[0]} fit, r = {r['r']:.2f}")
    ax.set_xlabel("Clean MedQA accuracy (%)")
    ax.set_ylabel("Accuracy loss (percentage points)")
    ax.set_ylim(-0.5, 20)
    ax.legend(fontsize=5.4, loc="upper right", frameon=False, borderaxespad=0.2, handletextpad=0.3, labelspacing=0.25)


def fig2(results: Path, figures: Path) -> None:
    mt = pd.read_csv(results / "qa_model_table.csv")
    gs = pd.read_csv(results / "qa_group_summary.csv")
    reg = pd.read_csv(results / "qa_regression.csv")
    fig = plt.figure(figsize=(NATURE_DOUBLE, 7.6))
    # two grids: the bottom row starts further left so that panel c's axis label lines up with panel a's model names
    gs_top = fig.add_gridspec(1, 2, wspace=0.55, left=0.17, right=0.985, top=0.965, bottom=0.424)
    gs_bot = fig.add_gridspec(1, 2, wspace=0.30, left=0.075, right=0.985, top=0.30, bottom=0.07)
    axa, axb = fig.add_subplot(gs_top[0, 0]), fig.add_subplot(gs_top[0, 1])
    axc, axd = fig.add_subplot(gs_bot[0, 0]), fig.add_subplot(gs_bot[0, 1])
    panel_bars(axa, mt, "nonliteral")
    panel_bars(axb, mt, "bystander")
    for ax, dist in ((axa, "nonliteral"), (axb, "bystander")):
        _bubble(ax, EXAMPLES[dist][0], 0.995, 0.985)
        _bubble(ax, EXAMPLES[dist][1], 0.995, 0.855)
    panel_groups(axc, gs)
    panel_scatter(axd, mt, reg)
    place_panel_labels(fig, [("a", axa, 0, 0), ("b", axb, 1, 0), ("c", axc, 0, 1), ("d", axd, 1, 1)])
    save(fig, figures / "fig2")


def ed_family(results: Path, figures: Path) -> None:
    fam = pd.read_csv(results / "qa_family_llama3.csv")
    fam = fam[fam["kind"] == "model"]
    order = ["Llama-3-8B-Instruct", "Llama-3-Meerkat-8B", "Llama-3-8B-UltraMedical", "Llama-3.1-8B-Instruct", "DeepSeek-R1-Distill-Llama-8B"]
    names = [n for n in order if n in set(fam["display_name"])]
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE, 2.4), gridspec_kw=dict(wspace=0.35))
    xs = np.arange(len(names))
    a = fam[fam["metric"] == "acc_clean"].set_index("display_name").loc[names]
    axes[0].bar(xs, a["estimate_pp"], yerr=[a["estimate_pp"] - a["ci_low"], a["ci_high"] - a["estimate_pp"]],
                color=PALETTE["slate"], edgecolor="black", linewidth=0.5, width=0.6, error_kw=dict(elinewidth=0.6, capsize=2))
    axes[0].set_ylabel("Clean MedQA accuracy (%)")
    axes[0].set_ylim(0, 80)
    w = 0.36
    for i, dist in enumerate(DISTRACTORS):
        m = fam[fam["metric"] == f"loss_{dist}_pp"].set_index("display_name").loc[names]
        axes[1].bar(xs + (i - 0.5) * w, m["estimate_pp"], width=w, yerr=[m["estimate_pp"] - m["ci_low"], m["ci_high"] - m["estimate_pp"]],
                    color=style.DISTRACTOR_COLORS[dist], edgecolor="black", linewidth=0.5, label=TITLES[dist].split()[0],
                    error_kw=dict(elinewidth=0.6, capsize=2))
    axes[1].set_ylabel("Accuracy loss (percentage points)")
    axes[1].legend(frameon=False, fontsize=6)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=6)
    place_panel_labels(fig, [("a", axes[0], 0, 0), ("b", axes[1], 1, 0)])
    save(fig, figures / "ed_qa_family")


def ed_prompting(results: Path, figures: Path) -> None:
    pt = pd.read_csv(results / "qa_prompting.csv")
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE, 3.0), gridspec_kw=dict(wspace=0.55, width_ratios=[1.1, 1]))
    for ax, variant, letter in ((axes[0], "ignore", "a"), (axes[1], "structured", "b")):
        d = pt[(pt["prompt_variant"] == variant) & (pt["condition"] == "nonliteral")].sort_values("delta_pp").reset_index(drop=True)
        y = np.arange(len(d))
        colors = [GROUP_COLORS[g] for g in d["group"]]
        ax.barh(y, d["delta_pp"], color=colors, edgecolor="black", linewidth=0.4, height=0.7)
        for yi, (p, v, fc) in enumerate(zip(d["mcnemar_p"], d["delta_pp"], d["format_collapse"])):
            mark = ("*" if p < 0.05 else "") + (" †" if fc else "")
            if mark:
                ax.text(v + (0.3 if v >= 0 else -0.3), yi, mark, va="center", ha="left" if v >= 0 else "right", fontsize=6)
        ax.set_yticks(y)
        ax.set_yticklabels(d["display_name"], fontsize=5.4)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Change in accuracy on nonliteral questions (percentage points)")
        ax.set_title("‘Ignore irrelevant information’ instruction" if variant == "ignore" else "Structured prompt warning about extraneous details",
                     fontsize=7, loc="left")
        ax.tick_params(axis="y", length=0)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - 0.14 * (hi - lo), hi + 0.06 * (hi - lo))  # keep the markers clear of the model names
        ax.margins(y=0.02)
    axes[1].text(1.0, -0.2, "* exact McNemar P < 0.05; † >10% not-evaluable outputs", transform=axes[1].transAxes,
                 fontsize=5.6, ha="right", va="top")
    place_panel_labels(fig, [("a", axes[0], 0, 0), ("b", axes[1], 1, 0)])
    save(fig, figures / "ed_qa_prompting")


def _heatmap(ax, cat: pd.DataFrame, mt: pd.DataFrame, title: str, rotation: int = 90, cbar_width: float = 0.04) -> None:
    d = cat[(cat["distractor"] == "mean")]
    piv = d.pivot(index="engine", columns="category", values="loss_pp")
    n_items = d.groupby("category")["n_items"].first()
    reported = d.groupby("category")["reported"].first()
    cols = [c for c in piv.columns if reported[c]]
    piv = piv[cols]
    order = mt.sort_values("acc_clean", ascending=False)["engine"]
    piv = piv.loc[[e for e in order if e in piv.index]]
    names = mt.set_index("engine")["display_name"]
    im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=-5, vmax=20, aspect="auto")
    ax.set_xticks(range(len(cols)))
    labels = [f"{c} (n = {int(n_items[c])})" for c in cols]
    if rotation == 90:
        ax.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=5.2)
    else:
        ax.set_xticklabels(labels, rotation=rotation, ha="right", rotation_mode="anchor", fontsize=5.2)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([names[e] for e in piv.index], fontsize=4.8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=4.2, color="black")
    ax.set_title(title, fontsize=7, loc="left")
    ax.tick_params(length=0)
    cax = ax.inset_axes([1.03, 0.3, cbar_width, 0.4])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("Mean accuracy loss (pp)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)


def ed_categories(results: Path, figures: Path) -> None:
    mt = pd.read_csv(results / "qa_model_table.csv")
    comp = pd.read_csv(results / "qa_category_competency.csv")
    syst = pd.read_csv(results / "qa_category_system.csv")
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE, 5.6), gridspec_kw=dict(wspace=0.9, width_ratios=[0.7, 1.4]))
    _heatmap(axes[0], comp, mt, "By USMLE physician competency", rotation=40, cbar_width=0.07)
    _heatmap(axes[1], syst, mt, "By USMLE organ system", rotation=90, cbar_width=0.035)
    place_panel_labels(fig, [("a", axes[0], 0, 0), ("b", axes[1], 1, 0)])
    save(fig, figures / "ed_qa_categories")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    args = ap.parse_args()
    style.set_style()
    fig2(args.results, args.figures)
    ed_family(args.results, args.figures)
    ed_prompting(args.results, args.figures)
    ed_categories(args.results, args.figures)
    print("figures written to", args.figures)


if __name__ == "__main__":
    main()
