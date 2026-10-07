"""Figure 3 and the ambient Extended Data figures from results/amb_*.csv.

    python -m llm_distract.ambient.plot [--results results] [--figures figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from llm_distract.common import style
from llm_distract.common.style import DISTRACTOR_COLORS, MODEL_COLORS, NATURE_DOUBLE, PALETTE, place_panel_labels, save

MODELS = ["Gemma-2-9B", "Mistral-7B", "Llama-3.1-8B", "Qwen2.5-7B"]  # open-weight models; API models follow when present
DISTS = ["nonliteral", "bystander"]


def models_in(summ: pd.DataFrame) -> list:
    """Models of the summary table in display order (open-weight first, then the API models)."""
    names = list(summ[summ["level"] == "model"]["name"])
    return [m for m in MODELS if m in names] + [m for m in names if m not in MODELS]


def _group_separator(ax, models: list, step: float = 1.0, offset: float = 0.0) -> None:
    """Dotted line between the open-weight block and the API models (when both are present)."""
    n_open = sum(m in MODELS for m in models)
    if 0 < n_open < len(models):
        ax.axhline((n_open - 0.5) * step + offset, color=PALETTE["grey"], lw=0.5, ls=":", zorder=1)
DLABEL = {"nonliteral": "Nonliteral", "bystander": "Bystander"}
QUALITY_LABEL = {"clinical_correctness": "Clinical\ncorrectness", "completeness": "Completeness", "succinctness": "Succinctness",
                 "hallucination": "Hallucination\n(0/1 flag)", "overall_quality": "Overall\nquality"}


def _ypos(model: str, dist: str, models: list = MODELS) -> float:
    return models.index(model) * 2.2 + (0.45 if dist == "bystander" else -0.45)


def panel_rates(ax, summ: pd.DataFrame) -> None:
    """Clean vs distracted contamination rate (pooled) per model x distractor, as a dumbbell."""
    models = models_in(summ)
    s = summ[summ["level"] == "model x distractor"]
    for _, r in s.iterrows():
        mdl, dist = r["name"].split(" | ")
        y = _ypos(mdl, dist, models)
        ax.plot([r["rate_clean"], r["rate_distracted"]], [y, y], color=DISTRACTOR_COLORS[dist], lw=1.2, zorder=2)
        ax.scatter([r["rate_clean"]], [y], s=16, facecolor="white", edgecolor=DISTRACTOR_COLORS[dist], lw=0.9, zorder=3)
        ax.scatter([r["rate_distracted"]], [y], s=18, facecolor=DISTRACTOR_COLORS[dist], edgecolor="black", lw=0.4, zorder=3)
        ax.text(r["rate_distracted"] + 2.6, y, f"{r['rate_distracted']:.0f}%", va="center", fontsize=5.6)
    ax.set_yticks([i * 2.2 for i in range(len(models))])
    ax.set_yticklabels(models)
    _group_separator(ax, models, step=2.2)
    ax.invert_yaxis()
    ax.set_xlim(-2, 90 if s["rate_distracted"].max() <= 80 else 104)
    ax.set_xlabel("Notes incorporating the inserted content (%)")
    ax.tick_params(axis="y", length=0)
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="black", ms=4), plt.Line2D([], [], marker="o", ls="", mfc="black", mec="black", ms=4),
         plt.Line2D([], [], color=DISTRACTOR_COLORS["nonliteral"], lw=1.5), plt.Line2D([], [], color=DISTRACTOR_COLORS["bystander"], lw=1.5)]
    ax.legend(h, ["Clean note", "Distracted note", "Nonliteral", "Bystander"], loc="lower left", bbox_to_anchor=(0.0, 1.0),
              fontsize=5.6, frameon=False, ncol=2, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.0, labelspacing=0.25)


def panel_delta(ax, summ: pd.DataFrame) -> None:
    """Pooled distracted-minus-clean increase with CI, per model and distractor; reference lines for the overall estimates."""
    models = models_in(summ)
    s = summ[summ["level"] == "model x distractor"]
    for _, r in s.iterrows():
        mdl, dist = r["name"].split(" | ")
        y = _ypos(mdl, dist, models)
        ax.errorbar(r["pooled_delta_pp"], y, xerr=[[r["pooled_delta_pp"] - r["pooled_ci_low"]], [r["pooled_ci_high"] - r["pooled_delta_pp"]]],
                    fmt="o", ms=4, color=DISTRACTOR_COLORS[dist], ecolor=DISTRACTOR_COLORS[dist], elinewidth=0.8, capsize=1.5, mec="black", mew=0.4)
    o = summ[summ["level"] == "overall"].iloc[0]
    ax.axvline(o["pooled_delta_pp"], color=PALETTE["slate"], lw=0.8, ls="--")
    ax.axvline(o["unweighted_mean_pp"], color=PALETTE["grey"], lw=0.8, ls=":")
    ax.text(o["pooled_delta_pp"] + 1.2, -1.4, f"pooled\n{o['pooled_delta_pp']:.1f} pp", fontsize=5.4, color=PALETTE["slate"], ha="left", va="bottom")
    ax.text(o["unweighted_mean_pp"] - 1.2, -1.4, f"stratum mean\n{o['unweighted_mean_pp']:.1f} pp", fontsize=5.4, color=PALETTE["grey"], ha="right", va="bottom")
    ax.set_yticks([i * 2.2 for i in range(len(models))])
    ax.set_yticklabels(models)
    _group_separator(ax, models, step=2.2)
    ax.set_ylim(len(models) * 2.2 - 1.2, -2.6)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Increase in contamination (percentage points)")
    ax.tick_params(axis="y", length=0)


def panel_heatmap(ax, summ: pd.DataFrame) -> None:
    models = models_in(summ)
    s = summ[summ["level"] == "model x family x distractor"]
    cols = [(fam, dist) for fam in ("ACI-Bench", "MTS-Dialog") for dist in DISTS]
    mat = np.zeros((len(models), len(cols)))
    for i, mdl in enumerate(models):
        for j, (fam, dist) in enumerate(cols):
            mat[i, j] = s[s["name"] == f"{mdl} | {fam} | {dist}"]["pooled_delta_pp"].iloc[0]
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=80, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="white" if mat[i, j] > 45 else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{fam}\n{DLABEL[d].lower()}" for fam, d in cols], fontsize=5.2, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    n_open = sum(m in MODELS for m in models)
    if 0 < n_open < len(models):
        ax.axhline(n_open - 0.5, color="white", lw=1.5)
    ax.tick_params(length=0)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Increase (pp)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)


COHORT_STYLE = {"open-weight": (PALETTE["slate"], "o", "Open-weight"), "API": (PALETTE["proprietary"], "s", "Frontier")}


def panel_length(ax, length: pd.DataFrame, cohorts: list | None = None) -> None:
    """Increase in contamination by clean-transcript-length quintile within MTS-Dialog, one line per cohort in ``cohorts``
    (default: the open-weight cohort only, as in the first-generation figure)."""
    d = length[length["scope"] == "MTS-Dialog"].copy()
    if "cohort" not in d:
        d["cohort"] = "open-weight"
    cohorts = [c for c in (cohorts or ["open-weight"]) if (d["cohort"] == c).any()]
    for k, cohort in enumerate(cohorts):
        c = d[d["cohort"] == cohort]
        color, marker, label = COHORT_STYLE[cohort]
        x = c["quintile"].values + (0.0 if len(cohorts) == 1 else (k - 0.5 * (len(cohorts) - 1)) * 0.12)
        ax.errorbar(x, c["pooled_delta_pp"], yerr=[c["pooled_delta_pp"] - c["ci_low"], c["ci_high"] - c["pooled_delta_pp"]],
                    fmt=f"{marker}-", color=color, ms=4, lw=1, capsize=2, mec="black", mew=0.4, label=f"{label} (ρ = {c['spearman_rho'].iloc[0]:.2f})")
    ref = d[d["cohort"] == cohorts[0]]
    x = ref["quintile"].values
    for xi, m in zip(x, ref["median_transcript_chars"]):
        ax.text(xi, -6, f"{int(m):,}", ha="center", fontsize=5.4, color=PALETTE["grey"])
    ax.text(0.5, -13.5, "median transcript length (characters)", fontsize=5.4, color=PALETTE["grey"], ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels(["shortest", "2", "3", "4", "longest"], fontsize=6)
    ax.set_xlabel("Clean-transcript length quintile (MTS-Dialog)")
    ax.set_ylabel("Increase in contamination (pp)")
    ax.set_ylim(-16, max(80, float(d[d["cohort"].isin(cohorts)]["ci_high"].max()) + 5))
    if len(cohorts) == 1:
        ax.text(0.98, 0.95, f"Spearman ρ = {ref['spearman_rho'].iloc[0]:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=6)
    else:
        ax.legend(fontsize=5.4, frameon=False, loc="upper right", handletextpad=0.4, borderaxespad=0.2, labelspacing=0.3)


def panel_quality(ax, qual: pd.DataFrame) -> None:
    metrics = list(QUALITY_LABEL)
    if "cohort" in qual:
        qual = qual[qual["cohort"] == "open-weight"]
    q = qual.set_index("metric").loc[metrics]
    x = np.arange(len(metrics))
    ax.errorbar(x, q["pooled_delta"], yerr=[q["pooled_delta"] - q["pooled_ci_low"], q["pooled_ci_high"] - q["pooled_delta"]],
                fmt="o", color=PALETTE["slate"], ms=4, capsize=2, lw=0.8, mec="black", mew=0.4)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([QUALITY_LABEL[m].replace(chr(10), " ") for m in metrics], fontsize=5.4, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Distracted minus clean score")
    hall = q.loc["hallucination"]
    ax.text(x[3], 0.55, f"flag raised in {100 * hall['clean_mean']:.0f}%\nof clean notes", ha="center", fontsize=5.4, color=PALETTE["grey"])


def panel_audio(ax, results: Path) -> None:
    cond = pd.read_csv(results / "audio_contamination_given_leakage.csv").sort_values("level_db")
    cont = pd.read_csv(results / "audio_contamination_by_level_model.csv")
    pooled = cont[(cont["model"] == "all") & (cont["level_db"].astype(str) != "all")]
    for leaked, color, label in ((True, DISTRACTOR_COLORS["bystander"], "Words leaked into transcript"),
                                 (False, PALETTE["grey"], "No leaked word")):
        g = cond[cond["leaked"] == leaked]
        ax.errorbar(g["level_db"], g["rate_mixed"], yerr=[g["rate_mixed"] - g["rate_mixed_ci_low"], g["rate_mixed_ci_high"] - g["rate_mixed"]],
                    fmt="o-", color=color, ms=4, lw=1, capsize=2, mec="black", mew=0.4, label=label)
    ax.axhline(pooled["rate_clean"].mean(), ls="--", color=PALETTE["clean"], lw=0.8, label="Clean recordings")
    ax.set_xticks(sorted(cond["level_db"].unique()))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Background consultation level (dB re foreground)")
    ax.set_ylabel("Notes incorporating the other\npatient's content (%)")
    ax.legend(loc="upper left", fontsize=5.4, frameon=False)


def fig3(results: Path, figures: Path) -> None:
    summ = pd.read_csv(results / "amb_summary.csv")
    length = pd.read_csv(results / "amb_length.csv")
    qual = pd.read_csv(results / "amb_quality.csv")
    extra = max(0, len(models_in(summ)) - 4)  # taller top row when the API models are present
    fig = plt.figure(figsize=(NATURE_DOUBLE, 5.4 + 0.42 * extra))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15 + 0.3 * extra, 1], hspace=0.55 - 0.03 * extra, wspace=0.55, left=0.09, right=0.98,
                          top=0.95, bottom=0.11 - 0.01 * extra)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    axd, axe = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    axf = fig.add_subplot(gs[1, 2])
    panel_rates(axa, summ)
    panel_delta(axb, summ)
    panel_heatmap(axc, summ)
    panel_length(axd, length)
    panel_quality(axe, qual)
    # panel f: the acoustic route (PriMock57) - contaminated notes from mixed recordings by background level,
    # split by whether any word of the overlaid segment leaked into the transcript (full version: ED audio figure)
    panel_audio(axf, results)
    place_panel_labels(fig, [("a", axa, 0, 0), ("b", axb, 1, 0), ("c", axc, 2, 0), ("d", axd, 0, 1), ("e", axe, 1, 1), ("f", axf, 2, 1)])
    save(fig, figures / "fig3")


def _split_label(split: str) -> str:
    """'mts_dialog_test1' -> 'MTS-Dialog test 1', 'aci_bench_valid' -> 'ACI-Bench validation'."""
    s = split.replace("mts_dialog", "MTS-Dialog").replace("aci_bench", "ACI-Bench").replace("_", " ")
    s = s.replace(" valid", " validation").replace("test1", "test 1").replace("test2", "test 2").replace("test3", "test 3")
    return s


def ed_strata(results: Path, figures: Path) -> None:
    strata = pd.read_csv(results / "amb_strata.csv")
    strata["label"] = strata["model"] + " | " + strata["split"].map(_split_label) + " | " + strata["distractor"].map(DLABEL)
    strata = strata.sort_values("delta_pp").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE * 0.75, max(6.2, 6.2 * len(strata) / 40)))
    y = np.arange(len(strata))
    for yi, r in strata.iterrows():
        ax.errorbar(r["delta_pp"], yi, xerr=[[r["delta_pp"] - r["ci_low"]], [r["ci_high"] - r["delta_pp"]]], fmt="o", ms=3,
                    color=DISTRACTOR_COLORS[r["distractor"]], ecolor=DISTRACTOR_COLORS[r["distractor"]], elinewidth=0.6, capsize=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(strata["label"], fontsize=5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Distracted minus clean contamination (percentage points), 95% paired bootstrap CI")
    ax.tick_params(axis="y", length=0)
    save(fig, figures / "ed_amb_strata")


def ed_judges(results: Path, figures: Path) -> None:
    js = pd.read_csv(results / "amb_judge_sensitivity.csv")
    js = js[js["pooled_delta_pp"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE * 0.7, 2.2), gridspec_kw=dict(wspace=0.5))
    x = np.arange(len(js))
    axes[0].bar(x, js["pooled_delta_pp"], yerr=[js["pooled_delta_pp"] - js["pooled_ci_low"], js["pooled_ci_high"] - js["pooled_delta_pp"]],
                color=[PALETTE["slate"], PALETTE["grey"]], edgecolor="black", lw=0.5, width=0.6, error_kw=dict(capsize=2, elinewidth=0.6))
    axes[0].set_ylabel("Pooled increase (pp)")
    axes[1].bar(x, js["clean_false_positive_rate_pct"], color=[PALETTE["slate"], PALETTE["grey"]], edgecolor="black", lw=0.5, width=0.6)
    axes[1].set_ylabel("Clean notes flagged (%)")
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(["Paired judge\n(Claude Sonnet 5)", "Single-note judge\n(GPT-5.4)"], fontsize=6)
    place_panel_labels(fig, [("a", axes[0], 0, 0), ("b", axes[1], 1, 0)])
    save(fig, figures / "ed_amb_judges")


def ed_attribution(results: Path, figures: Path) -> None:
    """Extended Data: per model and perturbation family, the share of encounters in which the distracted note misattributed
    the inserted content to the patient, recorded it correctly but used it for the patient, or only recorded it."""
    path = results / "amb_attribution.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    t = t[t["level"] == "model x distractor"]
    names = [n.split(" | ")[0] for n in t["name"]]
    models = [m for m in MODELS if m in set(names)] + [m for m in dict.fromkeys(names) if m not in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE * 0.85, 0.32 * len(models) + 1.6), sharey=True, gridspec_kw=dict(wspace=0.12))
    parts = [("misattributed_pct", "#B2182B", "Presented as the patient's own"), ("used_pct", "#EF8A62", "Correctly attributed, used in assessment/plan"),
             ("recorded_only_pct", "#D9D9D9", "Recorded only")]
    for ax, dist in zip(axes, DISTS):
        y = np.arange(len(models))
        left = np.zeros(len(models))
        for col, color, label in parts:
            vals = np.array([t[t["name"] == f"{m} | {dist}"][col].iloc[0] if (t["name"] == f"{m} | {dist}").any() else 0 for m in models])
            ax.barh(y, vals, left=left, color=color, edgecolor="black", lw=0.4, label=label)
            left += vals
        for yi, m in enumerate(models):
            r = t[t["name"] == f"{m} | {dist}"]
            if len(r):
                ax.text(left[yi] + 1, yi, f"{r['misattributed_pct'].iloc[0]:.0f}% misattributed", va="center", fontsize=5.4, color=PALETTE["slate"])
        ax.set_yticks(y); ax.set_yticklabels(models)
        if not ax.yaxis_inverted():
            ax.invert_yaxis()  # shared y axis: invert once
        _group_separator(ax, models, step=1.0)
        ax.set_xlim(0, 118); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_xlabel("Distracted notes (% of encounters)"); ax.set_title(DLABEL[dist], fontsize=7, loc="left")
        ax.tick_params(axis="y", length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=5.6, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.2)
    place_panel_labels(fig, [("a", axes[0], 0, 0), ("b", axes[1], 1, 0)])
    save(fig, figures / "ed_amb_attribution")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    args = ap.parse_args()
    style.set_style()
    fig3(args.results, args.figures)
    ed_strata(args.results, args.figures)
    ed_judges(args.results, args.figures)
    ed_attribution(args.results, args.figures)
    print("figures written to", args.figures)


if __name__ == "__main__":
    main()
