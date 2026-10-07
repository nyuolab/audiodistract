"""Main documentation figures for the adjacent (v3) insertion set: Fig. 2 (incorporation, attribution outcomes, corpus,
comparison with the first-generation set, transcript length, acoustic route) and Fig. 3 (failure modes: taxonomy, where the
content lands in the note, severity; the verbatim cases stay in Table 1 and results/v3/amb_failure_examples.md).

    python -m llm_distract.ambient.plot_v3 [--results results/v3] [--baseline results] [--figures figures]
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_distract.ambient.plot import DISTS, DLABEL, MODELS, _group_separator, _ypos, models_in, panel_audio, panel_heatmap, panel_length
from llm_distract.common import style
from llm_distract.common.style import DISTRACTOR_COLORS, NATURE_DOUBLE, PALETTE, place_panel_labels, save

OUTCOME_COLORS = {"misattributed": "#B2182B", "acted_on": "#EF8A62", "recorded_only": "#D9D9D9"}
OUTCOME_LABELS = {"misattributed": "Presented as the patient's own", "acted_on": "Correctly attributed, used in assessment/plan", "recorded_only": "Recorded only"}
MODE_ORDER = ["substituted_history", "family_history_inflation", "inferred_concern", "pathologised_conversation", "literalised_remark", "recorded_as_patient_activity", "other"]
MODE_LABELS = {"substituted_history": "Other person's condition becomes\nthe patient's complaint or diagnosis",
               "family_history_inflation": "Recorded as family history\nor risk factor",
               "inferred_concern": "Read as an unvoiced concern:\nscreening or counselling added",
               "pathologised_conversation": "Small talk documented as a\nmental-status or psychiatric sign",
               "literalised_remark": "Figurative remark recorded as\na fact about the patient",
               "recorded_as_patient_activity": "Recorded as the patient's\nown activity or social history",
               "other": "Other"}
MODE_COLORS = {"substituted_history": "#B2182B", "family_history_inflation": "#D6604D", "inferred_concern": "#F4A582", "pathologised_conversation": "#762A83",
               "literalised_remark": "#C2A5CF", "recorded_as_patient_activity": "#5AAE61", "other": "#BFC7CF"}
SECTIONS = ["Chief complaint / HPI", "History (PMH/meds/ROS)", "Family / social history", "Subjective (other)", "Assessment", "Plan"]
SECTION_LABELS = {"Chief complaint / HPI": "Chief complaint / HPI", "History (PMH/meds/ROS)": "Past history / meds / ROS", "Family / social history": "Family or social history",
                  "Subjective (other)": "Subjective, unsectioned", "Assessment": "Assessment", "Plan": "Plan"}


def panel_outcomes(ax, attribution: pd.DataFrame, models: list, dist: str, annotate: bool = True) -> None:
    """Stacked horizontal bars: misattributed / acted on / recorded only, as % of encounters, per model."""
    t = attribution[attribution["level"] == "model x distractor"]
    y = np.arange(len(models))
    left = np.zeros(len(models))
    for key, col in (("misattributed", "misattributed_pct"), ("acted_on", "used_pct"), ("recorded_only", "recorded_only_pct")):
        vals = np.array([t[t["name"] == f"{m} | {dist}"][col].iloc[0] if (t["name"] == f"{m} | {dist}").any() else 0.0 for m in models])
        ax.barh(y, vals, left=left, color=OUTCOME_COLORS[key], edgecolor="black", lw=0.4, label=OUTCOME_LABELS[key], height=0.68)
        left += vals
    if annotate:
        for yi, m in enumerate(models):
            r = t[t["name"] == f"{m} | {dist}"]
            if len(r):
                ax.text(left[yi] + 1.5, yi, f"{r['misattributed_pct'].iloc[0]:.1f}% misattributed", va="center", fontsize=5.2, color=PALETTE["slate"])
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    _group_separator(ax, models, step=1.0)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Distracted notes (% of encounters)")
    ax.set_title(DLABEL[dist], fontsize=7, loc="left")
    ax.tick_params(axis="y", length=0)


def panel_generation_compare(ax, summ_v3: pd.DataFrame, attr_v3: pd.DataFrame, summ_v1: pd.DataFrame, attr_v1: pd.DataFrame, models: list) -> None:
    """First-generation (pneumonia / Contagion) set versus adjacent set: incorporation and misattribution per model."""
    y = np.arange(len(models))
    for i, m in enumerate(models):
        for (summ, attr, marker, fill, lab) in ((summ_v1, attr_v1, "o", "white", "first-generation set"), (summ_v3, attr_v3, "o", None, "adjacent set")):
            s = summ[(summ["level"] == "model") & (summ["name"] == m)]
            a = attr[(attr["level"] == "model") & (attr["name"] == m)]
            if not len(s):
                continue
            inc = s["rate_distracted"].iloc[0]
            mis = a["misattributed_pct"].iloc[0] if len(a) else np.nan
            ax.scatter([inc], [i - 0.17], s=20, marker=marker, facecolor=fill or PALETTE["slate"], edgecolor=PALETTE["slate"], lw=0.8, zorder=3)
            ax.scatter([mis], [i + 0.17], s=20, marker="s", facecolor=fill or OUTCOME_COLORS["misattributed"], edgecolor=OUTCOME_COLORS["misattributed"], lw=0.8, zorder=3)
        # connecting arrows
        s1 = summ_v1[(summ_v1["level"] == "model") & (summ_v1["name"] == m)]
        s3 = summ_v3[(summ_v3["level"] == "model") & (summ_v3["name"] == m)]
        a1 = attr_v1[(attr_v1["level"] == "model") & (attr_v1["name"] == m)]
        a3 = attr_v3[(attr_v3["level"] == "model") & (attr_v3["name"] == m)]
        if len(s1) and len(s3):
            ax.annotate("", xy=(s3["rate_distracted"].iloc[0], i - 0.17), xytext=(s1["rate_distracted"].iloc[0], i - 0.17),
                        arrowprops=dict(arrowstyle="-|>", color=PALETTE["slate"], lw=0.7, mutation_scale=6), zorder=2)
        if len(a1) and len(a3):
            ax.annotate("", xy=(a3["misattributed_pct"].iloc[0], i + 0.17), xytext=(a1["misattributed_pct"].iloc[0], i + 0.17),
                        arrowprops=dict(arrowstyle="-|>", color=OUTCOME_COLORS["misattributed"], lw=0.7, mutation_scale=6), zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    _group_separator(ax, models, step=1.0)
    ax.set_xlim(-2, 100)
    ax.set_xlabel("Distracted notes (% of encounters)")
    ax.tick_params(axis="y", length=0)
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=PALETTE["slate"], ms=4), plt.Line2D([], [], marker="o", ls="", mfc=PALETTE["slate"], mec=PALETTE["slate"], ms=4),
         plt.Line2D([], [], marker="s", ls="", mfc="white", mec=OUTCOME_COLORS["misattributed"], ms=4), plt.Line2D([], [], marker="s", ls="", mfc=OUTCOME_COLORS["misattributed"], mec=OUTCOME_COLORS["misattributed"], ms=4)]
    ax.legend(h, ["Incorporated, first-generation set", "Incorporated, adjacent set", "Misattributed, first-generation set", "Misattributed, adjacent set"],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=5.2, frameon=False, ncol=2, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0, labelspacing=0.25)


def panel_rates_v3(ax, summ: pd.DataFrame, models: list) -> None:
    s = summ[summ["level"] == "model x distractor"]
    for _, r in s.iterrows():
        mdl, dist = r["name"].split(" | ")
        y = _ypos(mdl, dist, models)
        ax.plot([r["rate_clean"], r["rate_distracted"]], [y, y], color=DISTRACTOR_COLORS[dist], lw=1.2, zorder=2)
        ax.scatter([r["rate_clean"]], [y], s=16, facecolor="white", edgecolor=DISTRACTOR_COLORS[dist], lw=0.9, zorder=3)
        ax.errorbar(r["rate_distracted"], y, xerr=[[r["rate_distracted"] - r["pooled_ci_low"]], [r["pooled_ci_high"] - r["rate_distracted"]]], fmt="o", ms=4,
                    color=DISTRACTOR_COLORS[dist], ecolor=DISTRACTOR_COLORS[dist], elinewidth=0.7, capsize=1.2, mec="black", mew=0.4, zorder=3)
        ax.text(r["pooled_ci_high"] + 2.0, y, f"{r['rate_distracted']:.0f}%", va="center", fontsize=5.6)
    ax.set_yticks([i * 2.2 for i in range(len(models))])
    ax.set_yticklabels(models)
    _group_separator(ax, models, step=2.2)
    ax.invert_yaxis()
    ax.set_xlim(-2, 100)
    ax.set_xlabel("Notes incorporating the inserted content (%)")
    ax.tick_params(axis="y", length=0)
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="black", ms=4), plt.Line2D([], [], marker="o", ls="", mfc="black", mec="black", ms=4),
         plt.Line2D([], [], color=DISTRACTOR_COLORS["nonliteral"], lw=1.5), plt.Line2D([], [], color=DISTRACTOR_COLORS["bystander"], lw=1.5)]
    ax.legend(h, ["Clean note", "Distracted note (95% CI)", "Nonliteral", "Bystander"], loc="lower left", bbox_to_anchor=(0.0, 1.0),
              fontsize=5.4, frameon=False, ncol=2, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.0, labelspacing=0.25)


def fig_documentation(results: Path, baseline: Path, figures: Path, name: str = "fig2_v3") -> None:
    summ = pd.read_csv(results / "amb_summary.csv")
    summ = summ[summ["n_pairs"] > 0]
    attr = pd.read_csv(results / "amb_attribution.csv")
    length = pd.read_csv(results / "amb_length.csv")
    summ_v1 = pd.read_csv(baseline / "amb_summary.csv")
    attr_v1 = pd.read_csv(baseline / "amb_attribution.csv")
    models = models_in(summ)
    n = len(models)
    fig = plt.figure(figsize=(NATURE_DOUBLE, 5.2 + 0.28 * max(0, n - 4)))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0 + 0.12 * max(0, n - 4), 1], hspace=0.6, wspace=0.62, left=0.10, right=0.985, top=0.93, bottom=0.10)
    axa, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])
    gsb = gs[0, 1].subgridspec(1, 2, wspace=0.12)
    axb1, axb2 = fig.add_subplot(gsb[0, 0]), fig.add_subplot(gsb[0, 1], sharey=None)
    axd, axe, axf = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    panel_rates_v3(axa, summ, models)
    panel_outcomes(axb1, attr, models, "nonliteral", annotate=False)
    panel_outcomes(axb2, attr, models, "bystander", annotate=False)
    axb2.set_yticklabels([])
    for axx in (axb1, axb2):
        axx.set_xlim(0, 100)
        axx.set_xlabel("")
    axb1.set_xticks([0, 50])
    axb2.set_xticks([0, 50, 100])
    axb1.set_xlabel("Distracted notes (% of encounters)", x=1.05)
    axb = axb1
    handles, labels = axb1.get_legend_handles_labels()
    axb1.legend(handles, labels, fontsize=4.9, loc="lower left", bbox_to_anchor=(0.0, 1.12), frameon=False, ncol=1, handletextpad=0.4, borderaxespad=0.0, labelspacing=0.2)
    panel_heatmap(axc, summ)
    panel_generation_compare(axd, summ, attr, summ_v1, attr_v1, models)
    present = [c for c in ("open-weight", "API") if "cohort" in length and length[length["cohort"] == c]["n_pairs"].sum() > 0]
    panel_length(axe, length, cohorts=present or None)  # both cohorts when both are folded
    panel_audio(axf, baseline)
    place_panel_labels(fig, [("a", axa, 0, 0), ("b", [axb1, axb2], 1, 0), ("c", axc, 2, 0), ("d", axd, 0, 1), ("e", axe, 1, 1), ("f", axf, 2, 1)])
    save(fig, figures / name)


# ---------------------------------------------------------------------------------------------------------- failure modes
def panel_taxonomy(ax, tax: pd.DataFrame, n_enc: pd.Series, models: list) -> None:
    """Per model: misattributed + acted-on notes as % of encounters, stacked by failure mode (both families pooled)."""
    t = tax[tax["model_short"] != "all"].groupby(["model_short", "failure_mode"])["n"].sum().unstack(fill_value=0)
    y = np.arange(len(models))
    left = np.zeros(len(models))
    for mode in MODE_ORDER:
        vals = np.array([100 * t.loc[m, mode] / n_enc[m] if (m in t.index and mode in t.columns) else 0.0 for m in models])
        ax.barh(y, vals, left=left, color=MODE_COLORS[mode], edgecolor="black", lw=0.4, label=MODE_LABELS[mode].replace("\n", " "), height=0.68)
        left += vals
    for yi, m in enumerate(models):
        tot = left[yi]
        k = int(t.loc[m].sum()) if m in t.index else 0
        ax.text(tot + 0.25, yi, f"{tot:.1f}% (n = {k})", va="center", fontsize=5.2, color=PALETTE["slate"])
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    _group_separator(ax, models, step=1.0)
    ax.set_xlim(0, max(12, left.max() * 1.45))
    ax.set_xlabel("Misattributed or acted-on notes\n(% of encounters)")
    ax.tick_params(axis="y", length=0)
    ax.legend(fontsize=4.9, loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False, ncol=1, handletextpad=0.4, borderaxespad=0.0, labelspacing=0.18)


def panel_sections(ax, sections: pd.DataFrame, models: list) -> None:
    """Where the incorporated content lands: share of incorporated notes with the content in each note section (both families)."""
    s = sections[sections["section"].isin(SECTIONS)]
    # weight the two families by their number of incorporated notes
    mat = np.zeros((len(models), len(SECTIONS)))
    for i, m in enumerate(models):
        for j, sec in enumerate(SECTIONS):
            g = s[(s["model"] == m) & (s["section"] == sec)]
            if len(g):
                mat[i, j] = 100 * np.average(g["share"], weights=g["n_incorporated"])
    im = ax.imshow(mat, cmap="Purples", vmin=0, vmax=max(60, mat.max()), aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if mat[i, j] > 0.55 * max(60, mat.max()) else "black")
    ax.set_xticks(range(len(SECTIONS)))
    ax.set_xticklabels([SECTION_LABELS[x] for x in SECTIONS], fontsize=5.2, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    n_open = sum(m in MODELS for m in models)
    if 0 < n_open < len(models):
        ax.axhline(n_open - 0.5, color="white", lw=1.5)
    ax.tick_params(length=0)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Incorporated notes with the\ncontent in this section (%)", fontsize=5.4)
    cb.ax.tick_params(labelsize=5.2)


def panel_severity(ax, sev: pd.DataFrame, models: list) -> None:
    """Judge severity (1 = mention, 3 = clinically consequential) among incorporated notes, per model and family."""
    x = np.arange(len(models))
    w = 0.38
    shades = {1: "#FDE0DD", 2: "#FA9FB5", 3: "#C51B8A"}
    for k, dist in enumerate(DISTS):
        bottom = np.zeros(len(models))
        for lvl in (1, 2, 3):
            vals = np.array([100 * sev[(sev["model_short"] == m) & (sev["distractor_type"] == dist) & (sev["severity_v3"] == lvl)]["share"].sum() for m in models])
            ax.bar(x + (k - 0.5) * w, vals, w, bottom=bottom, color=shades[lvl], edgecolor="black", lw=0.4, label=f"Severity {lvl}" if k == 0 else None)
            bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", rotation_mode="anchor", fontsize=5.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Incorporated notes (%)")
    ax.legend(fontsize=5.2, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handletextpad=0.4, columnspacing=0.8, borderaxespad=0.0)
    ax.set_xlabel("left bar nonliteral, right bar bystander", fontsize=5.4, color=PALETTE["grey"])


def _pick_examples(examples: list, keys: list) -> list:
    """keys: (model, family, outcome, failure_mode or None, topic substring or None)."""
    out = []
    for key in keys:
        model, dist, outcome, mode = key[:4]
        topic = key[4] if len(key) > 4 else None
        cand = [e for e in examples if e["model"] == model and e["distractor"] == dist and e["outcome"] == outcome and (mode is None or e.get("failure_mode") == mode)
                and (topic is None or topic.lower() in e["topic"].lower())]
        cand = [e for e in cand if e["note_lines"]]
        if cand:
            out.append(sorted(cand, key=lambda e: (-(e["severity"] or 0), -len(e["note_lines"][0])))[0])
    return out


def panel_examples(ax, examples: list) -> None:
    """Annotated real cases: the inserted aside, what the note wrote, and the judge's reading."""
    ax.axis("off")
    n = len(examples)
    if not n:
        return
    h = 1.0 / n
    for i, e in enumerate(examples):
        y0 = 1.0 - i * h
        aside = " / ".join(l.strip() for l in e["aside"].splitlines() if l.strip())
        aside = textwrap.shorten(aside, 300, placeholder=" …")
        note = " ‖ ".join(l.replace("**", "").replace("__", "") for l in e["note_lines"][:2])
        note = textwrap.shorten(note, 300, placeholder=" …")
        judge = textwrap.shorten(e["attribution_reasoning"] or "", 230, placeholder=" …")
        head = f"{e['model']} · {DLABEL[e['distractor']]} exchange · {MODE_LABELS.get(e.get('failure_mode', ''), 'Other').replace(chr(10), ' ')} · judge severity {e['severity']} of 3"
        rows = [("Inserted aside", aside, DISTRACTOR_COLORS[e["distractor"]], True),
                ("The note wrote", note, OUTCOME_COLORS["misattributed"] if e["outcome"] == "misattributed" else OUTCOME_COLORS["acted_on"], False),
                ("Judge", judge, PALETTE["grey"], False)]
        ax.text(0.0, y0 - 0.02 * h, head, fontsize=5.8, fontweight="bold", va="top", ha="left", transform=ax.transAxes, color=PALETTE["slate"])
        y = y0 - 0.15 * h
        for label, text, color, italic in rows:
            lines = textwrap.wrap(text, 128)
            ax.text(0.0, y, label, fontsize=5.2, fontweight="bold", va="top", ha="left", transform=ax.transAxes, color=color)
            ax.text(0.12, y, "\n".join(lines), fontsize=5.0, va="top", ha="left", transform=ax.transAxes, style="italic" if italic else "normal", linespacing=1.15)
            y -= (0.05 + 0.078 * len(lines)) * h
        if i < n - 1:
            ax.plot([0, 1], [y0 - h + 0.02 * h, y0 - h + 0.02 * h], color=PALETTE["grey"], lw=0.4, transform=ax.transAxes, clip_on=False)


DEFAULT_EXAMPLE_KEYS = [("GPT-5.4", "bystander", "misattributed", "substituted_history", "cyst"), ("Claude Opus 5", "nonliteral", "acted_on", "pathologised_conversation", "sourdough"),
                        ("Claude Fable 5.1", "bystander", "misattributed", "inferred_concern", "sleep apnoea")]


def fig_failure(results: Path, figures: Path, name: str = "fig3_failure", example_keys: list | None = None) -> None:
    tax = pd.read_csv(results / "amb_failure_taxonomy.csv")
    sections = pd.read_csv(results / "amb_failure_sections.csv")
    sev = pd.read_csv(results / "amb_failure_severity.csv")
    notes = pd.read_csv(results / "amb_failure_notes.csv")
    examples = json.loads((results / "amb_failure_examples.json").read_text())
    summ = pd.read_csv(results / "amb_summary.csv")
    models = models_in(summ[summ["n_pairs"] > 0])
    n_enc = notes.groupby("model_short").size()
    fig = plt.figure(figsize=(NATURE_DOUBLE, 3.3 + 0.25 * max(0, len(models) - 4)))  # verbatim-examples panel dropped (author, 2026-09-08)
    gs = fig.add_gridspec(1, 3, wspace=0.7, left=0.10, right=0.985, top=0.76, bottom=0.17)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    panel_taxonomy(axa, tax, n_enc, models)
    panel_sections(axb, sections, models)
    panel_severity(axc, sev, models)
    place_panel_labels(fig, [("a", axa, 0, 0), ("b", axb, 1, 0), ("c", axc, 2, 0)])
    save(fig, figures / name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results/v3"))
    ap.add_argument("--baseline", type=Path, default=Path("results"), help="first-generation results (comparison panel, audio panel)")
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    args = ap.parse_args()
    style.set_style()
    fig_documentation(args.results, args.baseline, args.figures)
    fig_failure(args.results, args.figures)
    print("figures written to", args.figures)


if __name__ == "__main__":
    main()
