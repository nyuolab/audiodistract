"""Extended Data figure of the acoustic-overlay sub-study (dose-response) from results/audio_*.csv.

Panels: (a) transcription leakage vs background level, (b) contamination rate of the mixed notes vs level per model
with the pooled estimate and its CI, (c) contamination conditional on leakage.

    python -m llm_distract.audio.plot [--results results] [--figures figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from llm_distract.common import style
from llm_distract.common.style import MODEL_COLORS, NATURE_DOUBLE, PALETTE, place_panel_labels, save

XLABEL = "Background level (dB re foreground)"


def _err(y: pd.Series, lo: pd.Series, hi: pd.Series) -> list:
    return [y - lo, hi - y]


def panel_leakage(ax, leak: pd.DataFrame) -> None:
    leak = leak.sort_values("level_db")
    x = leak["level_db"]
    ax.errorbar(x, leak["leakage_pct"], yerr=_err(leak["leakage_pct"], leak["ci_low"], leak["ci_high"]), fmt="o-",
                color=PALETTE["distractor"], ms=4, lw=1, capsize=2, mec="black", mew=0.4, label="Segment words in mixed transcript")
    ax.plot(x, leak["any_leak_pct"], "s--", color=PALETTE["slate"], ms=3, lw=0.8, label="Mixes with any leaked word")
    ax.plot(x, leak["overlap_clean_pct"], ":", color=PALETTE["grey"], lw=0.8, label="Already in clean transcript")
    ax.set_xticks(list(x))
    ax.set_ylim(0, 100)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Transcription leakage (%)")
    ax.legend(loc="upper left", fontsize=5.4)


def panel_contamination(ax, cont: pd.DataFrame) -> None:
    c = cont.copy()
    c["level_db"] = pd.to_numeric(c["level_db"], errors="coerce")
    c = c.dropna(subset=["level_db"]).sort_values("level_db")
    for model, g in c[c["model"] != "all"].groupby("model", sort=False):
        ax.plot(g["level_db"], g["rate_mixed"], "o-", color=MODEL_COLORS.get(model, PALETTE["grey"]), ms=3, lw=0.8, label=model)
    p = c[c["model"] == "all"]
    ax.fill_between(p["level_db"], p["rate_mixed_ci_low"], p["rate_mixed_ci_high"], color=PALETTE["slate"], alpha=0.15, lw=0)
    ax.plot(p["level_db"], p["rate_mixed"], "-", color=PALETTE["slate"], lw=1.4, label="Pooled (95% CI)")
    ax.axhline(p["rate_clean"].mean(), ls="--", color=PALETTE["clean"], lw=0.8, label="Clean notes")
    ax.set_xticks(list(p["level_db"]))
    ax.set_ylim(0, 30)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Notes incorporating the other\npatient's content (%)")
    ax.legend(loc="upper left", fontsize=5.4)


def panel_conditional(ax, cond: pd.DataFrame) -> None:
    cond = cond.sort_values("level_db")
    for leaked, color, label in ((True, PALETTE["distractor"], "Leaked into transcript"),
                                 (False, PALETTE["grey"], "No leaked word")):
        g = cond[cond["leaked"] == leaked]
        if len(g):
            ax.errorbar(g["level_db"], g["rate_mixed"], yerr=_err(g["rate_mixed"], g["rate_mixed_ci_low"], g["rate_mixed_ci_high"]),
                        fmt="o-", color=color, ms=4, lw=1, capsize=2, mec="black", mew=0.4, label=label)
    ax.set_xticks(sorted(cond["level_db"].unique()))
    ax.set_ylim(0, 30)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Contaminated mixed notes (%)")
    ax.legend(loc="upper left", fontsize=5.4)


def ed_audio(results: Path, figures: Path) -> list:
    leak = pd.read_csv(results / "audio_leakage_by_level.csv")
    cont = pd.read_csv(results / "audio_contamination_by_level_model.csv")
    cond = pd.read_csv(results / "audio_contamination_given_leakage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE * 0.7, 2.4), gridspec_kw=dict(wspace=0.5))
    panel_leakage(axes[0], leak)
    panel_contamination(axes[1], cont)
    place_panel_labels(fig, [("a", axes[0], 0, 0), ("b", axes[1], 1, 0)])  # the leakage-conditional panel is Fig. 2f
    return save(fig, figures / "ed_audio")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    args = ap.parse_args()
    style.set_style()
    print("wrote", ", ".join(str(p) for p in ed_audio(args.results, args.figures)))


if __name__ == "__main__":
    main()
