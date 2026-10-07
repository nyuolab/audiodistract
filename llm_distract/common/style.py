"""Figure style shared by every figure in the paper (Nature-style, matplotlib).

Colours follow the MedDistractQA preprint: orange = open-weight general, blue = open-weight medical,
purple = proprietary; red = distractor / contaminated; green = clean.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
NATURE_SINGLE = 89 * MM      # inches
NATURE_DOUBLE = 183 * MM
NATURE_MAX_HEIGHT = 247 * MM

PALETTE = {
    "general": "#F28E2B",
    "medical": "#4E9BD3",
    "proprietary": "#B07AA1",
    "distractor": "#D64545",
    "clean": "#3C9D5D",
    "nonliteral": "#E15759",
    "bystander": "#4E79A7",
    "slate": "#2F3E4E",
    "grey": "#8A9BA8",
    "light": "#F5F7FA",
    "interference": "#D64545",
    "random": "#BFC7CF",
    "all": "#8A9BA8",
}
GROUP_COLORS = {"Open-weight, general": PALETTE["general"], "Open-weight, medical": PALETTE["medical"],
                "Proprietary": PALETTE["proprietary"]}
GROUP_MARKERS = {"Open-weight, general": "o", "Open-weight, medical": "s", "Proprietary": "^"}
DISTRACTOR_COLORS = {"nonliteral": PALETTE["nonliteral"], "bystander": PALETTE["bystander"], "clean": PALETTE["clean"]}
MODEL_COLORS = {  # four mechanism models, then the API models of the documentation arm
    "Llama-3.1-8B": "#4E79A7", "Qwen2.5-7B": "#F28E2B", "Mistral-7B": "#59A14F", "Gemma-2-9B": "#B07AA1",
    "GPT-5.4": "#9C755F", "GPT-5.6 Sol": "#E15759", "Claude Opus 5": "#76B7B2", "Claude Fable 5.1": "#BAB0AC",
}
MODEL_SHORT = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "google/gemma-2-9b-it": "Gemma-2-9B",
    # API models of the documentation arm (added September 2026)
    "gpt-5.4-2026-03-05": "GPT-5.4",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "claude-opus-5": "Claude Opus 5",
    "claude-fable-5-1": "Claude Fable 5.1",
}
FRONTIER_MODELS = ["GPT-5.4", "GPT-5.6 Sol", "Claude Opus 5", "Claude Fable 5.1"]
MECHANISM_MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-9b-it"]
DISTRACTOR_LABEL = {"nonliteral": "Nonliteral", "bystander": "Bystander", "clean": "Clean"}


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 120,
    })


def panel_label(ax, letter: str, dx: float = -0.12, dy: float = 1.04) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def place_panel_labels(fig, panels, gap_pt: float = 2.0, fontsize: float = 9) -> None:
    """Panel letters at the top-left of each panel's drawn extent (axes, tick labels, axis labels, titles, legends).

    ``panels`` is a list of ``(letter, axes_or_list_of_axes, column, row)``. Letters in the same column share the
    leftmost extent of that column and letters in the same row share the topmost extent of that row, so they line
    up across the grid regardless of tick-label width or legend placement.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    boxes = []
    for letter, axs, col, row in panels:
        axs = list(axs) if isinstance(axs, (list, tuple)) else [axs]
        bb = mpl.transforms.Bbox.union([a.get_tightbbox(renderer) for a in axs])
        (x0, _), (_, y1) = to_fig.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        boxes.append((letter, x0, y1, col, row))
    col_x = {}
    row_y = {}
    for _, x0, y1, col, row in boxes:
        col_x[col] = min(col_x.get(col, 1.0), x0)
        row_y[row] = max(row_y.get(row, 0.0), y1)
    gap = gap_pt / 72 / fig.get_size_inches()[1]
    for letter, _, _, col, row in boxes:
        fig.text(col_x[col], row_y[row] + gap, letter, fontsize=fontsize, fontweight="bold", va="bottom", ha="left")


def save(fig, path: str | Path, formats=("pdf", "png")) -> list[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        p = path.with_suffix(f".{fmt}")
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
        out.append(p)
    plt.close(fig)
    return out
