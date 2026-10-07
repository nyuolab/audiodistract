"""Tables of the acoustic-overlay sub-study from results/raw/audio_*.parquet (CPU only).

Primary endpoint: mixed-minus-clean rate of target-specific contamination (paired judge) at -10 dB pooled over the
four models, paired within encounter. CIs are 95% percentile bootstraps over encounters (2,000 resamples, seed 0):
the unit is consultation A, whose donors x models are resampled together. Secondary: transcription leakage by level,
contamination conditional on leakage, note-quality deltas. Model x level strata are reported alongside.

    python -m llm_distract.audio.analyze [--collect outputs/audio] [--raw results/raw] [--out results]

``--collect`` first gathers outputs/audio/<model>/judge_{contamination_paired,quality}.parquet into
results/raw/audio_judgments.parquet.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from llm_distract.ambient.common import MODEL_ORDER, QUALITY_METRICS, short_model, write_parquet
from llm_distract.common.stats import bootstrap_ci

KEYS = ["source_dataset", "item_id", "model", "distractor_type"]
QUALITY = list(QUALITY_METRICS)
DT_RE = re.compile(r"^audio_(-?\d+)dB_d(\d+)$")
PRIMARY_LEVEL = -10
N_BOOT = 2000


def parse_distractor(distractor_type: str) -> tuple:
    """'audio_-10dB_d1' -> (-10, 1)."""
    m = DT_RE.match(str(distractor_type))
    if not m:
        raise ValueError(f"not an audio distractor type: {distractor_type!r}")
    return int(m.group(1)), int(m.group(2))


def collect(runs_root: str | Path) -> pd.DataFrame:
    """Judge outputs of every model run (source_dataset primock57) as one long per-note table."""
    frames = []
    for c_path in sorted(Path(runs_root).glob("*/judge_contamination_paired.parquet")):
        c = pd.read_parquet(c_path)[KEYS + ["condition", "contamination", "severity", "judge_reasoning", "parse_error",
                                             "judge_model", "judge_protocol"]]
        q_path = c_path.with_name("judge_quality.parquet")
        if q_path.exists():
            q = pd.read_parquet(q_path).reindex(columns=KEYS + ["condition"] + QUALITY)
            c = c.merge(q, on=KEYS + ["condition"], how="left")
        frames.append(c)
    if not frames:
        raise RuntimeError(f"no judge_contamination_paired.parquet under {runs_root}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["source_dataset"] == "primock57"].copy()
    df["item_id"] = df["item_id"].astype(str)
    df.insert(2, "family", "PriMock57")
    return df


def load_pairs(raw: Path) -> pd.DataFrame:
    """One row per (encounter A, model, donor, level): clean/mixed judge scores, deltas, level, donor, leakage."""
    j = pd.read_parquet(raw / "audio_judgments.parquet")
    values = ["contamination", "severity"] + QUALITY
    for v in values:
        j[v] = pd.to_numeric(j[v], errors="coerce") if v in j else np.nan
    j["model"] = j["model"].map(short_model)
    wide = j.set_index(KEYS + ["condition"])[values].unstack("condition")
    wide.columns = [f"{v}_{c}" for v, c in wide.columns]
    wide = wide.reset_index().dropna(subset=["contamination_clean", "contamination_distracted"])
    if wide.empty:
        raise RuntimeError("no encounter with both notes judged in audio_judgments.parquet")
    parsed = [parse_distractor(d) for d in wide["distractor_type"]]
    wide["level_db"], wide["donor"] = [p[0] for p in parsed], [p[1] for p in parsed]
    for v in ["contamination"] + QUALITY:
        wide[f"d_{v}"] = wide[f"{v}_distracted"] - wide[f"{v}_clean"]
    leak = pd.read_parquet(raw / "audio_leakage.parquet")[["item_id", "distractor_type", "leakage", "n_leaked", "wer_vs_clean"]]
    wide = wide.merge(leak, on=["item_id", "distractor_type"], how="left")
    wide["leaked"] = wide["n_leaked"].fillna(0) > 0
    return wide


def encounter_ci(df: pd.DataFrame, col: str, scale: float = 100.0) -> tuple:
    """Mean of ``col`` and its percentile-bootstrap CI over encounters (per-encounter means are the resampled units)."""
    per = df.groupby("item_id")[col].mean().to_numpy(float) * scale
    return bootstrap_ci(per, n_boot=N_BOOT, seed=0)


def leakage_by_level(leak: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, g in leak.groupby("level_db"):
        m, lo, hi = encounter_ci(g, "leakage")
        rows.append({"level_db": int(level), "n_mixes": len(g), "n_encounters": g["item_id"].nunique(), "leakage_pct": m,
                     "ci_low": lo, "ci_high": hi, "any_leak_pct": 100 * (g["n_leaked"] > 0).mean(),
                     "overlap_clean_pct": 100 * g["overlap_clean"].mean(), "wer_vs_clean": g["wer_vs_clean"].mean(),
                     "wer_clean_vs_human": g["wer_clean_vs_human"].mean(), "segment_seconds": g["segment_seconds"].mean(),
                     "segment_keywords": g["segment_score"].mean(), "content_words": g["n_content"].mean()})
    return pd.DataFrame(rows)


def _rate_block(g: pd.DataFrame) -> dict:
    d, dlo, dhi = encounter_ci(g, "d_contamination")
    r, rlo, rhi = encounter_ci(g, "contamination_distracted")
    return {"n_pairs": len(g), "n_encounters": g["item_id"].nunique(), "rate_clean": 100 * g["contamination_clean"].mean(),
            "rate_mixed": r, "rate_mixed_ci_low": rlo, "rate_mixed_ci_high": rhi, "delta_pp": d, "ci_low": dlo, "ci_high": dhi,
            "severity_mixed": g["severity_distracted"].mean()}


def contamination_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Rows per level x model plus pooled rows (model 'all'); level 'all' pools the levels."""
    present = set(pairs["model"])
    models = [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))
    rows = []
    for level in sorted(pairs["level_db"].unique()) + ["all"]:
        sub = pairs if level == "all" else pairs[pairs["level_db"] == level]
        for model in ["all"] + models:
            g = sub if model == "all" else sub[sub["model"] == model]
            rows.append({"level_db": level, "model": model, **_rate_block(g)})
    return pd.DataFrame(rows)


def conditional_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Contamination by level among mixes whose segment words leaked into the transcript vs those without leakage."""
    rows = []
    for level, sub in pairs.groupby("level_db"):
        rho, p = stats.spearmanr(sub["leakage"], sub["contamination_distracted"]) if sub["leakage"].nunique() > 1 else (np.nan, np.nan)
        for leaked, g in sub.groupby("leaked"):
            rows.append({"level_db": int(level), "leaked": bool(leaked), **_rate_block(g),
                         "mean_leakage_pct": 100 * g["leakage"].mean(), "spearman_rho": rho, "spearman_p": p})
    return pd.DataFrame(rows)


def quality_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Mixed-minus-clean quality deltas by level (empty, with columns, until the quality judge has run)."""
    rows = []
    for level in ["all"] + sorted(pairs["level_db"].unique()):
        g = pairs if level == "all" else pairs[pairs["level_db"] == level]
        for q in QUALITY:
            gq = g.dropna(subset=[f"d_{q}"])
            if gq.empty:
                continue
            m, lo, hi = encounter_ci(gq, f"d_{q}", scale=1.0)
            rows.append({"level_db": level, "metric": q, "scale": "0/1" if q == "hallucination" else "1-5", "n_pairs": len(gq),
                         "clean_mean": gq[f"{q}_clean"].mean(), "mixed_mean": gq[f"{q}_distracted"].mean(),
                         "delta": m, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows, columns=["level_db", "metric", "scale", "n_pairs", "clean_mean", "mixed_mean", "delta", "ci_low", "ci_high"])


def run(raw: Path, out: Path) -> dict:
    pairs = load_pairs(raw)
    tables = {"leakage_by_level": leakage_by_level(pd.read_parquet(raw / "audio_leakage.parquet")),
              "contamination_by_level_model": contamination_table(pairs),
              "contamination_given_leakage": conditional_table(pairs),
              "quality": quality_table(pairs)}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"audio_{name}.csv", index=False, float_format="%.6g")
    return tables


def summarize(t: dict) -> str:
    c = t["contamination_by_level_model"]
    pooled = c[(c["model"] == "all") & (c["level_db"] != "all")]
    lines = []
    for _, p in pooled[pooled["level_db"] == PRIMARY_LEVEL].iterrows():
        lines.append(f"primary ({PRIMARY_LEVEL} dB, pooled over models, {int(p['n_encounters'])} encounters, {int(p['n_pairs'])} pairs): "
                     f"clean {p['rate_clean']:.1f}% -> mixed {p['rate_mixed']:.1f}%; delta {p['delta_pp']:.1f} pp "
                     f"(CI {p['ci_low']:.1f} to {p['ci_high']:.1f})")
        per = c[(c["level_db"] == PRIMARY_LEVEL) & (c["model"] != "all")]
        lines.append("  per model: " + "; ".join(f"{r['model']} {r['delta_pp']:.1f} ({r['ci_low']:.1f} to {r['ci_high']:.1f})" for _, r in per.iterrows()))
    lines.append("pooled delta by level: " + "; ".join(f"{r['level_db']} dB {r['delta_pp']:.1f} pp" for _, r in pooled.iterrows()))
    lines.append("leakage by level: " + "; ".join(f"{int(r['level_db'])} dB {r['leakage_pct']:.1f}% of segment words "
                                                  f"(any leak in {r['any_leak_pct']:.0f}% of mixes)" for _, r in t["leakage_by_level"].iterrows()))
    lines.append("mixed-note contamination | leakage: " + "; ".join(f"{int(r['level_db'])} dB {'leaked' if r['leaked'] else 'none'} "
                                                                     f"{r['rate_mixed']:.1f}% (n={int(r['n_pairs'])})" for _, r in t["contamination_given_leakage"].iterrows()))
    qa = t["quality"][t["quality"]["level_db"] == "all"]
    if len(qa):
        lines.append("quality deltas (pooled): " + "; ".join(f"{r['metric']} {r['delta']:+.3f} ({r['ci_low']:+.3f} to {r['ci_high']:+.3f})" for _, r in qa.iterrows()))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collect", metavar="RUNS_ROOT", default=None, help="gather outputs/audio/<model>/judge_*.parquet first")
    ap.add_argument("--raw", type=Path, default=Path("results/raw"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    if args.collect:
        df = collect(args.collect)
        print(f"collected {len(df)} judged notes ->", write_parquet(df, args.raw / "audio_judgments.parquet", meta={"runs_root": args.collect}))
    print(summarize(run(args.raw, args.out)))


if __name__ == "__main__":
    main()
