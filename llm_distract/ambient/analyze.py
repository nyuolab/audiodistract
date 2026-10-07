"""Regenerate every ambient-documentation table from results/raw/amb_*.parquet (CPU only).

Primary endpoint: distracted-minus-clean rate of target-specific contamination (paired judge, Claude Sonnet 5),
paired within encounter. The pooled (encounter-weighted) estimate is the primary estimand; the unweighted mean of
the 40 model x split x distractor strata is reported alongside. Differences are in percentage points (pp) with
2,000-resample paired percentile-bootstrap CIs (seed 0).

    python -m llm_distract.ambient.analyze [--raw results/raw] [--data data/ambient] [--out results]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from llm_distract.common.stats import bootstrap_ci
from llm_distract.common.style import FRONTIER_MODELS, MODEL_SHORT

KEYS = ["source_dataset", "item_id", "model", "distractor_type"]
QUALITY = ["clinical_correctness", "completeness", "succinctness", "hallucination", "overall_quality"]
MODEL_ORDER = ["Gemma-2-9B", "Mistral-7B", "Llama-3.1-8B", "Qwen2.5-7B"]  # open-weight models of the paper


def model_order(pairs: pd.DataFrame) -> list:
    """Open-weight models first, then whichever API models (``FRONTIER_MODELS``) have judged notes."""
    present = set(pairs["model"].unique())
    return [m for m in MODEL_ORDER + FRONTIER_MODELS if m in present]


def cohorts(pairs: pd.DataFrame) -> list:
    """(cohort name, models) for the open-weight cohort of the paper and, when present, the API models added later.
    The open-weight cohort is the prespecified primary analysis; every table carries a ``cohort`` column."""
    out = [("open-weight", [m for m in MODEL_ORDER if m in set(pairs["model"])])]
    api = [m for m in FRONTIER_MODELS if m in set(pairs["model"])]
    if api:
        out.append(("API", api))
    return out


def load_pairs(raw: Path) -> pd.DataFrame:
    """One row per clean-distracted note pair with both judges' scores and transcript length."""
    j = pd.read_parquet(raw / "amb_judgments.parquet")
    n = pd.read_parquet(raw / "amb_notes.parquet")[KEYS + ["condition", "note_chars", "transcript_chars"]]
    j = j.merge(n, on=KEYS + ["condition"], how="left")
    j["model"] = j["model"].map(lambda m: MODEL_SHORT.get(m, m))
    value_cols = ["contamination_v3", "severity_v3", "contamination_v2_gpt", "severity_v2", "note_chars", "transcript_chars"] + QUALITY
    wide = j.pivot_table(index=KEYS + ["family"], columns="condition", values=value_cols, aggfunc="first")
    wide.columns = [f"{v}_{c}" for v, c in wide.columns]
    wide = wide.reset_index()
    for v in value_cols:  # judges that were not run on this cohort (e.g. the single-note judge) yield all-NA columns
        for c in ("clean", "distracted"):
            if f"{v}_{c}" not in wide.columns:
                wide[f"{v}_{c}"] = np.nan
    for v in ["contamination_v3", "contamination_v2_gpt"] + QUALITY:
        wide[f"d_{v}"] = wide[f"{v}_distracted"] - wide[f"{v}_clean"]
    return wide


def _ci(d: np.ndarray, scale: float = 100.0) -> tuple[float, float, float]:
    m, lo, hi = bootstrap_ci(np.asarray(d, float) * scale)
    return m, lo, hi


def strata_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split, dist), g in pairs.groupby(["model", "source_dataset", "distractor_type"]):
        m, lo, hi = _ci(g["d_contamination_v3"].values)
        rows.append({"model": model, "split": split, "family": g["family"].iloc[0], "distractor": dist, "n_pairs": len(g),
                     "rate_clean": 100 * g["contamination_v3_clean"].mean(), "rate_distracted": 100 * g["contamination_v3_distracted"].mean(),
                     "delta_pp": m, "ci_low": lo, "ci_high": hi, "severity_distracted": g["severity_v3_distracted"].mean()})
    return pd.DataFrame(rows)


def summary_table(pairs: pd.DataFrame, strata: pd.DataFrame, col: str = "d_contamination_v3") -> pd.DataFrame:
    def block(name, level, mask_pairs, mask_strata):
        g = pairs[mask_pairs]
        m, lo, hi = _ci(g[col].values)
        s = strata[mask_strata]
        return {"level": level, "name": name, "n_pairs": len(g), "n_strata": len(s),
                "rate_clean": 100 * g[col.replace("d_", "") + "_clean"].mean(),
                "rate_distracted": 100 * g[col.replace("d_", "") + "_distracted"].mean(),
                "pooled_delta_pp": m, "pooled_ci_low": lo, "pooled_ci_high": hi,
                "unweighted_mean_pp": s["delta_pp"].mean(), "n_strata_positive": int((s["delta_pp"] > 0).sum()),
                "n_strata_zero": int((s["delta_pp"] == 0).sum()), "n_strata_negative": int((s["delta_pp"] < 0).sum())}
    rows = []
    for cohort, models in cohorts(pairs):
        cp, cs = pairs["model"].isin(models), strata["model"].isin(models)
        rows.append({**block("all", "overall", cp, cs), "cohort": cohort})
        for mdl in models:
            rows.append({**block(mdl, "model", pairs["model"] == mdl, strata["model"] == mdl), "cohort": cohort})
        for dist in ("nonliteral", "bystander"):
            rows.append({**block(dist, "distractor", cp & (pairs["distractor_type"] == dist), cs & (strata["distractor"] == dist)), "cohort": cohort})
        for fam in ("ACI-Bench", "MTS-Dialog"):
            rows.append({**block(fam, "family", cp & (pairs["family"] == fam), cs & (strata["family"] == fam)), "cohort": cohort})
        for fam in ("ACI-Bench", "MTS-Dialog"):
            for dist in ("nonliteral", "bystander"):
                rows.append({**block(f"{fam} | {dist}", "family x distractor", cp & (pairs["family"] == fam) & (pairs["distractor_type"] == dist),
                                     cs & (strata["family"] == fam) & (strata["distractor"] == dist)), "cohort": cohort})
        for mdl in models:
            for dist in ("nonliteral", "bystander"):
                rows.append({**block(f"{mdl} | {dist}", "model x distractor", (pairs["model"] == mdl) & (pairs["distractor_type"] == dist),
                                     (strata["model"] == mdl) & (strata["distractor"] == dist)), "cohort": cohort})
        for mdl in models:
            for fam in ("ACI-Bench", "MTS-Dialog"):
                for dist in ("nonliteral", "bystander"):
                    rows.append({**block(f"{mdl} | {fam} | {dist}", "model x family x distractor",
                                         (pairs["model"] == mdl) & (pairs["family"] == fam) & (pairs["distractor_type"] == dist),
                                         (strata["model"] == mdl) & (strata["family"] == fam) & (strata["distractor"] == dist)), "cohort": cohort})
    df = pd.DataFrame(rows)
    return df[["cohort"] + [c for c in df.columns if c != "cohort"]]


def quality_table(all_pairs: pd.DataFrame, strata_keys=("model", "source_dataset", "distractor_type")) -> pd.DataFrame:
    rows = []
    for cohort, models in cohorts(all_pairs):
        pairs = all_pairs[all_pairs["model"].isin(models)]
        for q in QUALITY:
            per_stratum = pairs.groupby(list(strata_keys))[f"d_{q}"].mean()
            m, lo, hi = _ci(pairs[f"d_{q}"].dropna().values, scale=1.0)
            rows.append({"cohort": cohort, "metric": q, "scale": "0/1" if q == "hallucination" else "1-5",
                         "clean_mean": pairs[f"{q}_clean"].mean(), "distracted_mean": pairs[f"{q}_distracted"].mean(),
                         "pooled_delta": m, "pooled_ci_low": lo, "pooled_ci_high": hi, "unweighted_strata_delta": per_stratum.mean()})
    return pd.DataFrame(rows)


def judge_sensitivity(pairs: pd.DataFrame, strata: pd.DataFrame) -> pd.DataFrame:
    """Paired vs single-note judge on the notes that both judges scored (the open-weight models; the API models added in
    September 2026 were judged with the paired protocol only)."""
    pairs = pairs[pairs["contamination_v2_gpt_distracted"].notna() & pairs["contamination_v2_gpt_clean"].notna()]
    rows = []
    for label, col in (("v3 paired (Claude Sonnet 5)", "contamination_v3"), ("v2 single-note (GPT-5.4)", "contamination_v2_gpt")):
        per_stratum = pairs.groupby(["model", "source_dataset", "distractor_type"])[f"d_{col}"].mean() * 100
        m, lo, hi = _ci(pairs[f"d_{col}"].values)
        rows.append({"judge": label, "pooled_delta_pp": m, "pooled_ci_low": lo, "pooled_ci_high": hi,
                     "unweighted_mean_pp": per_stratum.mean(), "clean_false_positive_rate_pct": 100 * pairs[f"{col}_clean"].mean(),
                     "distracted_positive_rate_pct": 100 * pairs[f"{col}_distracted"].mean(),
                     "n_strata_negative": int((per_stratum < 0).sum())})
    a = pairs["contamination_v3_distracted"].astype(int).values
    b = pairs["contamination_v2_gpt_distracted"].astype(int).values
    po = (a == b).mean()
    pe = (a.mean() * b.mean()) + ((1 - a.mean()) * (1 - b.mean()))
    kappa = (po - pe) / (1 - pe)
    rows.append({"judge": "agreement on distracted notes (v3 vs v2)", "percent_agreement": 100 * po, "cohen_kappa": kappa,
                 "n_notes": len(a)})
    return pd.DataFrame(rows)


def attribution_table(pairs: pd.DataFrame, raw: Path) -> pd.DataFrame:
    """Per cohort/model/family/distractor: incorporation (paired judge), misattribution to the patient and clinical use
    (attribution sub-judge on the incorporated distracted notes), as encounter-level rates with bootstrap CIs.
    Empty frame if results/raw/amb_attribution.parquet is absent."""
    path = raw / "amb_attribution.parquet"
    if not path.exists():
        return pd.DataFrame()
    a = pd.read_parquet(path); a["item_id"] = a["item_id"].astype(str)
    a["model"] = a["model"].map(lambda m: MODEL_SHORT.get(m, m))
    keys = ["source_dataset", "item_id", "model", "distractor_type"]
    p = pairs.copy(); p["item_id"] = p["item_id"].astype(str)
    p = p.merge(a[keys + ["attribution", "used_for_patient"]], on=keys, how="left")
    inc = p["contamination_v3_distracted"] == 1
    # three mutually exclusive outcomes among incorporated notes
    p["misattributed"] = (inc & (p["attribution"] == "patient")).astype(int)
    p["used"] = (inc & (p["attribution"] == "correct") & (p["used_for_patient"] == 1)).astype(int)  # correctly attributed, acted on
    p["recorded_only"] = (inc & (p["attribution"] == "correct") & (p["used_for_patient"] != 1)).astype(int)
    judged = inc & p["attribution"].notna()
    rows = []
    def block(cohort, level, name, mask):
        g = p[mask]
        if not len(g):
            return
        r = {"cohort": cohort, "level": level, "name": name, "n_pairs": len(g), "n_incorporated": int(g["contamination_v3_distracted"].sum()),
             "n_incorporated_judged": int((judged[mask]).sum())}
        for col in ("misattributed", "used", "recorded_only"):
            m, lo, hi = _ci(g[col].values)
            r[f"{col}_pct"], r[f"{col}_ci_low"], r[f"{col}_ci_high"] = m, lo, hi
        r["incorporation_pct"] = 100 * g["contamination_v3_distracted"].mean()
        rows.append(r)
    for cohort, models in cohorts(p):
        cp = p["model"].isin(models)
        block(cohort, "overall", "all", cp)
        for dist in ("nonliteral", "bystander"):
            block(cohort, "distractor", dist, cp & (p["distractor_type"] == dist))
        for fam in ("ACI-Bench", "MTS-Dialog"):
            block(cohort, "family", fam, cp & (p["family"] == fam))
        for mdl in models:
            block(cohort, "model", mdl, p["model"] == mdl)
            for dist in ("nonliteral", "bystander"):
                block(cohort, "model x distractor", f"{mdl} | {dist}", (p["model"] == mdl) & (p["distractor_type"] == dist))
    return pd.DataFrame(rows)


def length_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, models in cohorts(pairs):
        cohort_pairs = pairs[pairs["model"].isin(models)]
        for scope, sub in (("all", cohort_pairs), ("MTS-Dialog", cohort_pairs[cohort_pairs["family"] == "MTS-Dialog"]),
                           ("ACI-Bench", cohort_pairs[cohort_pairs["family"] == "ACI-Bench"])):
            sub = sub.copy()
            if len(sub) < 25:  # cohort absent from this fold
                continue
            sub["quintile"] = pd.qcut(sub["transcript_chars_clean"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
            rho, p = stats.spearmanr(sub["transcript_chars_clean"], sub["d_contamination_v3"])
            for q, g in sub.groupby("quintile", observed=True):
                m, lo, hi = _ci(g["d_contamination_v3"].values)
                rows.append({"cohort": cohort, "scope": scope, "quintile": int(q), "n_pairs": len(g), "median_transcript_chars": g["transcript_chars_clean"].median(),
                             "pooled_delta_pp": m, "ci_low": lo, "ci_high": hi, "spearman_rho": rho, "spearman_p": p})
    return pd.DataFrame(rows)


def topics_table(data: Path, insertions: str = "insertions.parquet") -> pd.DataFrame:
    ins = pd.read_parquet(data / insertions)
    rows = []
    for dist, g in ins.groupby("distractor_type"):
        n = len(g)
        topics = g["distractor_topic"].str.lower()
        for kw in (["pneumonia", "chest pain", "diabetes", "seizure"] if dist == "bystander" else ["contagion", "lupus", "strokes", "cancer", "zodiac", "horoscope"]):
            rows.append({"distractor": dist, "keyword": kw, "n": int(topics.str.contains(kw).sum()), "pct": 100 * topics.str.contains(kw).mean()})
        base_topic = topics.str.replace(r" \(.*\)$", "", regex=True)  # v3 topics carry "(relation; system)" suffixes
        for t, c in base_topic.value_counts().head(5).items():
            rows.append({"distractor": dist, "keyword": f"top: {t}", "n": int(c), "pct": 100 * c / n})
        rows.append({"distractor": dist, "keyword": "(unique topic strings)", "n": int(topics.nunique()), "pct": np.nan})
        rows.append({"distractor": dist, "keyword": "(mean inserted characters)", "n": int(round(g["distractor_conversation"].str.len().mean())), "pct": np.nan})
        rows.append({"distractor": dist, "keyword": "(mean inserted turns)", "n": round(g["distractor_conversation"].str.count("\n").mean() + 1, 2), "pct": np.nan})
    return pd.DataFrame(rows)


def examples_md(raw: Path, data: Path, pairs: pd.DataFrame, k: int = 3, insertions: str = "insertions.parquet") -> str:
    """Severity-3 examples: inserted exchange, contaminated note excerpt, judge rationale."""
    j = pd.read_parquet(raw / "amb_judgments.parquet")
    n = pd.read_parquet(raw / "amb_notes.parquet")
    ins = pd.read_parquet(data / insertions)
    j["model"] = j["model"].map(lambda m: MODEL_SHORT.get(m, m))
    n["model"] = n["model"].map(lambda m: MODEL_SHORT.get(m, m))
    sel = pairs[(pairs["severity_v3_distracted"] == 3) & (pairs["contamination_v3_clean"] == 0)]
    out = ["# Examples of target-specific contamination (severity 3)\n"]
    for mdl in model_order(pairs):
        for dist in ("bystander", "nonliteral"):
            cand = sel[(sel["model"] == mdl) & (sel["distractor_type"] == dist)].sort_values("transcript_chars_clean", ascending=False).head(1)
            for _, r in cand.iterrows():
                key = (r["source_dataset"], r["item_id"], mdl, dist)
                note = n[(n["source_dataset"] == key[0]) & (n["item_id"] == key[1]) & (n["model"] == mdl) & (n["distractor_type"] == dist)]
                jd = j[(j["source_dataset"] == key[0]) & (j["item_id"] == key[1]) & (j["model"] == mdl) & (j["distractor_type"] == dist)]
                ex = ins[(ins["source_dataset"] == key[0]) & (ins["item_id"].astype(str) == str(key[1])) & (ins["distractor_type"] == dist)]
                if ex.empty or note.empty:
                    continue
                out.append(f"## {mdl} | {key[0]} | item {key[1]} | {dist}\n")
                out.append("**Inserted exchange**\n\n```\n" + ex["distractor_conversation"].iloc[0] + "\n```\n")
                out.append(f"**Judge target (summary)**: {ex['distractor_summary'].iloc[0]}\n")
                dn = note[note["condition"] == "distracted"]["note"].iloc[0]
                out.append("**Distracted note (excerpt)**\n\n```\n" + dn[:1200] + "\n```\n")
                out.append("**Judge rationale**: " + str(jd[jd["condition"] == "distracted"]["judge_reasoning_v3"].iloc[0]) + "\n")
    return "\n".join(out)


def run(raw: Path, data: Path, out: Path, insertions: str = "insertions.parquet") -> dict:
    pairs = load_pairs(raw)
    n_ow = int(pairs["model"].isin(MODEL_ORDER).sum())
    if n_ow not in (0, 4608, 4616):
        print(f"warning: open-weight cohort has {n_ow} pairs (expected 4,608 for v3, 4,616 for legacy, or none)")
    strata = strata_table(pairs)
    tables = {
        "strata": strata,
        "summary": summary_table(pairs, strata),
        "quality": quality_table(pairs),
        "judge_sensitivity": judge_sensitivity(pairs, strata),
        "length": length_table(pairs),
        "attribution": attribution_table(pairs, raw),
        "topics": topics_table(data, insertions),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"amb_{name}.csv", index=False, float_format="%.6g")
    (out / "amb_examples.md").write_text(examples_md(raw, data, pairs, insertions=insertions))
    return tables


def summarize(t: dict) -> str:
    s = t["summary"][t["summary"]["cohort"] == "open-weight"].set_index("name")
    lines = [f"pairs: {int(s.loc['all', 'n_pairs'])}; strata: {int(s.loc['all', 'n_strata'])}; "
             f"pooled delta {s.loc['all', 'pooled_delta_pp']:.1f} pp (CI {s.loc['all', 'pooled_ci_low']:.1f}-{s.loc['all', 'pooled_ci_high']:.1f}); "
             f"unweighted mean {s.loc['all', 'unweighted_mean_pp']:.1f} pp; positive/zero/negative strata "
             f"{int(s.loc['all', 'n_strata_positive'])}/{int(s.loc['all', 'n_strata_zero'])}/{int(s.loc['all', 'n_strata_negative'])}; "
             f"clean rate {s.loc['all', 'rate_clean']:.2f}%, distracted rate {s.loc['all', 'rate_distracted']:.1f}%"]
    for lvl in ("model", "distractor", "family"):
        sub = t["summary"][t["summary"]["level"] == lvl]
        lines.append(lvl + ": " + "; ".join(f"{r['name']} {r['unweighted_mean_pp']:.1f} (pooled {r['pooled_delta_pp']:.1f})" for _, r in sub.iterrows()))
    q = t["quality"]
    lines.append("quality deltas (unweighted / pooled): " + "; ".join(f"{r['metric']} {r['unweighted_strata_delta']:+.3f}/{r['pooled_delta']:+.3f}" for _, r in q.iterrows()))
    js = t["judge_sensitivity"]
    lines.append("judges: " + "; ".join(f"{r['judge']}: pooled {r.get('pooled_delta_pp', np.nan):.1f}, clean FP {r.get('clean_false_positive_rate_pct', np.nan):.2f}%" for _, r in js.iterrows() if "pooled_delta_pp" in r and pd.notna(r["pooled_delta_pp"])))
    ln = t["length"]
    mts = ln[ln["scope"] == "MTS-Dialog"]
    lines.append("MTS length quintiles (short->long): " + ", ".join(f"{v:.1f}" for v in mts["pooled_delta_pp"]) + f"; rho={mts['spearman_rho'].iloc[0]:.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("results/raw_v3"))
    ap.add_argument("--data", type=Path, default=Path("data/ambient"))
    ap.add_argument("--out", type=Path, default=Path("results/v3"))
    ap.add_argument("--insertions", default="insertions_v3.parquet", help="insertion file under --data (insertions_v3.parquet for the adjacent set)")
    args = ap.parse_args()
    print(summarize(run(args.raw, args.data, args.out, args.insertions)))


if __name__ == "__main__":
    main()
