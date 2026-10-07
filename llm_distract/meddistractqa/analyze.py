"""Tables of the registered MedDistractQA benchmark, recomputed from results/raw (CPU only).

Reads results/raw/qa_per_item.parquet (+ qa_engines.csv, qa_run_files.csv) and data/meddistractqa and writes
results/qa_model_table.csv, qa_group_summary.csv, qa_regression.csv, qa_prompting.csv, qa_prompting_summary.csv,
qa_category_competency.csv, qa_category_system.csv, qa_family_llama3.csv and qa_priming.csv (lexical-priming check).

Conventions: accuracy in percent; loss = clean accuracy - distracted accuracy in percentage points (pp, positive =
worse under distraction), computed within item on *evaluable pairs* (items with an extractable answer letter in both
conditions); answers without an extractable letter are "not evaluable" and are counted per condition. Accuracies
(``acc_*``) are over all items with not-evaluable answers scored as incorrect, and the same all-items rule gives the
sensitivity columns ``loss_*_pp_allitems``. The prompting comparison keeps the all-items rule because format collapse
under the alternative prompts is part of what it measures.

  python -m llm_distract.meddistractqa.analyze [--raw-dir results/raw] [--out-dir results]
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..common.stats import bootstrap_ci, mcnemar_exact, ols, paired_accuracy_diff, welch_t, wilcoxon_signed
from .data import load_meddistractqa
from .engines import DISPLAY_NAME, ENGINE_NAMES, GROUP_GENERAL, GROUP_MEDICAL, GROUP_PROPRIETARY, GROUPS, \
    LLAMA3_8B_FAMILY, group_of

DISTRACTORS = ("nonliteral", "bystander")
CONDITIONS = ("clean",) + DISTRACTORS
COLLAPSE_INVALID_RATE = 0.10   # a prompt run with >10% unparseable answers is flagged as format collapse
N_BOOT, SEED = 2000, 0


def load_raw(raw_dir: str | Path):
    raw_dir = Path(raw_dir)
    per_item = pd.read_parquet(raw_dir / "qa_per_item.parquet")
    run_files = pd.read_csv(raw_dir / "qa_run_files.csv")
    return per_item, run_files


def correctness(per_item: pd.DataFrame, prompt_variant: str) -> dict:
    """{(engine, condition): bool array over item_id} for one prompt variant."""
    sub = per_item[per_item["prompt_variant"] == prompt_variant].sort_values("item_id")
    return {key: g["is_correct"].to_numpy(dtype=bool) for key, g in sub.groupby(["engine", "condition"])}


def validity(per_item: pd.DataFrame, prompt_variant: str) -> dict:
    """{(engine, condition): bool array over item_id}, True where the answer was evaluable (a letter was extracted)."""
    sub = per_item[per_item["prompt_variant"] == prompt_variant].sort_values("item_id")
    return {key: ~g["is_invalid"].to_numpy(dtype=bool) for key, g in sub.groupby(["engine", "condition"])}


def invalid_counts(per_item: pd.DataFrame) -> dict:
    return per_item.groupby(["prompt_variant", "engine", "condition"])["is_invalid"].sum().to_dict()


def model_table(per_item: pd.DataFrame, run_files: pd.DataFrame) -> pd.DataFrame:
    """Per-engine accuracies, paired losses with binomial-covariance SE, exact McNemar p, invalids, provenance."""
    C, V, inv = correctness(per_item, "standard"), validity(per_item, "standard"), invalid_counts(per_item)
    files = run_files[run_files["prompt_variant"] == "standard"].set_index(["engine", "condition"])
    rows = []
    for e in ENGINE_NAMES:
        if (e, "clean") not in C:
            print(f"model_table: no standard runs for {e}, skipped")
            continue
        clean = C[(e, "clean")]
        row = {"engine": e, "display_name": DISPLAY_NAME[e], "group": group_of(e), "n": int(clean.size),
               "acc_clean": clean.mean() * 100}
        for d in DISTRACTORS:
            ok = V[(e, "clean")] & V[(e, d)]
            diff, se = paired_accuracy_diff(clean[ok], C[(e, d)][ok])
            row[f"acc_{d}"] = C[(e, d)].mean() * 100
            row[f"n_evaluable_{d}"] = int(ok.sum())
            row[f"loss_{d}_pp"], row[f"se_{d}_pp"] = -diff, se
            row[f"mcnemar_p_{d}"] = mcnemar_exact(clean[ok], C[(e, d)][ok])
            diff_all, se_all = paired_accuracy_diff(clean, C[(e, d)])
            row[f"loss_{d}_pp_allitems"], row[f"se_{d}_pp_allitems"] = -diff_all, se_all
        for c in CONDITIONS:
            row[f"invalid_{c}"] = int(inv[("standard", e, c)])
            row[f"max_tokens_{c}"] = int(files.loc[(e, c), "max_tokens"])
            row[f"file_{c}"] = files.loc[(e, c), "source_file"]
        rows.append(row)
    return pd.DataFrame(rows)


def group_summary(mt: pd.DataFrame) -> pd.DataFrame:
    """Group means (models as units) and Welch t-tests, per distractor."""
    rows = []
    for d in DISTRACTORS:
        col = f"loss_{d}_pp"
        by = {g: mt.loc[mt["group"] == g, col].to_numpy() for g in GROUPS}
        for name, vals in list(by.items()) + [("All", mt[col].to_numpy())]:
            rows.append({"distractor": d, "kind": "group", "name": name, "n": vals.size, "mean_loss_pp": vals.mean(),
                         "sd_pp": vals.std(ddof=1), "sem_pp": vals.std(ddof=1) / np.sqrt(vals.size),
                         "min_pp": vals.min(), "max_pp": vals.max()})
        for a, b in ((GROUP_PROPRIETARY, GROUP_GENERAL), (GROUP_PROPRIETARY, GROUP_MEDICAL),
                     (GROUP_MEDICAL, GROUP_GENERAL)):
            t, p1 = welch_t(by[a], by[b], "less")
            _, p2 = welch_t(by[a], by[b], "two-sided")
            rows.append({"distractor": d, "kind": "welch", "name": f"{a} < {b}", "n": f"{by[a].size} vs {by[b].size}",
                         "mean_loss_pp": by[a].mean() - by[b].mean(), "t_stat": t, "p_one_tailed": p1, "p_two_tailed": p2})
    return pd.DataFrame(rows)


def regression(mt: pd.DataFrame) -> pd.DataFrame:
    """OLS of loss on clean accuracy across the engines."""
    return pd.DataFrame([{"distractor": d, "n": len(mt), **ols(mt["acc_clean"], mt[f"loss_{d}_pp"])} for d in DISTRACTORS])


def prompting(per_item: pd.DataFrame, run_files: pd.DataFrame):
    """Effect of the two alternative prompts on nonliteral-distracted accuracy (paired within engine and item)."""
    std, inv = correctness(per_item, "standard"), invalid_counts(per_item)
    mt = run_files.set_index(["prompt_variant", "engine", "condition"])["max_tokens"]
    rows = []
    excluded_cap_mismatch = []
    for variant in ("ignore", "structured"):
        alt = correctness(per_item, variant)
        for e in ENGINE_NAMES:
            if (e, "nonliteral") not in alt:
                continue
            standard_cap = int(mt[("standard", e, "nonliteral")])
            variant_cap = int(mt[(variant, e, "nonliteral")])
            caps_match = standard_cap == variant_cap
            if (e, "clean") in alt:
                standard_clean_cap = int(mt[("standard", e, "clean")])
                variant_clean_cap = int(mt[(variant, e, "clean")])
                caps_match = caps_match and standard_clean_cap == variant_clean_cap
                caps_match = caps_match and standard_clean_cap == standard_cap and variant_clean_cap == variant_cap
            if not caps_match:
                excluded_cap_mismatch.append((variant, e, standard_cap, variant_cap))
                continue
            s, a, n = std[(e, "nonliteral")], alt[(e, "nonliteral")], std[(e, "nonliteral")].size
            row = {"prompt_variant": variant, "engine": e, "display_name": DISPLAY_NAME[e], "group": group_of(e),
                   "condition": "nonliteral", "n": n, "acc_standard": s.mean() * 100, "acc_variant": a.mean() * 100,
                   "delta_pp": (a.mean() - s.mean()) * 100, "mcnemar_p": mcnemar_exact(s, a),
                   "invalid_standard": int(inv[("standard", e, "nonliteral")]),
                   "invalid_variant": int(inv[(variant, e, "nonliteral")]),
                   "max_tokens_standard": standard_cap,
                   "max_tokens_variant": variant_cap}
            row["invalid_rate_variant"] = row["invalid_variant"] / n
            row["format_collapse"] = row["invalid_rate_variant"] > COLLAPSE_INVALID_RATE
            if (e, "clean") in alt:   # the structured prompt also has clean runs -> distraction loss under each prompt
                sc, ac = std[(e, "clean")], alt[(e, "clean")]
                row.update({"acc_clean_standard": sc.mean() * 100, "acc_clean_variant": ac.mean() * 100,
                            "loss_standard_pp": (sc.mean() - s.mean()) * 100, "loss_variant_pp": (ac.mean() - a.mean()) * 100})
                row["delta_loss_pp"] = row["loss_variant_pp"] - row["loss_standard_pp"]
            rows.append(row)
    if excluded_cap_mismatch:
        print("Prompting comparisons excluded for mismatched output caps:")
        for variant, engine, standard_cap, variant_cap in excluded_cap_mismatch:
            print(f"  {variant}: {engine} ({standard_cap} standard vs {variant_cap} variant tokens)")
    table = pd.DataFrame(rows)

    def summarise(sub: pd.DataFrame, variant: str, subset: str, metric: str) -> dict:
        x = sub[metric].to_numpy(dtype=float)
        lo, hi = stats.t.interval(0.95, x.size - 1, loc=x.mean(), scale=stats.sem(x))
        return {"prompt_variant": variant, "subset": subset, "metric": metric, "k": x.size, "mean_pp": x.mean(),
                "ci_low": lo, "ci_high": hi, "t_p": stats.ttest_1samp(x, 0).pvalue, "wilcoxon_p": wilcoxon_signed(x),
                "n_mcnemar_sig": int((sub["mcnemar_p"] < 0.05).sum()), "n_format_collapse": int(sub["format_collapse"].sum())}

    summary = []
    for variant, g in table.groupby("prompt_variant"):
        metrics = ["delta_pp"] + (["delta_loss_pp"] if g["delta_loss_pp"].notna().all() else [])
        for metric in metrics:
            summary.append(summarise(g, variant, "all", metric))
            if g["format_collapse"].any():
                summary.append(summarise(g[~g["format_collapse"]], variant, "format-stable", metric))
    return table, pd.DataFrame(summary)


def category_tables(per_item: pd.DataFrame, dataset: pd.DataFrame, min_items: int = 15):
    """Per-engine and all-model-mean loss per USMLE competency / system category (evaluable pairs)."""
    std = per_item[(per_item["prompt_variant"] == "standard") & per_item["engine"].isin(ENGINE_NAMES)]
    wide = std.pivot_table(index=["engine", "item_id"], columns="condition", values=["is_correct", "is_invalid"], aggfunc="first")
    out = {}
    for label in ("medical_competency", "medical_system"):
        cats = dataset.set_index("item_id")[label]
        n_items = dataset[label].value_counts()
        rows = []
        for e, g in wide.groupby(level="engine"):
            cat = cats.reindex(g.index.get_level_values("item_id")).to_numpy()
            corr = {c: g[("is_correct", c)].to_numpy(dtype=bool) for c in CONDITIONS}
            valid = {c: ~g[("is_invalid", c)].to_numpy(dtype=bool) for c in CONDITIONS}
            for c_name in n_items.index:
                m = cat == c_name
                acc_clean = 100 * corr["clean"][m].mean()
                losses = {}
                for d in DISTRACTORS:
                    ok = m & valid["clean"] & valid[d]
                    losses[d] = 100 * (corr["clean"][ok].mean() - corr[d][ok].mean()) if ok.any() else np.nan
                for d, v in list(losses.items()) + [("mean", float(np.nanmean(list(losses.values()))))]:
                    rows.append({"engine": e, "group": group_of(e), "category": c_name, "n_items": int(n_items[c_name]),
                                 "reported": bool(n_items[c_name] >= min_items), "distractor": d, "acc_clean": acc_clean,
                                 "acc_distracted": acc_clean - v, "loss_pp": v})
        df = pd.DataFrame(rows)
        mean = df.groupby(["category", "n_items", "reported", "distractor"], as_index=False)["loss_pp"].mean()
        mean.insert(0, "engine", f"Mean ({std['engine'].nunique()} models)")
        out[label] = pd.concat([df, mean], ignore_index=True)
    return out["medical_competency"], out["medical_system"]


def _boot(d: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Percentile bootstrap CI and two-sided p-value for the mean of per-item differences (pp)."""
    point, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"estimate_pp": point, "ci_low": lo, "ci_high": hi, "p_value": max(min(p, 1.0), 1 / n_boot)}


def family_table(per_item: pd.DataFrame) -> pd.DataFrame:
    """Llama-3-8B family: clean accuracy and losses with bootstrap CIs; pairwise paired comparisons over items."""
    C, V = correctness(per_item, "standard"), validity(per_item, "standard")
    rows = []
    for e in LLAMA3_8B_FAMILY:
        clean = C[(e, "clean")].astype(float)
        rows.append({"kind": "model", "engine": e, "display_name": DISPLAY_NAME[e], "reference": "", "metric": "acc_clean",
                     **_boot(clean * 100), "test": "bootstrap"})
        for d in DISTRACTORS:
            ok = V[(e, "clean")] & V[(e, d)]
            rows.append({"kind": "model", "engine": e, "display_name": DISPLAY_NAME[e], "reference": "",
                         "metric": f"loss_{d}_pp", **_boot(((clean - C[(e, d)]) * 100)[ok]),
                         "mcnemar_p": mcnemar_exact(C[(e, "clean")][ok], C[(e, d)][ok]), "test": "paired bootstrap; McNemar"})
    for a, b in combinations(LLAMA3_8B_FAMILY, 2):
        ca, cb = C[(a, "clean")].astype(float), C[(b, "clean")].astype(float)
        rows.append({"kind": "pair", "engine": b, "display_name": DISPLAY_NAME[b], "reference": a, "metric": "acc_clean",
                     **_boot((cb - ca) * 100), "mcnemar_p": mcnemar_exact(C[(a, "clean")], C[(b, "clean")]),
                     "test": "paired bootstrap; McNemar"})
        for d in DISTRACTORS:
            ok = V[(a, "clean")] & V[(a, d)] & V[(b, "clean")] & V[(b, d)]
            diff = ((cb - C[(b, d)]) - (ca - C[(a, d)]))[ok]
            rows.append({"kind": "pair", "engine": b, "display_name": DISPLAY_NAME[b], "reference": a,
                         "metric": f"loss_{d}_pp", **_boot(diff * 100), "test": "paired bootstrap"})
    return pd.DataFrame(rows)


def priming_table(per_item: pd.DataFrame, dataset: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Lexical priming: each inserted sentence reuses a clinical term from one (incorrect) answer option; among wrong
    answers, how often is that option chosen, in the clean and in the distracted condition (chance 1/3), and how often do
    answers that flip from correct to wrong land on it. One row per engine x distractor plus pooled rows."""
    import json, re
    def norm(x): return re.sub(r"[^a-z0-9 ]", "", str(x).lower()).strip()
    d = dataset.set_index("item_id")
    correct = d["correct_answer"].str.strip().str.upper()
    std = per_item[(per_item["prompt_variant"] == "standard") & per_item["engine"].isin(ENGINE_NAMES)].copy()
    std["ans"] = std["parsed_answer"].str.extract(r"([A-D])", expand=False)
    std["correct"] = std["item_id"].map(correct)
    rows = []
    for dist in DISTRACTORS:
        conf = json.load(open(data_dir / f"confounders_{dist}.json"))
        src = {}
        for i, rec in enumerate(conf):
            topic = norm(rec["clinical_topic"])
            opts = {k: norm(d.loc[i, f"choice_{k}"]) for k in "ABCD"}
            hit = [k for k, v in opts.items() if v == topic] or [k for k, v in opts.items() if topic and (topic in v or v in topic)]
            src[i] = hit[0] if hit else None
        sub = std[std["condition"].isin(["clean", dist]) & std["ans"].notna()].copy()
        sub["primed"] = sub["item_id"].map(src)
        sub = sub[sub["primed"].notna() & (sub["primed"] != sub["correct"])]
        wide = sub.pivot_table(index=["engine", "item_id"], columns="condition", values="ans", aggfunc="first").reset_index()
        wide["correct"] = wide["item_id"].map(correct); wide["primed"] = wide["item_id"].map(src)
        for e, g in list(sub.groupby("engine")) + [("all", sub)]:
            wrong = g[g["ans"] != g["correct"]]
            w = wide if e == "all" else wide[wide["engine"] == e]
            flipped = w[(w["clean"] == w["correct"]) & (w[dist] != w["correct"]) & w[dist].notna()]
            rows.append({"distractor": dist, "engine": e, "display_name": DISPLAY_NAME.get(e, e), "group": group_of(e) if e != "all" else "all",
                         "n_items_primed_wrong_option": int(g["item_id"].nunique()),
                         "primed_share_wrong_clean_pct": 100 * (wrong[wrong["condition"] == "clean"]["ans"] == wrong[wrong["condition"] == "clean"]["primed"]).mean(),
                         "primed_share_wrong_distracted_pct": 100 * (wrong[wrong["condition"] == dist]["ans"] == wrong[wrong["condition"] == dist]["primed"]).mean(),
                         "n_newly_wrong": int(len(flipped)), "primed_share_newly_wrong_pct": 100 * (flipped[dist] == flipped["primed"]).mean() if len(flipped) else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="results/raw")
    ap.add_argument("--data", default="data/meddistractqa/meddistractqa_v2.parquet")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_item, run_files = load_raw(args.raw_dir)
    dataset = load_meddistractqa(args.data)

    mt = model_table(per_item, run_files)
    mt.to_csv(out / "qa_model_table.csv", index=False, float_format="%.4f")
    gs = group_summary(mt)
    gs.to_csv(out / "qa_group_summary.csv", index=False, float_format="%.6g")
    regression(mt).to_csv(out / "qa_regression.csv", index=False, float_format="%.6g")
    pt, ps = prompting(per_item, run_files)
    pt.to_csv(out / "qa_prompting.csv", index=False, float_format="%.4f")
    ps.to_csv(out / "qa_prompting_summary.csv", index=False, float_format="%.4g")
    comp, syst = category_tables(per_item, dataset)
    comp.to_csv(out / "qa_category_competency.csv", index=False, float_format="%.3f")
    syst.to_csv(out / "qa_category_system.csv", index=False, float_format="%.3f")
    family_table(per_item).to_csv(out / "qa_family_llama3.csv", index=False, float_format="%.4g")
    pr = priming_table(per_item, dataset, Path(args.data).parent)
    pr.to_csv(out / "qa_priming.csv", index=False, float_format="%.4g")
    print("priming (pooled):\n", pr[pr["engine"] == "all"][["distractor", "primed_share_wrong_clean_pct", "primed_share_wrong_distracted_pct", "n_newly_wrong", "primed_share_newly_wrong_pct"]].round(1).to_string(index=False))

    g = gs[gs["kind"] == "group"].pivot(index="name", columns="distractor", values="mean_loss_pp")
    print("mean loss (pp) by group:\n", g.round(2).to_string())
    print("Welch one-tailed p:\n", gs[gs["kind"] == "welch"][["distractor", "name", "p_one_tailed", "p_two_tailed"]].to_string(index=False))
    for d in DISTRACTORS:
        i, j = mt[f"loss_{d}_pp"].idxmin(), mt[f"loss_{d}_pp"].idxmax()
        print(f"{d}: range {mt.loc[i, f'loss_{d}_pp']:.2f} ({mt.loc[i, 'engine']}) to {mt.loc[j, f'loss_{d}_pp']:.2f} ({mt.loc[j, 'engine']})")
    print("regression r:", regression(mt)[["distractor", "r", "r2", "p"]].to_string(index=False))
    print("prompting:\n", ps.to_string(index=False))


if __name__ == "__main__":
    main()
