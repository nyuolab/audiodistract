"""Collect generated notes and judgments into a common analysis table.

Reads ``<runs-root>/<model_label>/<split>/<distractor>/notes.parquet`` (from ``llm_distract.ambient.generate_notes``)
with the judge outputs written next to it by ``llm_distract.ambient.judge`` (``judge_contamination_paired``,
``judge_quality`` and, optionally, ``judge_contamination_single``), converts them to the ``amb_notes`` /
``amb_judgments`` schema and replaces the rows of results/raw with the same (source_dataset, item_id, model,
distractor_type, condition) keys. The files being replaced are kept once as ``amb_*_original.parquet``.

    python -m llm_distract.ambient.export_reruns --runs-root outputs/cluster/ambient_aci --raw results/raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from llm_distract.ambient.common import family_of, write_parquet

KEYS = ["source_dataset", "item_id", "model", "distractor_type", "condition"]
QUALITY = ["clinical_correctness", "completeness", "succinctness", "hallucination", "overall_quality"]
INT_COLS = ["contamination_v3", "severity_v3", "contamination_v2_gpt", "severity_v2"] + QUALITY


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["item_id"] = df["item_id"].astype(str)
    return df


ATTR_COLS = ["source_dataset", "item_id", "family", "model", "distractor_type", "condition", "attribution", "used_for_patient", "judge_reasoning",
             "parse_error", "judge_model", "judge_protocol"]


def collect_attribution(root: Path) -> pd.DataFrame:
    """Attribution sub-judge outputs (``judge_attribution.parquet`` next to the notes), tidy; empty frame if none."""
    frames = []
    for path in sorted(root.glob("*/*/*/judge_attribution.parquet")):
        a = _read(path)
        a["attribution"] = a["attribution"].replace({"third_party": "correct"})
        a.insert(2, "family", a["source_dataset"].map(family_of))
        frames.append(a.reindex(columns=ATTR_COLS))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ATTR_COLS)


def fold_attribution(raw: Path, new: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Replace/append rows of results/raw/amb_attribution.parquet by (source_dataset, item_id, model, distractor_type)."""
    path = raw / "amb_attribution.parquet"
    keys = ["source_dataset", "item_id", "model", "distractor_type"]
    base = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=ATTR_COLS)
    base["item_id"] = base["item_id"].astype(str)
    key_new = set(map(tuple, new[keys].values))
    keep = ~base[keys].apply(tuple, axis=1).isin(key_new) if len(base) else pd.Series([], dtype=bool)
    merged = pd.concat([base[keep] if len(base) else base, new[ATTR_COLS]], ignore_index=True)
    write_parquet(merged, path, meta={**meta, "rows_replaced": int((~keep).sum()) if len(base) else 0, "rows_added": len(new)})
    print(f"{path}: total {len(merged)} attribution rows")
    return merged


def collect(root: Path) -> tuple:
    notes, judgments = [], []
    for notes_path in sorted(root.glob("*/*/*/notes.parquet")):
        d = notes_path.parent
        if not (d / "judge_contamination_paired.parquet").exists():
            print(f"  skipping {d}: paired judge pending", flush=True)
            continue
        n = _read(notes_path)
        n["family"] = n["source_dataset"].map(family_of)
        n["note_chars"], n["transcript_chars"] = n["note"].str.len(), n["transcript"].str.len()
        notes.append(n[KEYS[:2] + ["family"] + KEYS[2:] + ["note", "note_chars", "transcript_chars"]])
        v3 = _read(d / "judge_contamination_paired.parquet")[KEYS + ["contamination", "severity", "judge_reasoning", "judge_protocol"]]
        v3 = v3.rename(columns={"contamination": "contamination_v3", "severity": "severity_v3", "judge_reasoning": "judge_reasoning_v3"})
        if (d / "judge_quality.parquet").exists():
            q = _read(d / "judge_quality.parquet")[KEYS + QUALITY]
            j = v3.merge(q, on=KEYS, how="outer", validate="one_to_one")
        else:  # quality judge still pending: keep the contamination judgments, fill quality with NA
            j = v3.copy()
            for col in QUALITY:
                j[col] = pd.NA
        single = d / "judge_contamination_single.parquet"
        if single.exists():
            v2 = _read(single)[KEYS + ["contamination", "severity"]].rename(columns={"contamination": "contamination_v2_gpt", "severity": "severity_v2"})
            j = j.merge(v2, on=KEYS, how="outer", validate="one_to_one")
        else:
            j["contamination_v2_gpt"], j["severity_v2"] = pd.NA, pd.NA
        j.insert(2, "family", j["source_dataset"].map(family_of))
        judgments.append(j)
    if not notes:
        raise SystemExit(f"no notes.parquet under {root}")
    return pd.concat(notes, ignore_index=True), pd.concat(judgments, ignore_index=True)


def fold(raw: Path, name: str, new: pd.DataFrame, meta: dict) -> pd.DataFrame:
    path, orig = raw / f"{name}.parquet", raw / f"{name}_original.parquet"
    raw.mkdir(parents=True, exist_ok=True)
    if not path.exists():  # fresh namespace (e.g. results/raw_v3): nothing to preserve
        base = new.iloc[0:0].copy()
    else:
        base = pd.read_parquet(path)
        if not orig.exists():
            path.rename(orig)
    base["item_id"] = base["item_id"].astype(str)
    key_new = set(map(tuple, new[KEYS].values))
    keep = ~base[KEYS].apply(tuple, axis=1).isin(key_new)
    merged = pd.concat([base[keep], new[base.columns]], ignore_index=True)
    for col in INT_COLS:
        if col in merged.columns:
            merged[col] = merged[col].astype("Int8")
    write_parquet(merged, path, meta={**meta, "rows_replaced": int((~keep).sum()), "rows_added": int(len(new) - (~keep).sum())})
    print(f"{path}: replaced {int((~keep).sum())} rows, added {int(len(new) - (~keep).sum())}, total {len(merged)}")
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=Path("outputs/cluster/ambient_aci"))
    ap.add_argument("--raw", type=Path, default=Path("results/raw"))
    ap.add_argument("--source", default="released-code rerun (ACI-Bench exchanges relabelled to [doctor]/[patient] and spliced mid-dialogue)",
                    help="provenance note stored with the folded files")
    args = ap.parse_args()
    notes, judgments = collect(args.runs_root)
    meta = {"exporter": "export_reruns", "runs_root": str(args.runs_root), "source": args.source}
    fold(args.raw, "amb_notes", notes, meta)
    fold(args.raw, "amb_judgments", judgments, meta)
    attribution = collect_attribution(args.runs_root)
    if len(attribution):
        fold_attribution(args.raw, attribution, meta)


if __name__ == "__main__":
    main()
