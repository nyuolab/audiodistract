"""Build, load and validate the MedDistractQA v2 table (one row per MedQA test item, both distractors).

Inputs (data/meddistractqa/): the frozen GPT-4o distractor sentences ``confounders_{nonliteral,bystander}.json``
and the o3-mini labels ``labels.csv``; the clean questions come from the public HF dataset
GBaker/MedQA-USMLE-4-options (test split, 1,273 items, positional alignment). The distracted question is produced by
``insert_distractor`` - the exact rule of the original runner.

CLI: ``python -m llm_distract.meddistractqa.data [--validate-hf KrithikV/MedDistractQA] [--run-files CSV]``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/meddistractqa")
HF_DATASET = "GBaker/MedQA-USMLE-4-options"
HF_RELEASE = "KrithikV/MedDistractQA"
CONDITIONS = ("clean", "nonliteral", "bystander")
COLUMNS = ["item_id", "clean_question", "choice_A", "choice_B", "choice_C", "choice_D", "correct_answer",
           "nonliteral_sentence", "bystander_sentence", "nonliteral_question", "bystander_question",
           "medical_system", "medical_competency"]
# Short label names of the public release (create_gen_dset.py), keyed by the number returned by o3-mini.
SYSTEM_NAMES = {1: "Human Development", 2: "Immune System", 3: "Blood & Lymphoreticular", 4: "Behavioral Health",
                5: "Nervous System & Special Senses", 6: "Musculoskeletal & Skin", 7: "Cardiovascular System",
                8: "Respiratory System", 9: "Gastrointestinal System", 10: "Renal & Urinary System",
                11: "Pregnancy & Childbirth", 12: "Endocrine System", 13: "Multisystem Processes",
                14: "Biostatistics & Epidemiology", 15: "Communication Skills", 16: "Legal/Ethical Issues"}
COMPETENCY_NAMES = {1: "Medical Knowledge/Scientific Concepts", 2: "Patient Care: Diagnosis",
                    3: "Patient Care: Management", 4: "Communication",
                    5: "Professionalism, Including Legal and Ethical Issues",
                    6: "Systems-based Practice, Including Patient Safety", 7: "Practice-based Learning"}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?]) +")


def insert_distractor(question: str, sentence: str) -> str:
    """Insert ``sentence`` before the final sentence of ``question`` (original rule; a one-sentence question gets
    the distractor prepended, and runs of spaces after sentence-final punctuation collapse to one space)."""
    sentences = _SENTENCE_SPLIT.split(question)
    sentences.insert(-1, sentence)
    return " ".join(sentences)


def load_confounders(path: str | Path) -> list[str]:
    """Distractor sentences in MedQA test order, with the generator's surrounding double quotes stripped."""
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    return [r["confounder_sentence"].strip('"') for r in records]


def load_medqa_test(hf_dataset: str = HF_DATASET) -> pd.DataFrame:
    """Clean MedQA test items (cache-aware HF download)."""
    from datasets import load_dataset

    ds = load_dataset(hf_dataset, split="test")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({"item_id": i, "clean_question": ex["question"], "correct_answer": ex["answer_idx"],
                     **{f"choice_{k}": ex["options"][k] for k in "ABCD"}})
    return pd.DataFrame(rows)


def build_labels(system_json: str | Path, competency_json: str | Path, medqa: pd.DataFrame,
                 out_csv: str | Path) -> pd.DataFrame:
    """Convert the two index-aligned o3-mini label files into the compact labels.csv."""
    out = {"item_id": medqa["item_id"].tolist()}
    for col, path, names in (("medical_system", system_json, SYSTEM_NAMES),
                             ("medical_competency", competency_json, COMPETENCY_NAMES)):
        with open(path, encoding="utf-8") as fh:
            recs = json.load(fh)
        if len(recs) != len(medqa) or any(r["question"] != q for r, q in zip(recs, medqa["clean_question"])):
            raise ValueError(f"{path}: label file is not aligned with the MedQA test split")
        out[col] = [names[int(r["section number"])] for r in recs]
    labels = pd.DataFrame(out)
    labels.to_csv(out_csv, index=False)
    return labels


def build_meddistractqa(data_dir: str | Path = DATA_DIR, hf_dataset: str = HF_DATASET) -> pd.DataFrame:
    """Assemble meddistractqa_v2 and write it as parquet and JSONL into ``data_dir``."""
    data_dir = Path(data_dir)
    df = load_medqa_test(hf_dataset)
    for cond in ("nonliteral", "bystander"):
        sentences = load_confounders(data_dir / f"confounders_{cond}.json")
        if len(sentences) != len(df):
            raise ValueError(f"{cond}: {len(sentences)} sentences for {len(df)} items")
        df[f"{cond}_sentence"] = sentences
        df[f"{cond}_question"] = [insert_distractor(q, s) for q, s in zip(df["clean_question"], sentences)]
    df = df.merge(pd.read_csv(data_dir / "labels.csv"), on="item_id", how="left")[COLUMNS]
    df.to_parquet(data_dir / "meddistractqa_v2.parquet", index=False)
    df.to_json(data_dir / "meddistractqa_v2.jsonl", orient="records", lines=True, force_ascii=False)
    return df


def load_meddistractqa(path: str | Path | None = None) -> pd.DataFrame:
    """Load the built table (parquet if present, else the JSONL twin)."""
    path = Path(path) if path else DATA_DIR / "meddistractqa_v2.parquet"
    if path.suffix == ".parquet" and path.exists():
        return pd.read_parquet(path)
    return pd.read_json(path.with_suffix(".jsonl"), orient="records", lines=True)


def question_md5(questions) -> str:
    """Order-sensitive fingerprint of a list of question strings (shared with export_raw)."""
    return hashlib.md5(json.dumps(list(questions), ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_against_hf(df: pd.DataFrame, source: str = HF_RELEASE) -> dict:
    """Field-wise match rates against the public release (a local directory holding the two JSON files or the
    HF dataset id; positional alignment)."""
    from huggingface_hub import hf_hub_download

    report = {}
    for cond in ("nonliteral", "bystander"):
        fname = f"MedDistractQA_{cond.capitalize()}.json"
        path = Path(source) / fname if Path(source).is_dir() else hf_hub_download(source, fname, repo_type="dataset")
        with open(path, encoding="utf-8") as fh:
            rel = json.load(fh)
        n = len(rel)
        checks = {
            "question": [r["question"] == q for r, q in zip(rel, df[f"{cond}_question"])],
            "sentence": [r["distracting_sentence"] == s for r, s in zip(rel, df[f"{cond}_sentence"])],
            "choices": [all(r["question_choices"][k] == c for k, c in zip("ABCD", cs))
                        for r, cs in zip(rel, df[["choice_A", "choice_B", "choice_C", "choice_D"]].itertuples(index=False))],
            "answer": [r["correct_answer"] == a for r, a in zip(rel, df["correct_answer"])],
            "medical_system": [r["medical_system"] == a for r, a in zip(rel, df["medical_system"])],
            "medical_competency": [r["medical_competency"] == a for r, a in zip(rel, df["medical_competency"])],
        }
        report[cond] = {"n_release": n, "n_built": len(df), **{k: sum(v) for k, v in checks.items()}}
    return report


def validate_against_runs(df: pd.DataFrame, run_files_csv: str | Path) -> dict:
    """Check that the built questions fingerprint-match the queries of every exported standard-prompt run."""
    runs = pd.read_csv(run_files_csv)
    runs = runs[runs["prompt_variant"] == "standard"]
    out = {}
    for cond in CONDITIONS:
        col = "clean_question" if cond == "clean" else f"{cond}_question"
        expected = question_md5(df[col])
        sub = runs[runs["condition"] == cond]
        out[cond] = {"n_runs": int(len(sub)), "n_matching": int((sub["query_md5"] == expected).sum())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--hf-dataset", default=HF_DATASET)
    ap.add_argument("--labels-system", help="section_results_medqa_system.json (rebuild labels.csv)")
    ap.add_argument("--labels-competency", help="section_results_medqa_competencyV2.json")
    ap.add_argument("--validate-hf", nargs="?", const=HF_RELEASE, help="HF dataset id or local dir with the release JSONs")
    ap.add_argument("--run-files", help="results/raw/qa_run_files.csv to cross-check queries of the exported runs")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    if args.labels_system and args.labels_competency:
        build_labels(args.labels_system, args.labels_competency, load_medqa_test(args.hf_dataset), data_dir / "labels.csv")
    df = build_meddistractqa(data_dir, args.hf_dataset)
    print(f"built {len(df)} items -> {data_dir / 'meddistractqa_v2.parquet'} (+ .jsonl)")
    if args.validate_hf:
        print("release match:", json.dumps(validate_against_hf(df, args.validate_hf)))
    if args.run_files:
        print("run-query match:", json.dumps(validate_against_runs(df, args.run_files)))


if __name__ == "__main__":
    main()
