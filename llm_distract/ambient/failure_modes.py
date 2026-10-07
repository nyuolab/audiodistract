"""Failure-mode analysis of the documentation arm: where the inserted content lands in the note, how the outcome
(misattributed / correctly attributed but acted on / recorded only) depends on the aside, and verbatim examples.

Reads the folded raw files (``amb_notes``, ``amb_judgments``, ``amb_attribution``) and the frozen insertions, and writes
``amb_failure_sections.csv`` (share of incorporated notes with the content in each SOAP section, per model and family),
``amb_failure_by_aside.csv`` (outcome by aside class: relative vs non-relative, body-system match, nonliteral item),
``amb_failure_severity.csv`` and ``amb_failure_examples.md`` / ``.json`` (verbatim cases for the figure and the table).

    python -m llm_distract.ambient.failure_modes --raw results/raw_v3 --insertions data/ambient/insertions_v3.parquet --out results/v3
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from llm_distract.common.style import MODEL_SHORT

KEYS = ["source_dataset", "item_id", "model", "distractor_type"]
# note section headers -> coarse section; order matters (first match wins on a header line)
SECTION_PATTERNS = [
    ("chief_complaint", r"chief complaint|\bcc\b|reason for visit"),
    ("hpi", r"history of present illness|\bhpi\b|interval history|present illness"),
    ("pmh", r"past medical|medical history|\bpmh\b|past surgical|\bpsh\b|surgical history|medications?|allerg|review of systems|\bros\b|immuniz"),
    ("family_history", r"family history|\bfh\b|\bfhx\b"),
    ("social_history", r"social history|\bsh\b|\bshx\b|social"),
    ("subjective", r"subjective|^\**s\**[:.]|^s$"),
    ("objective", r"objective|^\**o\**[:.]|physical exam|vitals|examination|labs|results"),
    ("assessment", r"assessment|impression|diagnos|^\**a\**[:.]|differential|^a$"),
    ("plan", r"^\**p\**[:.]|\bplan\b|recommendation|follow.?up|disposition|orders|^p$"),
]
COARSE = {"chief_complaint": "Chief complaint / HPI", "hpi": "Chief complaint / HPI", "pmh": "History (PMH/meds/ROS)", "family_history": "Family / social history",
          "social_history": "Family / social history", "subjective": "Subjective (other)", "objective": "Objective", "assessment": "Assessment", "plan": "Plan"}
SECTION_ORDER = ["Chief complaint / HPI", "History (PMH/meds/ROS)", "Family / social history", "Subjective (other)", "Objective", "Assessment", "Plan", "Unsectioned"]
HEADER_RE = re.compile(r"^\s*(#+\s*|\*\*|__)?\s*([A-Za-z][A-Za-z /&()\-]{1,60}?)\s*(\*\*|__)?\s*:?\s*(\*\*|__)?\s*$")
STOP = {"the", "and", "with", "for", "from", "that", "this", "their", "have", "has", "had", "was", "were", "been", "being", "about", "after", "into",
        "very", "just", "over", "some", "more", "than", "then", "them", "they", "also", "only", "when", "what", "which", "while", "your", "you",
        "patient", "mentions", "mentioned", "says", "said", "small", "talk", "aside", "joke", "jokes", "joking", "friend", "person", "someone", "who",
        "week", "weeks", "month", "months", "last", "ago", "now", "still", "back", "getting", "doing", "going", "one", "two", "named", "called"}


def _stems(row: pd.Series) -> list[str]:
    """Content words of the assigned aside (topic phrase + assigned content), used to find the aside inside the note."""
    text = f"{row.get('distractor_topic', '')} {row.get('assigned_content', '')}".lower()
    text = re.sub(r"\(.*?\)", " ", text)
    words = [w for w in re.findall(r"[a-z][a-z'\-]{3,}", text) if w not in STOP]
    # A length-only sort of a set made the eight-stem cutoff hash-order dependent.
    stems = sorted({w[:6] for w in words}, key=lambda word: (-len(word), word))
    return stems[:8]


def _section_of_line(lines: list[str], idx: int) -> str:
    """Nearest preceding header line, mapped to a coarse section."""
    for j in range(idx, -1, -1):
        raw = lines[j].strip()
        if not raw:
            continue
        raw = re.sub(r"^(\*\*|__|#+\s*)?[SOAP]\s*[:.]\s*", lambda mm: mm.group(1) or "", raw)  # "**S: Subjective**" -> "**Subjective**"
        m = HEADER_RE.match(raw)
        cand = None
        if m:
            cand = m.group(2)
        else:  # inline header "**Plan:** text" or "Assessment: text"
            m2 = re.match(r"^\s*(#+\s*|\*\*|__)?([A-Za-z][A-Za-z /&()\-]{1,40}?)\s*(\*\*|__)?\s*:", raw)
            if m2 and (j == idx or True):
                cand = m2.group(2)
        if cand:
            low = cand.lower()
            for name, pat in SECTION_PATTERNS:
                if re.search(pat, low):
                    return COARSE[name]
    return "Unsectioned"


def locate(note: str, stems: list[str]) -> tuple[set, list[str]]:
    """Sections whose lines contain at least two of the aside's stems (or the single strongest stem), and those lines."""
    lines = note.splitlines()
    hits, found = set(), []
    for i, line in enumerate(lines):
        low = line.lower()
        n_hit = sum(1 for s in stems if s in low)
        if n_hit >= 2 or (n_hit == 1 and len(stems) <= 2):
            sec = _section_of_line(lines, i)
            if HEADER_RE.match(line.strip()) and len(line.strip()) < 40:
                continue  # a bare header
            hits.add(sec)
            found.append(line.strip())
    return hits, found


TAXONOMY = [  # (label, pattern over attribution rationale + contamination rationale + matched note lines); first match wins
    ("pathologised_conversation", r"tangential|off-topic|circumstantial|cognitive|mental status|thought process|confusion|coheren|delirium|hallucinat|memory (impairment|difficult)|anxiety|panic|psychiat|mood"),
    ("substituted_history", r"chief complaint|primary diagnosis|as the patient's own (condition|diagnosis|surgical|history|symptom|complaint|report)|patient's own (medical|surgical|past)|status post|s/p |own (diagnosis|surgery|procedure|condition)"),
    ("family_history_inflation", r"family history|hereditary|familial|genetic|inherit|risk factor|given (the )?family"),
    ("inferred_concern", r"unvoiced|unstated|implied concern|may reflect .{0,30}concern|screen|counsel|inquir|interest in|consider (evaluat|assess|screen)|prompt|explore"),
    ("recorded_as_patient_activity", r"activity|social history|recreational|hobby|plays|participates|lifestyle"),
    ("literalised_remark", r"convert|recast|misrepresent|reinterpret|reframe|as (a )?factual|as if reporting|literal|clinical detail|fabricated|household|treat(s|ed)? .{0,40}as (a )?(genuine|real|clinical|symptom|complaint)|misinterpret"),
]


def classify(rationale: str, lines: list[str]) -> str:
    text = " ".join([rationale or ""] + list(lines)).lower()
    for label, pat in TAXONOMY:
        if re.search(pat, text):
            return label
    return "other"


def run(raw: Path, insertions: Path, out: Path, examples_per_cell: int = 3,
        frozen_locations: Path | None = None) -> dict:
    notes = pd.read_parquet(raw / "amb_notes.parquet")
    judg = pd.read_parquet(raw / "amb_judgments.parquet")
    attr = pd.read_parquet(raw / "amb_attribution.parquet")
    ins = pd.read_parquet(insertions)
    for d in (notes, judg, attr, ins):
        d["item_id"] = d["item_id"].astype(str)
    ins = ins.set_index(["source_dataset", "item_id", "distractor_type"])
    dist = notes[notes["condition"] == "distracted"].merge(judg[judg["condition"] == "distracted"][KEYS + ["contamination_v3", "severity_v3", "judge_reasoning_v3"]], on=KEYS, how="left")
    dist = dist.merge(attr[KEYS + ["attribution", "used_for_patient", "judge_reasoning"]], on=KEYS, how="left")
    dist["outcome"] = np.select([dist["attribution"] == "patient", (dist["attribution"] == "correct") & (dist["used_for_patient"] == 1), dist["attribution"] == "correct"],
                                ["misattributed", "acted_on", "recorded_only"], default=np.where(dist["contamination_v3"] == 1, "incorporated_unclassified", "not_incorporated"))
    # aside classes
    def aside_info(r):
        i = ins.loc[(r["source_dataset"], r["item_id"], r["distractor_type"])]
        topic = str(i.get("distractor_topic", ""))
        rel = re.search(r"\((.*?);", topic)
        relation = rel.group(1).strip() if rel else ""
        cls = ("relative" if relation.startswith("my ") else "non-relative") if r["distractor_type"] == "bystander" else "nonliteral"
        return pd.Series({"topic": re.sub(r" \(.*\)$", "", topic), "relation": relation, "aside_class": cls, "body_system": i.get("body_system", ""),
                          "aside_text": i.get("distractor_conversation", ""), "aside_summary": i.get("distractor_summary", ""), "stems": _stems(i)})
    dist = pd.concat([dist, dist.apply(aside_info, axis=1)], axis=1)
    dist["model_short"] = dist["model"].map(lambda m: MODEL_SHORT.get(m, m))
    # sections
    secs, lines = zip(*[locate(n, s) for n, s in zip(dist["note"], dist["stems"])])
    dist["sections"], dist["note_lines"] = list(secs), list(lines)
    frozen = None
    if frozen_locations is not None:
        # Approved-version reproduction explicitly uses the saved annotations.
        # Primary judge outcomes do not depend on this lexical location parser.
        saved = pd.read_csv(frozen_locations, dtype={"item_id": str}).fillna({"sections": ""})
        if saved.duplicated(KEYS).any():
            raise ValueError("Duplicate keys in frozen location annotations")
        frozen = dist[KEYS].merge(saved[KEYS + ["sections", "in_assessment_or_plan", "action_in_plan"]],
                                  on=KEYS, how="left", validate="one_to_one", indicator=True)
        if len(saved) != len(dist) or not frozen["_merge"].eq("both").all():
            raise ValueError("Frozen location annotations must match every distracted note")
        dist["sections"] = frozen["sections"].map(lambda s: set(s.split(";")) if s else set())
    dist["failure_mode"] = [classify((a or "") + " " + (c or ""), l) if o in ("misattributed", "acted_on") else ("recorded_only" if o == "recorded_only" else "")
                            for a, c, l, o in zip(dist["judge_reasoning"], dist["judge_reasoning_v3"], dist["note_lines"], dist["outcome"])]
    inc = dist[dist["contamination_v3"] == 1].copy()
    rows = []
    for (m, fam), g in inc.groupby(["model_short", "distractor_type"]):
        n = len(g)
        for sec in SECTION_ORDER:
            rows.append({"model": m, "distractor": fam, "section": sec, "n_incorporated": n, "share": float(np.mean([sec in s for s in g["sections"]]))})
        rows.append({"model": m, "distractor": fam, "section": "Assessment or Plan", "n_incorporated": n, "share": float(np.mean([bool({"Assessment", "Plan"} & s) for s in g["sections"]]))})
        rows.append({"model": m, "distractor": fam, "section": "(located anywhere)", "n_incorporated": n, "share": float(np.mean([len(s) > 0 for s in g["sections"]]))})
    sections = pd.DataFrame(rows)
    # outcome by aside class / body-system match / relation
    n_enc = dist.groupby(["model_short", "distractor_type"]).size().rename("n_encounters")
    by = []
    for (m, fam, cls), g in dist.groupby(["model_short", "distractor_type", "aside_class"]):
        d = {"model": m, "distractor": fam, "aside_class": cls, "n": len(g)}
        for o in ["misattributed", "acted_on", "recorded_only"]:
            d[o] = float((g["outcome"] == o).mean())
        d["incorporated"] = float((g["contamination_v3"] == 1).mean())
        by.append(d)
    for (fam, cls), g in dist.groupby(["distractor_type", "aside_class"]):
        d = {"model": "all", "distractor": fam, "aside_class": cls, "n": len(g), "incorporated": float((g["contamination_v3"] == 1).mean())}
        for o in ["misattributed", "acted_on", "recorded_only"]:
            d[o] = float((g["outcome"] == o).mean())
        by.append(d)
    by_aside = pd.DataFrame(by)
    # topics most often misattributed (pooled over models)
    tp = dist.groupby(["distractor_type", "topic"]).agg(n=("outcome", "size"), incorporated=("contamination_v3", "mean"), misattributed=("outcome", lambda s: (s == "misattributed").mean()),
                                                         acted_on=("outcome", lambda s: (s == "acted_on").mean())).reset_index().sort_values(["distractor_type", "misattributed"], ascending=[True, False])
    # the inserted content in the assessment or plan, and whether that line carries an action (per encounter, all distracted notes)
    ACTION = re.compile(r"\b(screen|refer|referral|evaluat|assess for|consider|order|obtain|check|monitor|counsel|educat|discuss|follow[- ]?up|imaging|labs?|test|workup|work-up|recommend|advise|encourage|prescri|start)\w*", re.I)
    def ap_lines(note, lines):
        ls = note.splitlines()
        return [l for j, l in enumerate(ls) if l.strip() in lines and _section_of_line(ls, j) in ("Assessment", "Plan")]
    dist["ap_lines"] = [ap_lines(nt, set(l)) for nt, l in zip(dist["note"], dist["note_lines"])]
    dist["in_assessment_or_plan"] = dist["ap_lines"].map(bool)
    dist["action_in_plan"] = dist["ap_lines"].map(lambda ls: any(ACTION.search(l) for l in ls))
    if frozen is not None:
        for column in ["in_assessment_or_plan", "action_in_plan"]:
            if not frozen[column].isin([True, False]).all():
                raise ValueError(f"Non-boolean frozen annotation: {column}")
            dist[column] = frozen[column].astype(bool)
    act = []
    for keys_, g in list(dist.groupby(["model_short", "distractor_type"])) + [(("all", d_), g) for d_, g in dist.groupby("distractor_type")] + [(("all", "both"), dist)]:
        act.append({"model": keys_[0], "distractor": keys_[1], "n_encounters": len(g), "in_assessment_or_plan_pct": 100 * g["in_assessment_or_plan"].mean(),
                    "action_in_plan_pct": 100 * g["action_in_plan"].mean(), "acted_on_judge_pct": 100 * (g["outcome"] == "acted_on").mean(), "misattributed_judge_pct": 100 * (g["outcome"] == "misattributed").mean()})
    actions = pd.DataFrame(act)
    # failure-mode taxonomy among misattributed / acted-on notes
    harm = dist[dist["outcome"].isin(["misattributed", "acted_on"])]
    tax = harm.groupby(["model_short", "distractor_type", "outcome", "failure_mode"]).size().rename("n").reset_index()
    tax_all = harm.groupby(["distractor_type", "failure_mode"]).size().rename("n").reset_index().assign(model_short="all", outcome="both")
    tax = pd.concat([tax, tax_all], ignore_index=True)
    # severity distribution among incorporated notes
    sev = inc.groupby(["model_short", "distractor_type", "severity_v3"]).size().rename("n").reset_index()
    sev["share"] = sev["n"] / sev.groupby(["model_short", "distractor_type"])["n"].transform("sum")
    # examples: misattributed and acted_on, highest severity, longest transcripts first (full visits preferred), per model and family
    ex = []
    for (m, fam, o), g in dist[dist["outcome"].isin(["misattributed", "acted_on"])].groupby(["model_short", "distractor_type", "outcome"]):
        g = g.assign(has_lines=g["note_lines"].map(len) > 0).sort_values(["severity_v3", "has_lines", "transcript_chars"], ascending=[False, False, False])
        for _, r in g.head(examples_per_cell).iterrows():
            ex.append({"model": m, "distractor": fam, "outcome": o, "failure_mode": r["failure_mode"], "source_dataset": r["source_dataset"], "item_id": r["item_id"], "severity": int(r["severity_v3"]) if pd.notna(r["severity_v3"]) else None,
                       "transcript_chars": int(r["transcript_chars"]), "body_system": r["body_system"], "topic": r["topic"], "aside": r["aside_text"], "aside_summary": r["aside_summary"],
                       "note_lines": r["note_lines"][:4], "sections": sorted(r["sections"]), "attribution_reasoning": r["judge_reasoning"], "contamination_reasoning": r["judge_reasoning_v3"]})
    out.mkdir(parents=True, exist_ok=True)
    sections.to_csv(out / "amb_failure_sections.csv", index=False, float_format="%.6g")
    by_aside.to_csv(out / "amb_failure_by_aside.csv", index=False, float_format="%.6g")
    tp.to_csv(out / "amb_failure_topics.csv", index=False, float_format="%.6g")
    sev.to_csv(out / "amb_failure_severity.csv", index=False, float_format="%.6g")
    tax.to_csv(out / "amb_failure_taxonomy.csv", index=False)
    actions.to_csv(out / "amb_failure_actions.csv", index=False, float_format="%.6g")
    (out / "amb_failure_examples.json").write_text(json.dumps(ex, indent=1, default=str))
    md = ["# Failure-mode examples (verbatim)\n"]
    for e in ex:
        md.append(f"## {e['model']} · {e['distractor']} · {e['outcome']} · {e['source_dataset']} {e['item_id']} (severity {e['severity']}, {e['transcript_chars']} chars, {e['body_system']})\n")
        md.append("Inserted aside:\n\n```\n" + e["aside"] + "\n```\n")
        md.append("Note lines:\n\n" + "\n".join(f"> {l}" for l in e["note_lines"]) + "\n")
        md.append(f"Attribution judge: {e['attribution_reasoning']}\n")
    (out / "amb_failure_examples.md").write_text("\n".join(md))
    dist[KEYS + ["model_short", "outcome", "failure_mode", "severity_v3", "aside_class", "body_system", "topic", "in_assessment_or_plan", "action_in_plan"]].assign(sections=[";".join(sorted(s)) for s in dist["sections"]]).to_csv(out / "amb_failure_notes.csv", index=False)
    return {"sections": sections, "by_aside": by_aside, "topics": tp, "severity": sev, "taxonomy": tax, "actions": actions, "examples": ex, "notes": dist}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("results/raw_v3"))
    ap.add_argument("--insertions", type=Path, default=Path("data/ambient/insertions_v3.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/v3"))
    ap.add_argument("--frozen-locations", type=Path, help="Saved per-note locations for approved-version reproduction")
    args = ap.parse_args()
    t = run(args.raw, args.insertions, args.out, frozen_locations=args.frozen_locations)
    s = t["sections"]
    print("share of incorporated notes with the content in Assessment or Plan:")
    print(s[s.section == "Assessment or Plan"].pivot(index="model", columns="distractor", values="share").round(3).to_string())
    print("\nlocated anywhere:")
    print(s[s.section == "(located anywhere)"].pivot(index="model", columns="distractor", values="share").round(3).to_string())
    b = t["by_aside"]
    print("\noutcomes by aside class (all models):")
    print(b[b.model == "all"].round(3).to_string(index=False))
    print("\ncontent in the assessment or plan, per encounter:"); print(t["actions"].round(1).to_string(index=False))
    tx = t["taxonomy"]; print("\nfailure modes among misattributed/acted-on notes (all models):")
    print(tx[tx.model_short == "all"].pivot(index="failure_mode", columns="distractor_type", values="n").fillna(0).astype(int).to_string())
    print(f"\n{len(t['examples'])} examples -> {args.out / 'amb_failure_examples.md'}")


if __name__ == "__main__":
    main()
