"""Body-system label per documentation encounter (for the adjacent insertion design of ``perturb_v2 --systems``).

GPT-5.4 reads the clean transcript (and the reference note) and returns the organ system the encounter is about,
inferred from any complaint, condition, symptom or question it contains; excerpts with no clinical content (an
allergy list, a social history) are labelled ``general``. Output: ``data/ambient/encounter_systems.parquet`` with
``source_dataset, item_id, body_system, rationale``.

    python -m llm_distract.ambient.label_systems --pairs data/ambient/pairs_v2.parquet --out data/ambient/encounter_systems.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

SYSTEMS = ["musculoskeletal", "cardiovascular", "respiratory", "gastrointestinal", "neurological", "genitourinary", "endocrine",
           "dermatological", "psychiatric", "ent_ophthalmic_dental", "hematologic_oncologic", "obstetric_gynecologic", "general"]
SYSTEM_PROMPT = ("You classify clinical encounter transcripts by the organ system they are about. Respond only with JSON.")
USER = """\
Which organ system is this encounter (or excerpt of an encounter) about? Infer it from the chief complaint, any condition, symptom,
examination, medication or question that is discussed. If the excerpt contains no clinical content at all (for example only an
allergy list, a social history or scheduling talk), answer "general".

Allowed labels: {labels}

Transcript:
{transcript}

Reference note (may be empty):
{note}

Return JSON: {{"body_system": "<label>", "rationale": "<one short sentence>"}}
"""
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def label_one(transcript: str, note: str, model: str, retries: int = 3) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(model=model, temperature=0.0, max_completion_tokens=200, response_format={"type": "json_object"},
                                               messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                                         {"role": "user", "content": USER.format(labels=", ".join(SYSTEMS), transcript=transcript[:6000], note=(note or "")[:1500])}])
            text = r.choices[0].message.content or ""
            j = json.loads(text) if text.strip().startswith("{") else json.loads(JSON_RE.search(text).group())
            label = str(j["body_system"]).strip().lower()
            if label not in SYSTEMS:
                raise ValueError(label)
            return {"body_system": label, "rationale": str(j.get("rationale", ""))[:300]}
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                return {"body_system": "general", "rationale": "labeller failed"}
            time.sleep(2 ** attempt)
    return {"body_system": "general", "rationale": ""}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="data/ambient/pairs_v2.parquet")
    ap.add_argument("--out", default="data/ambient/encounter_systems.parquet")
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    pairs = pd.read_parquet(args.pairs)
    enc = pairs.drop_duplicates(["source_dataset", "item_id"])[["source_dataset", "item_id", "clean_transcript", "reference_note"]].reset_index(drop=True)
    enc["item_id"] = enc["item_id"].astype(str)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        labels = list(pool.map(lambda r: label_one(r.clean_transcript, r.reference_note, args.model), enc.itertuples()))
    out = pd.concat([enc[["source_dataset", "item_id"]], pd.DataFrame(labels)], axis=1)
    out["labeller"] = args.model
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"{len(out)} encounters labelled -> {args.out}")
    print(out["body_system"].value_counts().to_string())


if __name__ == "__main__":
    main()
