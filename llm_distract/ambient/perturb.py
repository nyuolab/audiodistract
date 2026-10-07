"""(Re)generate incidental exchanges for a dialogue corpus with an OpenAI model (API; not needed to reproduce the paper).

The published insertions are frozen in data/ambient/insertions.parquet. This script documents and reproduces the
generation call (GPT-5.4, temperature 0, JSON mode, 2-4 turns in the transcript's speaker tags); outputs are
sampled by the API provider and will not be byte-identical to the frozen file.

    python -m llm_distract.ambient.perturb --sources data/ambient/sources --distractor bystander --out new_insertions.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd

from llm_distract.ambient.data import build_cohort, speaker_tags
from llm_distract.ambient.prompts import DISTRACTOR_STYLE_DESCRIPTIONS, DISTRACTOR_SYSTEM, DISTRACTOR_USER_TEMPLATE

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_prompt(transcript: str, distractor_type: str) -> str:
    tags = ", ".join(f"{t}:" for t in speaker_tags(transcript))
    return DISTRACTOR_USER_TEMPLATE.format(distractor_type=distractor_type, style_description=DISTRACTOR_STYLE_DESCRIPTIONS[distractor_type],
                                           speaker_tags=tags, transcript=transcript)


def generate_one(transcript: str, distractor_type: str, model: str, retries: int = 3) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tags = tuple(f"{t}:" for t in speaker_tags(transcript))
    for attempt in range(retries):
        r = client.chat.completions.create(model=model, temperature=0.0, max_completion_tokens=512, response_format={"type": "json_object"},
                                           messages=[{"role": "system", "content": DISTRACTOR_SYSTEM},
                                                     {"role": "user", "content": build_prompt(transcript, distractor_type)}])
        text = r.choices[0].message.content or ""
        try:
            j = json.loads(text) if text.strip().startswith("{") else json.loads(JSON_RE.search(text).group())
            conv = str(j["distractor_conversation"]).strip()
            lines = [ln.strip() for ln in conv.splitlines() if ln.strip()]
            if not conv or not any(ln.startswith(tags) for ln in lines):
                raise ValueError("no source speaker tag used")
            return {"distractor_topic": str(j["distractor_topic"]).strip(), "distractor_conversation": "\n".join(lines),
                    "distractor_summary": str(j["distractor_summary"]).strip(), "insert_after_turn": max(1, int(j.get("insertion_after_turn", 1)))}
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="data/ambient/sources")
    ap.add_argument("--distractor", required=True, choices=["nonliteral", "bystander"])
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cohort = build_cohort(args.sources)
    if args.limit:
        cohort = cohort.head(args.limit)
    out = Path(args.out)
    rows = pd.read_parquet(out).to_dict("records") if out.exists() else []
    done = {(r["source_dataset"], r["item_id"]) for r in rows}
    for _, r in cohort.iterrows():
        if (r["source_dataset"], r["item_id"]) in done:
            continue
        g = generate_one(r["clean_transcript"], args.distractor, args.model)
        rows.append({"source_dataset": r["source_dataset"], "item_id": r["item_id"], "distractor_type": args.distractor, **g,
                     "generation_model": args.model, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"{len(rows)} insertions -> {out}")


if __name__ == "__main__":
    main()
