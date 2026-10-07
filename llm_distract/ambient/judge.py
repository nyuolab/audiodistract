"""LLM judges for generated notes (API; resumable; records the judge model and protocol).

Modes
  contamination_paired  primary endpoint: both notes of an encounter in randomized order, target = insertion summary
                        (Anthropic, default claude-sonnet-5; protocol ``v3_paired_symmetric``)
  quality               secondary single-note rubric (OpenAI, default gpt-5.4)
  contamination_single  earlier single-note contamination rubric (OpenAI, default gpt-5.4; protocol ``v2_single_note``),
                        kept for the judge-sensitivity analysis

    python -m llm_distract.ambient.judge --notes outputs/ambient/<model>/<split>/<distractor>/notes.parquet \
        --mode contamination_paired --provider anthropic --model claude-sonnet-5 --pairs data/ambient/pairs.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import pandas as pd

from llm_distract.ambient.prompts import (ATTRIBUTION_JUDGE_SYSTEM, ATTRIBUTION_JUDGE_TEMPLATE, PAIRED_JUDGE_SYSTEM, PAIRED_JUDGE_TEMPLATE, QUALITY_JUDGE_SYSTEM, QUALITY_JUDGE_TEMPLATE,
                                          SINGLE_NOTE_CLEAN_CONTROL_TEMPLATE, SINGLE_NOTE_CONTAMINATION_TEMPLATE)

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
SINGLE_JUDGE_SYSTEM = ("You are a clinical documentation auditor. Score the generated progress note strictly by the rubric "
                       "and respond with a single JSON object.")


def _call(provider: str, model: str, system: str, user: str, max_tokens: int = 800, retries: int = 4) -> str:
    """One chat completion; OpenAI at temperature 0, Anthropic at the API default (as in the published runs)."""
    for attempt in range(retries):
        try:
            if provider == "openai":
                from openai import OpenAI

                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                r = client.chat.completions.create(model=model, temperature=0.0, max_completion_tokens=max_tokens,
                                                   response_format={"type": "json_object"},
                                                   messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                return r.choices[0].message.content or ""
            if provider == "anthropic":
                import anthropic

                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                r = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                           messages=[{"role": "user", "content": user}])
                return "".join(getattr(b, "text", "") for b in r.content)
            raise ValueError(provider)
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return ""


BATCH_STATE = {"path": None}  # set by main() when --batch is given: directory for batch state files


def _dispatch(provider: str, model: str, jobs: list, concurrency: int, tag: str) -> list:
    """Run (system, user, max_tokens) jobs live (``_call``, ``concurrency`` at a time) or, when a batch state directory
    is set, through the provider's batch API; returns the raw replies in order."""
    from functools import partial

    if BATCH_STATE["path"] is None:
        return _concurrent([partial(_call, provider, model, sy, us, max_tokens=mt) for sy, us, mt in jobs], concurrency)
    from llm_distract.ambient.batch import run_batch

    reqs = [{"custom_id": str(i), "model": model, "system": sy, "user": us, "max_tokens": mt, "temperature": 0.0, "json_mode": provider == "openai"}
            for i, (sy, us, mt) in enumerate(jobs)]
    res = run_batch(provider, reqs, Path(BATCH_STATE["path"]) / f"batch_{tag}.json", tag=tag)
    return [res.get(str(i), ("", {}, "error"))[0] for i in range(len(jobs))]


def _concurrent(calls: list, concurrency: int) -> list:
    """Run zero-argument callables ``concurrency`` at a time and return their results in order (1 = sequential)."""
    if concurrency <= 1:
        return [call() for call in calls]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda call: call(), calls))


def _parse(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = JSON_RE.search(text)
        if not m:
            raise
        return json.loads(m.group())


def _order(key: str, seed: int) -> bool:
    """Deterministic A/B order: True -> clean note shown as Note A."""
    return int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest(), 16) % 2 == 0


def judge_paired(notes: pd.DataFrame, pairs: pd.DataFrame, provider: str, model: str, seed: int, out: Path,
                 concurrency: int = 1) -> pd.DataFrame:
    """Paired judge; ``concurrency`` > 1 sends that many encounters at once (same prompts and order of results)."""
    from functools import partial

    done = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    done_keys = set(zip(done["source_dataset"], done["item_id"].astype(str), done["distractor_type"])) if len(done) else set()
    meta = pairs.set_index(["source_dataset", "item_id", "distractor_type"])
    rows = done.to_dict("records")
    keys = ["source_dataset", "item_id", "distractor_type"]
    tasks = []
    for (sd, iid, dt), g in notes.groupby(keys):
        if (sd, str(iid), dt) in done_keys:
            continue
        clean = g[g["condition"] == "clean"].iloc[0]
        dist = g[g["condition"] == "distracted"].iloc[0]
        m = meta.loc[(sd, str(iid), dt)]
        clean_first = _order(f"{sd}|{iid}|{dt}|{clean['model']}", seed)
        a, b = (clean, dist) if clean_first else (dist, clean)
        user = PAIRED_JUDGE_TEMPLATE.format(clean_transcript=m["clean_transcript"], distractor_summary=m["distractor_summary"],
                                            note_a=a["note"], note_b=b["note"])
        tasks.append((sd, iid, dt, clean["model"], clean_first, user))
    chunk = len(tasks) if BATCH_STATE["path"] else (1 if concurrency <= 1 else max(20, 5 * concurrency))
    for start in range(0, len(tasks), max(1, chunk)):
        batch = tasks[start:start + max(1, chunk)]
        raws = _dispatch(provider, model, [(PAIRED_JUDGE_SYSTEM, t[5], 800) for t in batch], concurrency, "paired")
        for (sd, iid, dt, model_name, clean_first, user), raw in zip(batch, raws):
            try:
                j = _parse(raw)
                scores = {"a": (int(j["note_a_contamination"]), int(j["note_a_severity"])), "b": (int(j["note_b_contamination"]), int(j["note_b_severity"]))}
                assert all(c in (0, 1) and 0 <= s <= 3 for c, s in scores.values())
                err = False
            except Exception:  # noqa: BLE001
                scores, err, j = {"a": (None, None), "b": (None, None)}, True, {}
            for cond, slot in (("clean", "a" if clean_first else "b"), ("distracted", "b" if clean_first else "a")):
                rows.append({"source_dataset": sd, "item_id": str(iid), "distractor_type": dt, "model": model_name, "condition": cond,
                             "contamination": scores[slot][0], "severity": scores[slot][1], "shown_as": slot.upper(),
                             "judge_reasoning": j.get("reasoning", ""), "judge_raw": raw, "parse_error": err,
                             "judge_model": model, "judge_provider": provider, "judge_protocol": "v3_paired_symmetric", "seed": seed,
                             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        pd.DataFrame(rows).to_parquet(out, index=False)  # checkpoint after every encounter (sequential) or chunk
    if not tasks and rows:
        pd.DataFrame(rows).to_parquet(out, index=False)
    return pd.DataFrame(rows)


def judge_quality(notes: pd.DataFrame, provider: str, model: str, out: Path, concurrency: int = 1) -> pd.DataFrame:
    from functools import partial

    done = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    done_keys = set(zip(done["source_dataset"], done["item_id"].astype(str), done["distractor_type"], done["condition"])) if len(done) else set()
    rows = done.to_dict("records")
    todo = [r for _, r in notes.iterrows() if (r["source_dataset"], str(r["item_id"]), r["distractor_type"], r["condition"]) not in done_keys]
    chunk = len(todo) if BATCH_STATE["path"] else (20 if concurrency <= 1 else max(20, 5 * concurrency))
    for start in range(0, len(todo), max(1, chunk)):
        batch = todo[start:start + max(1, chunk)]
        users = [QUALITY_JUDGE_TEMPLATE.format(transcript=r["transcript"], note=r["note"], reference=r.get("reference_note") or "No reference note provided.")
                 for r in batch]
        raws = _dispatch(provider, model, [(QUALITY_JUDGE_SYSTEM, u, 512) for u in users], concurrency, "quality")
        for r, raw in zip(batch, raws):
            try:
                j = _parse(raw)
                vals = {k: float(j[k]) for k in ("clinical_correctness", "completeness", "succinctness", "hallucination", "overall_quality")}
                assert all(1 <= vals[k] <= 5 for k in vals if k != "hallucination") and vals["hallucination"] in (0.0, 1.0)
                err = False
            except Exception:  # noqa: BLE001
                vals, err, j = {}, True, {}
            rows.append({**{k: r[k] for k in ("source_dataset", "item_id", "distractor_type", "condition", "model")}, **vals,
                         "judge_reasoning": j.get("reasoning", ""), "parse_error": err, "judge_model": model, "judge_provider": provider,
                         "judge_protocol": "single_note_quality", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        pd.DataFrame(rows).to_parquet(out, index=False)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return pd.DataFrame(rows)


def judge_single(notes: pd.DataFrame, provider: str, model: str, out: Path, concurrency: int = 1) -> pd.DataFrame:
    """Legacy single-note contamination rubric: distracted notes are scored against the insertion summary, clean notes
    with the control rubric (no target). ``concurrency`` > 1 sends that many notes at once (same prompts and order)."""
    from functools import partial

    done = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    done_keys = set(zip(done["source_dataset"], done["item_id"].astype(str), done["distractor_type"], done["condition"])) if len(done) else set()
    rows = done.to_dict("records")
    todo = [r for _, r in notes.iterrows() if (r["source_dataset"], str(r["item_id"]), r["distractor_type"], r["condition"]) not in done_keys]
    chunk = len(todo) if BATCH_STATE["path"] else (20 if concurrency <= 1 else max(20, 5 * concurrency))
    for start in range(0, len(todo), max(1, chunk)):
        batch = todo[start:start + max(1, chunk)]
        users = [SINGLE_NOTE_CONTAMINATION_TEMPLATE.format(transcript=r["transcript"], note=r["note"], distractor_summary=r["distractor_summary"])
                 if r["condition"] == "distracted" else SINGLE_NOTE_CLEAN_CONTROL_TEMPLATE.format(transcript=r["transcript"], note=r["note"])
                 for r in batch]
        raws = _dispatch(provider, model, [(SINGLE_JUDGE_SYSTEM, u, 400) for u in users], concurrency, "single")
        for r, raw in zip(batch, raws):
            try:
                j = _parse(raw)
                c, sev = int(j["distractor_contamination"]), int(j["contamination_severity"])
                assert c in (0, 1) and 0 <= sev <= 3
                err = False
            except Exception:  # noqa: BLE001
                c, sev, err, j = None, None, True, {}
            rows.append({**{k: r[k] for k in ("source_dataset", "item_id", "distractor_type", "condition", "model")}, "item_id": str(r["item_id"]),
                         "contamination": c, "severity": sev, "judge_reasoning": j.get("reasoning", ""), "judge_raw": raw, "parse_error": err,
                         "judge_model": model, "judge_provider": provider, "judge_protocol": "v2_single_note",
                         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        pd.DataFrame(rows).to_parquet(out, index=False)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return pd.DataFrame(rows)


def judge_attribution(notes: pd.DataFrame, paired: pd.DataFrame, pairs: pd.DataFrame, provider: str, model: str, out: Path,
                      concurrency: int = 1) -> pd.DataFrame:
    """Attribution sub-judge on the distracted notes that the paired judge scored as contaminated: does the note give the
    inserted content to the patient or to the third party it concerns? One note per call; resumable."""
    from functools import partial

    keys = ["source_dataset", "item_id", "distractor_type"]
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    if len(done):
        done = done[~done["parse_error"].astype(bool)]  # rows that failed to parse are judged again
    done_keys = set(zip(done["source_dataset"], done["item_id"].astype(str), done["distractor_type"])) if len(done) else set()
    rows = done.to_dict("records")
    meta = pairs.set_index(keys)
    hit = paired[(paired["condition"] == "distracted") & (paired["contamination"] == 1)]
    hit_keys = set(zip(hit["source_dataset"], hit["item_id"].astype(str), hit["distractor_type"]))
    todo = [r for _, r in notes.iterrows() if r["condition"] == "distracted" and (r["source_dataset"], str(r["item_id"]), r["distractor_type"]) in hit_keys
            and (r["source_dataset"], str(r["item_id"]), r["distractor_type"]) not in done_keys]
    chunk = len(todo) if BATCH_STATE["path"] else (20 if concurrency <= 1 else max(20, 5 * concurrency))
    for start in range(0, len(todo), max(1, chunk)):
        batch = todo[start:start + max(1, chunk)]
        users = [ATTRIBUTION_JUDGE_TEMPLATE.format(clean_transcript=meta.loc[(r["source_dataset"], str(r["item_id"]), r["distractor_type"])]["clean_transcript"],
                                                   distractor_summary=r["distractor_summary"], note=r["note"]) for r in batch]
        raws = _dispatch(provider, model, [(ATTRIBUTION_JUDGE_SYSTEM, u, 600) for u in users], concurrency, "attribution")
        raws = [raw if raw.strip() else _call(provider, model, ATTRIBUTION_JUDGE_SYSTEM, u, max_tokens=600) for raw, u in zip(raws, users)]  # empty reply: one live retry
        for r, raw in zip(batch, raws):
            try:
                j = _parse(raw)
                attribution = str(j["attribution"]).strip().lower(); used = int(j["used_for_patient"])
                attribution = "correct" if attribution == "third_party" else attribution  # pilot label
                assert attribution in ("patient", "correct", "absent") and used in (0, 1)
                err = False
            except Exception:  # noqa: BLE001
                attribution, used, err, j = None, None, True, {}
            rows.append({**{k: r[k] for k in keys + ["condition", "model"]}, "item_id": str(r["item_id"]), "attribution": attribution,
                         "used_for_patient": used, "judge_reasoning": j.get("reasoning", ""), "judge_raw": raw, "parse_error": err,
                         "judge_model": model, "judge_provider": provider, "judge_protocol": "attribution_v1", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        pd.DataFrame(rows).to_parquet(out, index=False)
    cols = keys + ["condition", "model", "attribution", "used_for_patient", "judge_reasoning", "judge_raw", "parse_error", "judge_model",
                   "judge_provider", "judge_protocol", "timestamp"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    df.to_parquet(out, index=False)  # an empty file records that no distracted note of this set was scored as contaminated
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notes", required=True)
    ap.add_argument("--mode", required=True, choices=["contamination_paired", "quality", "contamination_single", "attribution"])
    ap.add_argument("--provider", default=None, choices=["anthropic", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--pairs", default="data/ambient/pairs_v3.parquet", help="needed for contamination_paired (clean transcript + target)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: judge_<mode>.parquet next to the notes")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel judge requests (1 = sequential, as in the published runs)")
    ap.add_argument("--batch", action="store_true", help="submit through the provider's batch API (state files next to the notes)")
    args = ap.parse_args()
    if args.batch:
        BATCH_STATE["path"] = str(Path(args.notes).parent)
    notes = pd.read_parquet(args.notes)
    out = Path(args.out) if args.out else Path(args.notes).with_name(f"judge_{args.mode}.parquet")
    if args.mode == "contamination_paired":
        provider, model = args.provider or "anthropic", args.model or "claude-sonnet-5"
        pairs = pd.read_parquet(args.pairs)
        pairs["item_id"] = pairs["item_id"].astype(str)
        df = judge_paired(notes, pairs, provider, model, args.seed, out, args.concurrency)
    elif args.mode == "quality":
        provider, model = args.provider or "openai", args.model or "gpt-5.4"
        df = judge_quality(notes, provider, model, out, args.concurrency)
    elif args.mode == "attribution":
        provider, model = args.provider or "anthropic", args.model or "claude-sonnet-5"
        paired = pd.read_parquet(Path(args.notes).with_name("judge_contamination_paired.parquet"))
        pairs = pd.read_parquet(args.pairs)
        pairs["item_id"] = pairs["item_id"].astype(str)
        df = judge_attribution(notes, paired, pairs, provider, model, out, args.concurrency)
    else:
        provider, model = args.provider or "openai", args.model or "gpt-5.4"
        df = judge_single(notes, provider, model, out, args.concurrency)
    print(f"{len(df)} judgments -> {out}; parse errors: {int(df['parse_error'].astype(bool).sum()) if len(df) else 0}")


if __name__ == "__main__":
    main()
