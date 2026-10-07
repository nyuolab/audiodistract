"""Generate SOAP notes from clean and distracted transcripts with an open-weight model (GPU) or an API model.

The v3 open-weight runs use ``--pairs data/ambient/pairs_v3.parquet --chat-template
--max-new-tokens 1024``: bf16 weights at the pinned revision, native chat templates, left padding, greedy decoding
and batch size 2. These are the default local-generation settings; --no-chat-template selects raw prompting.
``--backend api`` sends the same prompt as a single user message to an OpenAI or Anthropic model
(temperature 0 where the model accepts it, default reasoning/thinking settings, 4,096-token cap by default because
Anthropic counts thinking tokens against ``max_tokens``) through
``meddistractqa.run_eval.call_api``; keys come from OPENAI_API_KEY / ANTHROPIC_API_KEY. Partial API results are
checkpointed next to the output so an interrupted run resumes.

    python -m llm_distract.ambient.generate_notes --model meta-llama/Llama-3.1-8B-Instruct \
        --pairs data/ambient/pairs_v3.parquet --chat-template --max-new-tokens 1024 \
        --split mts_dialog_test1 --distractor bystander --out outputs/ambient_v3
    python -m llm_distract.ambient.generate_notes --backend api --provider openai --model gpt-5.4-2026-03-05 \
        --pairs data/ambient/pairs_v3.parquet --max-new-tokens 1024 \
        --split aci_bench_test --distractor bystander --out outputs/ambient_v3_frontier
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml

from llm_distract.ambient.prompts import build_note_prompt


def load_config(path: str = "configs/default.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_model(model: str, revision: str | None, attn_implementation: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto", "revision": revision}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    mdl = AutoModelForCausalLM.from_pretrained(model, **kwargs).eval()
    return tok, mdl


def generate(tok, mdl, prompts: list, max_new_tokens: int, batch_size: int, chat_template: bool) -> list:
    import torch

    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        if chat_template:
            batch = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in batch]
        # the rendered template already carries its special tokens (BOS for Gemma/Llama/Mistral): never add them twice;
        # the earlier raw-text protocol adds none either
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(mdl.device)
        with torch.no_grad():
            gen = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outputs.append(tok.decode(row, skip_special_tokens=True).strip())
    return outputs


def generate_api(provider: str, model: str, keys: list, prompts: list, max_tokens: int, temperature: float, concurrency: int,
                 checkpoint: Path, reasoning_effort: str | None = None, thinking: str = "default", attempts: int = 6) -> pd.DataFrame:
    """One completion per prompt through ``call_api`` with ``concurrency`` threads and exponential back-off on errors.

    ``keys`` are (item_id, condition) pairs used to checkpoint finished notes in ``checkpoint`` (jsonl) and skip them
    on a rerun. Returns a DataFrame aligned with ``prompts``: note, input_tokens, output_tokens, reasoning_tokens,
    finish_reason, api_error (empty string unless every attempt failed)."""
    import random
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from llm_distract.meddistractqa.run_eval import call_api

    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[(str(rec["item_id"]), rec["condition"])] = rec

    def one(idx: int) -> dict:
        item_id, cond = keys[idx]
        if (str(item_id), cond) in done:
            return {"idx": idx, **done[(str(item_id), cond)]}
        last = ""
        for attempt in range(attempts):
            try:
                text, usage = call_api(provider, model, prompts[idx], temperature, max_tokens, False, reasoning_effort, thinking)
                return {"idx": idx, "item_id": str(item_id), "condition": cond, "note": (text or "").strip(),
                        "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                        "reasoning_tokens": usage.get("reasoning_tokens"), "finish_reason": usage.get("finish_reason"), "api_error": ""}
            except Exception as exc:  # noqa: BLE001 - rate limits, overloads, transient network errors
                last = f"{type(exc).__name__}: {str(exc)[:200]}"
                time.sleep(min(60, 2 ** attempt) + random.random())
        return {"idx": idx, "item_id": str(item_id), "condition": cond, "note": "", "input_tokens": None, "output_tokens": None,
                "reasoning_tokens": None, "finish_reason": "error", "api_error": last}

    results = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool, checkpoint.open("a") as ck:
        futures = [pool.submit(one, i) for i in range(len(prompts))]
        for k, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            results[rec["idx"]] = rec
            if (str(rec["item_id"]), rec["condition"]) not in done and rec["finish_reason"] != "error":
                ck.write(json.dumps({key: val for key, val in rec.items() if key != "idx"}) + "\n"); ck.flush()
            if k % 50 == 0 or k == len(prompts):
                print(f"  {k}/{len(prompts)} notes", flush=True)
    rows = [results[i] for i in range(len(prompts))]
    return pd.DataFrame(rows).drop(columns=["idx", "item_id", "condition"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", default="data/ambient/pairs_v3.parquet", help="output of llm_distract.ambient.data --build")
    ap.add_argument("--split", required=True, help="source_dataset value, e.g. mts_dialog_test1")
    ap.add_argument("--distractor", required=True, choices=["nonliteral", "bystander"])
    ap.add_argument("--out", default="outputs/ambient_v3")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="note length cap; defaults come from configs/default.yaml (use 1024 for the v3 open-weight and GPT runs, 4096 for Claude)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True, help="use the native chat template; --no-chat-template enables legacy raw prompts")
    ap.add_argument("--attn-implementation", default=None, choices=[None, "eager", "sdpa"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", default="hf", choices=["hf", "api"], help="hf: transformers on GPU (paper); api: OpenAI/Anthropic model")
    ap.add_argument("--provider", default=None, choices=["openai", "anthropic"], help="api backend: which API")
    ap.add_argument("--temperature", type=float, default=0.0, help="api backend: sampling temperature where the model accepts one")
    ap.add_argument("--reasoning-effort", default=None, help="api backend: OpenAI reasoning effort (default: model default)")
    ap.add_argument("--thinking", default="default", choices=["default", "disabled", "adaptive"], help="api backend: Anthropic thinking mode")
    ap.add_argument("--concurrency", type=int, default=8, help="api backend: parallel requests")
    ap.add_argument("--batch", action="store_true", help="api backend: submit through the provider's batch API (half price, up to 24 h)")
    args = ap.parse_args()
    if args.backend == "api" and not args.provider:
        ap.error("--backend api needs --provider")

    cfg = load_config(args.config)
    revision = cfg["mechanism"]["revisions"].get(args.model)
    pairs = pd.read_parquet(args.pairs)
    pairs = pairs[(pairs["source_dataset"] == args.split) & (pairs["distractor_type"] == args.distractor)]
    if args.limit:
        pairs = pairs.head(args.limit)
    rows = []
    for _, r in pairs.iterrows():
        for cond, col in (("clean", "clean_transcript"), ("distracted", "distracted_transcript")):
            rows.append({"source_dataset": r["source_dataset"], "family": r["family"], "item_id": r["item_id"],
                         "distractor_type": r["distractor_type"], "condition": cond, "transcript": r[col],
                         "reference_note": r["reference_note"], "distractor_summary": r["distractor_summary"]})
    df = pd.DataFrame(rows)
    prompts = [build_note_prompt(t) for t in df["transcript"]]
    out_dir = Path(args.out) / args.model.replace("/", "__") / args.split / args.distractor
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.backend == "api":
        max_new = args.max_new_tokens or cfg["ambient"].get("api_note_max_new_tokens", 4096)  # Anthropic counts thinking tokens against max_tokens
        if args.batch:
            from llm_distract.ambient.batch import run_batch

            reqs = [{"custom_id": f"{iid}|{cond}", "model": args.model, "user": pr, "max_tokens": max_new, "temperature": args.temperature,
                     "reasoning_effort": args.reasoning_effort, "thinking": args.thinking} for (iid, cond), pr in zip(zip(df["item_id"].astype(str), df["condition"]), prompts)]
            res = run_batch(args.provider, reqs, out_dir / "batch_state.json", tag=f"notes-{args.model}-{args.split}-{args.distractor}")
            recs = [res.get(r["custom_id"], ("", {}, "error")) for r in reqs]
            api = pd.DataFrame({"note": [t.strip() for t, _, _ in recs], "input_tokens": [u.get("input_tokens") for _, u, _ in recs],
                                "output_tokens": [u.get("output_tokens") for _, u, _ in recs], "reasoning_tokens": [u.get("reasoning_tokens") for _, u, _ in recs],
                                "finish_reason": [f for _, _, f in recs], "api_error": ["batch request failed" if f == "error" else "" for _, _, f in recs]})
        else:
            api = generate_api(args.provider, args.model, list(zip(df["item_id"], df["condition"])), prompts, max_new, args.temperature,
                               args.concurrency, out_dir / "notes_checkpoint.jsonl", args.reasoning_effort, args.thinking)
        for col in api.columns:
            df[col] = api[col].values
        usage = {"input_tokens": int(pd.to_numeric(df["input_tokens"], errors="coerce").fillna(0).sum()),
                 "output_tokens": int(pd.to_numeric(df["output_tokens"], errors="coerce").fillna(0).sum()),
                 "reasoning_tokens": int(pd.to_numeric(df["reasoning_tokens"], errors="coerce").fillna(0).sum())}
        manifest = {"model": args.model, "backend": "api", "provider": args.provider, "temperature": args.temperature, "batch": bool(args.batch),
                    "reasoning_effort": args.reasoning_effort, "thinking": args.thinking, "max_new_tokens": max_new,
                    "concurrency": args.concurrency, "n_notes": len(df), "n_error": int((df["finish_reason"] == "error").sum()),
                    "n_refusal": int((df["finish_reason"] == "refusal").sum()), "n_truncated": int(df["finish_reason"].isin(["length", "max_tokens"]).sum()),
                    "usage": usage}
    else:
        max_new = args.max_new_tokens or 1024
        tok, mdl = load_model(args.model, revision, args.attn_implementation)
        df["note"] = generate(tok, mdl, prompts, max_new, cfg["ambient"]["note_batch_size"], args.chat_template)
        manifest = {"model": args.model, "revision": revision, "chat_template": args.chat_template, "dtype": "bfloat16",
                    "max_new_tokens": max_new, "batch_size": cfg["ambient"]["note_batch_size"], "decoding": "greedy", "n_notes": len(df)}
    df["model"] = args.model
    df.to_parquet(out_dir / "notes.parquet", index=False)
    manifest.update({"seconds": round(time.time() - t0, 1), "code": git_commit(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(df)} notes to {out_dir}" + (f" (errors {manifest['n_error']}, refusals {manifest['n_refusal']}, truncated {manifest['n_truncated']})" if args.backend == "api" else ""))


if __name__ == "__main__":
    main()
