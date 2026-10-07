"""Evaluate one engine on MedDistractQA (clean / nonliteral / bystander) and write per-item results.

Backends: openai (chat completions; GPT-5 family via the same endpoint), anthropic, google (Gemini), deepseek
(OpenAI-compatible endpoint), vllm (open-weight, raw prompt without chat template as in the paper; ``--chat-template``
to apply it) and hf (transformers with chat template, used for MedMobile). Temperature 0 where the provider allows it:
OpenAI reasoning models ignore temperature, Gemini 3 is kept at its default of 1.0 unless ``--force-temperature``.

    python -m llm_distract.meddistractqa.run_eval --engine gpt-4o-mini --backend openai --condition nonliteral --out outputs/qa
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from llm_distract.meddistractqa.data import load_meddistractqa
from llm_distract.meddistractqa.parse import extract_answer, is_correct, is_invalid
from llm_distract.meddistractqa.prompts import build_prompt

QUESTION_COL = {"clean": "clean_question", "nonliteral": "nonliteral_question", "bystander": "bystander_question"}
# Anthropic models on which the sampling controls (temperature/top_p/top_k) are removed and rejected with a 400
ANTHROPIC_NO_SAMPLING = ("claude-opus-5", "claude-sonnet-5", "claude-fable", "claude-mythos", "claude-opus-4-7", "claude-opus-4-8")


def call_api(backend: str, engine: str, prompt: str, temperature: float, max_tokens: int, force_temperature: bool,
             reasoning_effort: str | None = None, thinking: str = "default") -> tuple:
    """One completion; returns (text, usage dict). ``reasoning_effort`` is passed to OpenAI reasoning models when set;
    ``thinking`` selects the Anthropic thinking mode: ``default`` omits the parameter (model default), ``disabled`` or
    ``adaptive`` pass ``{"type": ...}`` (Claude Fable rejects ``disabled``)."""
    usage: dict = {}
    if backend in ("openai", "deepseek"):
        from openai import OpenAI

        if backend == "deepseek":
            client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
        else:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        kwargs = {"model": engine, "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": max_tokens}
        if not engine.startswith(("o1", "o3", "o4", "gpt-5")):
            kwargs["temperature"] = temperature
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        r = client.chat.completions.create(**kwargs)
        if r.usage:
            details = getattr(r.usage, "completion_tokens_details", None)
            usage = {"input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens,
                     "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
                     "finish_reason": r.choices[0].finish_reason}
        return r.choices[0].message.content or "", usage
    if backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        kwargs = {"model": engine, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if force_temperature or not engine.startswith(ANTHROPIC_NO_SAMPLING):
            kwargs["temperature"] = temperature
        if thinking in ("disabled", "adaptive"):
            kwargs["thinking"] = {"type": thinking}
        for attempt in range(3):  # safety-classifier refusals are not fully deterministic: retry twice, then keep the refusal
            r = client.messages.create(**kwargs)
            if r.stop_reason != "refusal":
                break
        usage = {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens, "finish_reason": r.stop_reason}
        return "".join(getattr(b, "text", "") for b in r.content), usage
    if backend == "google":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"])
        cfg = types.GenerateContentConfig(max_output_tokens=max_tokens)
        if force_temperature or not engine.startswith("gemini-3"):
            cfg.temperature = temperature
        r = client.models.generate_content(model=engine, contents=prompt, config=cfg)
        um = getattr(r, "usage_metadata", None)
        if um:
            usage = {"input_tokens": um.prompt_token_count, "output_tokens": um.candidates_token_count}
        return r.text or "", usage
    raise ValueError(backend)


def run_local(backend: str, engine: str, prompts: list, temperature: float, max_tokens: int, chat_template: bool, revision: str | None,
              tensor_parallel: int = 1, max_model_len: int | None = None, gpu_memory_utilization: float = 0.9,
              batch_size: int = 4) -> list:
    if backend == "vllm":
        from vllm import LLM, SamplingParams

        kwargs = {"model": engine, "revision": revision, "tensor_parallel_size": tensor_parallel,
                  "gpu_memory_utilization": gpu_memory_utilization}
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        llm = LLM(**kwargs)
        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        if chat_template:
            # Let vLLM apply the tokenizer's native template and tokenize it in one
            # operation. Formatting to text and then calling generate() can add
            # special tokens a second time (notably BOS for Gemma).
            conversations = [[{"role": "user", "content": p}] for p in prompts]
            outs = llm.chat(conversations, sampling)
        else:
            outs = llm.generate(prompts, sampling)
        return [o.outputs[0].text for o in outs]
    if backend == "hf":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(engine, revision=revision)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(engine, torch_dtype=torch.bfloat16, device_map="auto", revision=revision).eval()
        outs = []
        for start in range(0, len(prompts), batch_size):
            texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                     for p in prompts[start:start + batch_size]]
            # The rendered template already contains all required special tokens.
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(mdl.device)
            with torch.no_grad():
                g = mdl.generate(**enc, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            prompt_len = enc["input_ids"].shape[1]
            outs.extend(tok.decode(row[prompt_len:], skip_special_tokens=True) for row in g)
        return outs
    raise ValueError(backend)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help="provider model id or HF model id")
    ap.add_argument("--backend", required=True, choices=["openai", "anthropic", "google", "deepseek", "vllm", "hf"])
    ap.add_argument("--condition", required=True, choices=list(QUESTION_COL))
    ap.add_argument("--prompt", default="standard", choices=["standard", "ignore", "structured"])
    ap.add_argument("--data", default="data/meddistractqa/meddistractqa_v2.parquet")
    ap.add_argument("--out", default="outputs/qa")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chat-template", action="store_true", help="vllm: apply the tokenizer's native chat template")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--force-temperature", action="store_true")
    ap.add_argument("--reasoning-effort", default=None, help="openai: reasoning_effort for reasoning models (e.g. none, low, medium, high)")
    ap.add_argument("--thinking", default="default", choices=["default", "disabled", "adaptive"], help="anthropic: thinking mode")
    ap.add_argument("--name", default=None, help="engine label for the output directory and rows (default: --engine); e.g. o3-mini-high for o3-mini at high effort")
    ap.add_argument("--tensor-parallel", type=int, default=1, help="vllm: tensor parallel size (GPUs)")
    ap.add_argument("--max-model-len", type=int, default=None, help="vllm: explicit context length (helps avoid oversized KV-cache reservations)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vllm: fraction of GPU memory available to the engine")
    ap.add_argument("--batch-size", type=int, default=4, help="hf: generation batch size")
    args = ap.parse_args()
    name = args.name or args.engine

    df = load_meddistractqa(args.data)
    if args.limit:
        df = df.head(args.limit)
    prompts = [build_prompt(r[QUESTION_COL[args.condition]], {k: r[f"choice_{k}"] for k in "ABCD"}, args.prompt) for _, r in df.iterrows()]
    t0 = time.time()
    if args.backend in ("vllm", "hf"):
        responses = [(t, {}) for t in run_local(args.backend, args.engine, prompts, args.temperature, args.max_tokens, args.chat_template, args.revision,
                                                args.tensor_parallel, args.max_model_len, args.gpu_memory_utilization, args.batch_size)]
    else:
        def one(p):
            for attempt in range(3):
                try:
                    return call_api(args.backend, args.engine, p, args.temperature, args.max_tokens, args.force_temperature,
                                    args.reasoning_effort, args.thinking)
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        return f"[error] {exc}", {}
                    time.sleep(2 ** attempt)
        with ThreadPoolExecutor(args.workers) as ex:
            responses = list(ex.map(one, prompts))
    rows = []
    for (_, r), p, (resp, usage) in zip(df.iterrows(), prompts, responses):
        parsed = extract_answer(resp)
        rows.append({"item_id": int(r["item_id"]), "condition": args.condition, "prompt_variant": args.prompt, "engine": name,
                     "response": resp, "parsed_answer": parsed, "correct_answer": r["correct_answer"],
                     "is_correct": is_correct(parsed, r["correct_answer"]), "is_invalid": is_invalid(parsed), **usage})
    out_dir = Path(args.out) / name.replace("/", "__") / args.prompt
    out_dir.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_json(out_dir / f"{args.condition}.jsonl", orient="records", lines=True)
    manifest = {"engine": name, "api_model_id": args.engine, "backend": args.backend, "condition": args.condition, "prompt_variant": args.prompt,
                "temperature": args.temperature, "max_tokens": args.max_tokens, "chat_template": args.chat_template,
                "reasoning_effort": args.reasoning_effort, "thinking": args.thinking,
                "revision": args.revision, "max_model_len": args.max_model_len, "gpu_memory_utilization": args.gpu_memory_utilization,
                "batch_size": args.batch_size, "n": len(res), "accuracy": float(res["is_correct"].mean()), "n_invalid": int(res["is_invalid"].sum()),
                "mean_input_tokens": float(res["input_tokens"].mean()) if "input_tokens" in res else None,
                "mean_output_tokens": float(res["output_tokens"].mean()) if "output_tokens" in res else None,
                "n_refusal": int((res["finish_reason"] == "refusal").sum()) if "finish_reason" in res else 0,
                "n_error": int(res["response"].astype(str).str.startswith("[error]").sum()),
                "seconds": round(time.time() - t0, 1), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / f"{args.condition}.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
