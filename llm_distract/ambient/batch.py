"""Batch submission of chat completions to OpenAI (Batch API) and Anthropic (Message Batches), used by the note generator
and the judges when ``--batch`` is given. Both services price batched requests at half the live rate and complete
within 24 hours (usually much sooner).

A request is a dict with ``custom_id``, ``model``, ``system`` (optional), ``user``, ``max_tokens`` and optional
``temperature``, ``reasoning_effort`` (OpenAI), ``thinking`` (Anthropic) and ``json_mode`` (OpenAI). ``run_batch``
submits, polls until every request has finished and returns ``{custom_id: (text, usage, finish_reason)}``; a failed
request maps to ``("", {}, "error")``. The batch id and the raw results are written next to ``state_path`` so that an
interrupted run can resume by re-attaching to the same batch.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

CHUNK = 5000  # requests per submitted batch (both services allow more, but smaller batches finish sooner)


def _state(state_path: Path) -> dict:
    return json.loads(state_path.read_text()) if state_path.exists() else {}


def _save(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1))


# ------------------------------------------------------------------ OpenAI
def _openai_submit(client, requests: list, tag: str) -> str:
    import io

    lines = []
    for r in requests:
        body = {"model": r["model"], "messages": ([{"role": "system", "content": r["system"]}] if r.get("system") else []) + [{"role": "user", "content": r["user"]}],
                "max_completion_tokens": r["max_tokens"]}
        if not r["model"].startswith(("o1", "o3", "o4", "gpt-5")) and r.get("temperature") is not None:
            body["temperature"] = r["temperature"]
        if r.get("reasoning_effort"):
            body["reasoning_effort"] = r["reasoning_effort"]
        if r.get("json_mode"):
            body["response_format"] = {"type": "json_object"}
        lines.append(json.dumps({"custom_id": r["custom_id"], "method": "POST", "url": "/v1/chat/completions", "body": body}))
    f = client.files.create(file=(f"{tag}.jsonl", io.BytesIO("\n".join(lines).encode("utf-8"))), purpose="batch")
    b = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h", metadata={"tag": tag})
    return b.id


def _openai_collect(client, batch_id: str) -> dict | None:
    b = client.batches.retrieve(batch_id)
    if b.status in ("validating", "in_progress", "finalizing"):
        return None
    out = {}
    if b.output_file_id:
        for line in client.files.content(b.output_file_id).text.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            resp = rec.get("response") or {}
            body = resp.get("body") or {}
            if resp.get("status_code") == 200 and body.get("choices"):
                ch = body["choices"][0]
                u = body.get("usage") or {}
                det = u.get("completion_tokens_details") or {}
                out[rec["custom_id"]] = (ch["message"].get("content") or "", {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
                                                                           "reasoning_tokens": det.get("reasoning_tokens"), "finish_reason": ch.get("finish_reason")}, ch.get("finish_reason"))
            else:
                out[rec["custom_id"]] = ("", {}, "error")
    if b.error_file_id:
        for line in client.files.content(b.error_file_id).text.splitlines():
            if line.strip():
                out.setdefault(json.loads(line)["custom_id"], ("", {}, "error"))
    return out


# ------------------------------------------------------------------ Anthropic
def _anthropic_submit(client, requests: list, tag: str) -> str:
    from llm_distract.meddistractqa.run_eval import ANTHROPIC_NO_SAMPLING

    reqs = []
    for i, r in enumerate(requests):
        params = {"model": r["model"], "max_tokens": r["max_tokens"], "messages": [{"role": "user", "content": r["user"]}]}
        if r.get("system"):
            params["system"] = r["system"]
        if r.get("temperature") is not None and not r["model"].startswith(ANTHROPIC_NO_SAMPLING):
            params["temperature"] = r["temperature"]
        if r.get("thinking") in ("disabled", "adaptive"):
            params["thinking"] = {"type": r["thinking"]}
        reqs.append({"custom_id": _safe_id(r["custom_id"]), "params": params})  # Anthropic allows only [A-Za-z0-9_-]{1,64}
    b = client.messages.batches.create(requests=reqs)
    return b.id


def _safe_id(custom_id: str) -> str:
    """Anthropic custom ids must match ^[a-zA-Z0-9_-]{1,64}$; encode arbitrary ids reversibly (see ``_unsafe_id``)."""
    import base64

    return "b64_" + base64.urlsafe_b64encode(custom_id.encode("utf-8")).decode("ascii").rstrip("=")


def _unsafe_id(safe: str) -> str:
    import base64

    if not safe.startswith("b64_"):
        return safe
    body = safe[4:]
    return base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)).decode("utf-8")


def _anthropic_collect(client, batch_id: str) -> dict | None:
    b = client.messages.batches.retrieve(batch_id)
    if b.processing_status != "ended":
        return None
    out = {}
    for entry in client.messages.batches.results(batch_id):
        cid = _unsafe_id(entry.custom_id)
        if entry.result.type == "succeeded":
            m = entry.result.message
            out[cid] = ("".join(getattr(c, "text", "") for c in m.content), {"input_tokens": m.usage.input_tokens, "output_tokens": m.usage.output_tokens, "finish_reason": m.stop_reason}, m.stop_reason)
        else:
            out[cid] = ("", {}, "error")
    return out


# ------------------------------------------------------------------ driver
def run_batch(provider: str, requests: list, state_path: Path, poll_seconds: int = 60, tag: str = "llm_distract") -> dict:
    """Submit ``requests`` (in chunks), wait for completion and return {custom_id: (text, usage, finish_reason)}.
    Re-running with the same ``state_path`` re-attaches to the batches recorded there instead of resubmitting."""
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        submit, collect = _openai_submit, _openai_collect
    elif provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        submit, collect = _anthropic_submit, _anthropic_collect
    else:
        raise ValueError(provider)
    state = _state(state_path)
    ids = state.get("batch_ids") or []
    if not ids:
        for i in range(0, len(requests), CHUNK):
            ids.append(submit(client, requests[i:i + CHUNK], f"{tag}-{i // CHUNK}"))
        state["batch_ids"], state["n_requests"], state["provider"], state["submitted"] = ids, len(requests), provider, time.strftime("%Y-%m-%dT%H:%M:%S")
        _save(state_path, state)
        print(f"  submitted {len(requests)} requests in {len(ids)} batch(es) to {provider}: {ids}", flush=True)
    results: dict = {}
    pending = list(ids)
    while pending:
        for bid in list(pending):
            got = collect(client, bid)
            if got is not None:
                results.update(got)
                pending.remove(bid)
                print(f"  batch {bid} finished: {len(got)} results ({time.strftime('%H:%M:%S')})", flush=True)
        if pending:
            time.sleep(poll_seconds)
    state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["n_results"] = len(results)
    _save(state_path, state)
    return results
