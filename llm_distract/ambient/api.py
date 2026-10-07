"""Minimal OpenAI / Anthropic HTTP clients (stdlib only) used by ``perturb`` and ``judge``.

Keys are read from ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` (export them or load a local .env yourself); they
are never written to disk or logs.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def _post(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{url} -> HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}") from exc


def openai_chat(model: str, system: str, user: str, max_tokens: int, json_mode: bool = False,
                temperature: float = 0.0, timeout: float = 120.0) -> tuple[str, dict]:
    """OpenAI chat completion; returns (text, usage). ``json_mode`` sets response_format json_object."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "max_completion_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    body = _post(OPENAI_URL, {"Authorization": f"Bearer {key}"}, payload, timeout)
    text = str(body.get("choices", [{}])[0].get("message", {}).get("content") or "")
    usage = body.get("usage", {})
    return text, {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}


def anthropic_message(model: str, system: str, user: str, max_tokens: int, timeout: float = 120.0) -> tuple[str, dict]:
    """Anthropic Messages call (no temperature set, as in the published judge run); returns (text, usage)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    payload = {"model": model, "max_tokens": max_tokens, "system": system,
               "messages": [{"role": "user", "content": user}]}
    body = _post(ANTHROPIC_URL, {"x-api-key": key, "anthropic-version": "2023-06-01"}, payload, timeout)
    text = "".join(part.get("text", "") for part in body.get("content", []) if part.get("type") == "text")
    usage = body.get("usage", {})
    return text, {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}


def complete(provider: str, model: str, system: str, user: str, max_tokens: int, json_mode: bool = False,
             timeout: float = 120.0) -> tuple[str, dict]:
    """Dispatch on provider name ('openai' or 'anthropic')."""
    if provider == "openai":
        return openai_chat(model, system, user, max_tokens, json_mode=json_mode, timeout=timeout)
    if provider == "anthropic":
        return anthropic_message(model, system, user, max_tokens, timeout=timeout)
    raise ValueError(f"unknown provider {provider!r}")


def extract_json(text: str) -> dict:
    """Parse the outermost JSON object in a model reply (first '{' to last '}')."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in reply")
    payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("reply is not a JSON object")
    return payload
