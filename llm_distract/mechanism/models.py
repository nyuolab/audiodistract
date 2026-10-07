"""Configuration, pinned model loading, seeding and provenance records shared by the GPU scripts."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import subprocess
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def label(model_id: str) -> str:
    """Filesystem-safe model label (``org/name`` -> ``org__name``)."""
    return model_id.replace("/", "__")


def load_config(path: Optional[Path] = None) -> dict:
    """configs/default.yaml (or the given file); ``paths`` entries are resolved relative to the repo root."""
    cfg = yaml.safe_load(Path(path or REPO_ROOT / "configs" / "default.yaml").read_text())
    for key, value in cfg["paths"].items():
        env = os.environ.get(f"LLM_DISTRACT_{key.upper()}")
        p = Path(env or value)
        cfg["paths"][key] = p if p.is_absolute() else REPO_ROOT / p
    return cfg


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Arguments every GPU script shares: model, config, data locations, dtype, attention implementation."""
    ap.add_argument("--model", required=True, help="HF model id (revision pinned via configs/default.yaml)")
    ap.add_argument("--config", type=Path, default=None, help="YAML config (default configs/default.yaml)")
    ap.add_argument("--data-dir", type=Path, default=None, help="data directory (default from config)")
    ap.add_argument("--json-dir", type=Path, default=None, help="directory with MedDistractQA_*.json (fallback source)")
    ap.add_argument("--revision", default=None, help="override the pinned HF revision")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn", default=None, choices=[None, "eager", "sdpa"], help="attention implementation")


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def code_version() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - not a git checkout
        from llm_distract import __version__
        return f"llm_distract-{__version__}"


def load_model_and_tokenizer(model_id: str, revision: Optional[str], dtype: str = "bfloat16",
                             attn_implementation: Optional[str] = None):
    """Frozen causal LM (eval mode, device_map auto) and tokenizer at a pinned revision; pad = eos when missing."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"revision": revision, "torch_dtype": getattr(torch, dtype), "device_map": "auto"}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def setup(args: argparse.Namespace, attn_implementation: Optional[str] = None):
    """Resolve config/revision, load pairs, model and tokenizer; returns (cfg, pairs, model, tokenizer, provenance)."""
    from llm_distract.mechanism.data import load_pairs

    cfg = load_config(args.config)
    revision = args.revision or cfg["mechanism"]["revisions"].get(args.model)
    attn = attn_implementation or args.attn
    if attn is None and "gemma-2" in args.model.lower():
        attn = "eager"  # Gemma-2 attention (logit soft-capping) is only faithful with eager attention
    pairs = load_pairs(args.data_dir or cfg["paths"]["data_dir"], args.json_dir)
    model, tokenizer = load_model_and_tokenizer(args.model, revision, args.dtype, attn)
    from llm_distract.mechanism import data as _data
    _data.set_tokenizer(tokenizer)
    prov = provenance(args.model, revision, dtype=args.dtype, attn_implementation=attn, args=vars(args))
    return cfg, pairs, model, tokenizer, prov


def provenance(model_id: str, revision: Optional[str], **extra) -> dict:
    """Provenance record written into every artifact: model, revision, code version, library versions, timestamp."""
    import torch
    import transformers

    from llm_distract.mechanism.data import ADD_BOS, CHAT_TEMPLATE
    return {"model": model_id, "revision": revision, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "code_version": code_version(), "torch": torch.__version__, "transformers": transformers.__version__, "add_bos": ADD_BOS,
            "chat_template": CHAT_TEMPLATE,
            **{k: v for k, v in extra.items()}}
