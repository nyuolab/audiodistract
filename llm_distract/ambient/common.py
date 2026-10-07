"""Constants, config loading and provenance-stamped parquet I/O shared by the ambient module (prefix ``amb``)."""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from llm_distract import __version__

SPLITS = ("aci_bench_test", "aci_bench_valid", "mts_dialog_test1", "mts_dialog_test2", "mts_dialog_valid")
DISTRACTORS = ("nonliteral", "bystander")
MODEL_ORDER = ("Gemma-2-9B", "Mistral-7B", "Llama-3.1-8B", "Qwen2.5-7B")  # short names, ascending effect
PAIR_KEYS = ["source_dataset", "item_id", "model", "distractor_type"]
QUALITY_METRICS = ("clinical_correctness", "completeness", "succinctness", "hallucination", "overall_quality")


def family_of(source_dataset: str) -> str:
    """Dataset family label used in every table and figure."""
    return "ACI-Bench" if str(source_dataset).startswith("aci") else "MTS-Dialog"


def short_model(model: str) -> str:
    """HF id -> short display name from the shared style module (imported lazily: matplotlib-free data paths)."""
    from llm_distract.common.style import MODEL_SHORT

    return MODEL_SHORT.get(model, model)


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """Read the repository config (paths + ambient hyperparameters)."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def provenance(**extra) -> dict:
    """Provenance block (code version, git commit, UTC timestamp, caller-supplied run parameters)."""
    return {"code_version": __version__, "git_commit": _git_commit(),
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), **extra}


def write_parquet(df: pd.DataFrame, path: str | Path, meta: dict | None = None, compression: str = "zstd") -> Path:
    """Write a DataFrame to parquet with a provenance JSON blob in the schema metadata (key ``llm_distract``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"llm_distract"] = json.dumps(provenance(**(meta or {})), default=str).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path, compression=compression)
    return path


def read_meta(path: str | Path) -> dict:
    """Return the provenance blob stored by :func:`write_parquet` (empty dict if absent)."""
    metadata = pq.read_schema(path).metadata or {}
    blob = metadata.get(b"llm_distract")
    return json.loads(blob.decode("utf-8")) if blob else {}
