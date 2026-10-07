"""Regenerate the published aggregate tables into a separate directory, without inference."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory; frozen tables are never overwritten")
    args = parser.parse_args()
    out = args.out.resolve()
    if out == ROOT / "results" or ROOT / "results" in out.parents:
        parser.error("Choose an output directory outside the frozen results tree")
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory must be new or empty")
    import sys
    sys.path.insert(0, str(ROOT))
    from llm_distract.ambient import analyze as documentation, failure_modes
    from llm_distract.audio import analyze as audio
    from llm_distract.mechanism import analyze as mechanism

    out.mkdir(parents=True, exist_ok=True)
    documentation.run(ROOT / "results/raw_v3", ROOT / "data/ambient", out / "v3", "insertions_v3.parquet")
    failure_modes.run(ROOT / "results/raw_v3", ROOT / "data/ambient/insertions_v3.parquet", out / "v3",
                      frozen_locations=ROOT / "results/v3/amb_failure_notes.csv")
    audio.run(ROOT / "results/raw", out)
    mechanism.run(ROOT / "results/raw", out)
    print(f"Documentation, audio and mechanism tables written to {out}")


if __name__ == "__main__":
    main()
