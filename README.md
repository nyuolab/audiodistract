# AudioDistract

Tools and benchmarks for studying incidental-information contamination in clinical audio and generated notes. AudioDistract provides paired clean/distracted inputs, note-generation and scoring tools, audio overlays, and attention-head interventions.

[Paper](https://arxiv.org/abs/2610.08585) · [MedDistractNotes](https://huggingface.co/datasets/NYU-OLAB/MedDistractNotes) · [MedDistractAudio](https://huggingface.co/datasets/NYU-OLAB/MedDistractAudio)

This repository accompanies the clinical documentation and audio study. Its mechanistic experiments reuse [MedDistractQA](https://huggingface.co/datasets/KrithikV/MedDistractQA), introduced in the earlier paper [Medical large language models are easily distracted](https://arxiv.org/abs/2504.01201). The QA benchmark retains its existing name and citation.

## Install and download

Use Python 3.10 or later. Run commands from this checkout.

```bash
git clone https://github.com/nyuolab/audiodistract.git
cd audiodistract
python -m venv .venv
source .venv/bin/activate
pip install -e .
meddistract-download --dataset all
```

The downloader installs the frozen benchmark tables at the paths used by the analysis code. Dataset revisions are pinned in `llm_distract/release.json`; every file is SHA256-verified. Existing files with different contents are rejected. This download excludes the large audio recordings; access those through the Audio dataset's `recordings` configuration.

```python
from datasets import load_dataset
pairs = load_dataset("NYU-OLAB/MedDistractNotes", "pairs", split="test")
example = pairs[0]
clean = example["clean_transcript"]
distracted = example["distracted_transcript"]
```

MedDistractNotes contains 1,152 paired inputs from 576 encounters. MedDistractAudio contains 456 paired inputs from 57 mock consultations, with 513 clean/mixed recordings. Each benchmark's `test` split is its full evaluation corpus; encounters recur across perturbations and must remain together when partitioning or bootstrapping.

## Evaluate a model

Install `pip install -e '.[gpu]'` for local generation, or `pip install -e '.[api]'` for API generation and judging. API keys are read from environment variables.

```bash
python -m llm_distract.ambient.generate_notes   --model meta-llama/Llama-3.1-8B-Instruct   --split mts_dialog_test1 --distractor bystander --out outputs/my_notes
python -m llm_distract.ambient.judge   --notes outputs/my_notes/meta-llama__Llama-3.1-8B-Instruct/mts_dialog_test1/bystander/notes.parquet   --pairs data/ambient/pairs_v3.parquet --mode contamination_paired
```

The Notes generator defaults to the current v3 pairs, native chat formatting and a 1,024-token local output limit. Consult [Notes](docs/amb.md), [Audio](docs/audio.md), [reasoning](docs/qa.md), and [head interventions](docs/mech.md) for the separate workflows and protocol details. Automated judge scores are research measurements, not clinician adjudications.

## Reproduce the frozen study tables

After downloading both datasets:

```bash
python scripts/reproduce.py --out outputs/reproduced
python scripts/analyze_judge_control.py --out outputs/judge_control
```

These CPU commands use saved outputs and do not call model APIs. They regenerate documentation, audio, head-intervention and matched-judge-control results. Reference aggregate tables are in `results/`; the small frozen QA inputs and mechanistic measurements are bundled. Generating new model outputs requires model access and appropriate compute, and may incur provider charges.

The audio recordings distributed on Hugging Face were reconstructed from hash-verified PriMock57 source WAVs and the saved mixing recipes. All 513 PCM waveforms match the original experiment audio exactly; FLAC preserves those samples losslessly. Saved transcripts, generated notes and judgments are unchanged. The original run manifests confirm raw prompts and a 512-token limit for audio notes, which differs from the newer text-only Notes protocol.

## Development and license

```bash
pip install -e '.[dev]'
pytest -q
```

Code is MIT-licensed. Benchmark data and derived annotations have separate terms and source attribution in [DATA_LICENSE.md](DATA_LICENSE.md) and the dataset cards. Model weights are downloaded from their providers and retain their own licenses. Cite the paper and the source corpora used in your experiment; see `CITATION.cff`.
