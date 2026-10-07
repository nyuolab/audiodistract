# Clinical note evaluation

Download the frozen v3 inputs with `meddistract-download --dataset notes`. The benchmark contains 576 encounters (77 ACI-Bench and 499 MTS-Dialog) and two insertions per encounter. `pairs_v3.parquet` is the evaluation input; `insertions_v3.parquet` holds generated exchanges, targets and labels.

The local generator defaults to native chat templates, greedy decoding, a 1,024-token output cap and batch size two. Model revisions are in `configs/default.yaml`. `--no-chat-template` explicitly selects the older raw-prompt mode. API generation uses a single user message, with provider-specific reasoning defaults; pass `--max-new-tokens` explicitly for your desired budget.

```bash
python -m llm_distract.ambient.generate_notes --backend api --provider openai   --model gpt-5.4-2026-03-05 --max-new-tokens 1024   --split aci_bench_test --distractor bystander --out outputs/my_notes
```

Run both clean and distracted conditions under identical settings. The generator writes `notes.parquet` and a run manifest under `<out>/<model_label>/<split>/<distractor>/`. API runs save checkpoints for resuming.

```bash
python -m llm_distract.ambient.judge --notes PATH_TO_NOTES   --pairs data/ambient/pairs_v3.parquet --mode contamination_paired
python -m llm_distract.ambient.judge --notes PATH_TO_NOTES --mode quality
python -m llm_distract.ambient.export_reruns --runs-root outputs/my_notes --raw outputs/my_raw
python -m llm_distract.ambient.analyze --raw outputs/my_raw --out outputs/my_analysis
```

The paired contamination judge scores each note independently against the clean transcript and insertion target. Contamination is binary; severity ranges from 0 (absent) to 3 (clinically consequential incorporation). Note-quality dimensions use a 1–5 scale, with hallucination scored separately as a binary indicator. Prompt templates are in `llm_distract/ambient/prompts.py` and in the dataset's `protocol/prompts.json`.

The saved study results include eight note models and 18,432 records. Two empty saved notes remain in the primary denominator. Missing and unresolved attribution labels are retained. The matched single-note control uses the same target and judge; reproduce it with `scripts/analyze_judge_control.py`. Use encounter-level paired resampling, because distractors and models share encounters.

Frozen analyses can be regenerated together with `python scripts/reproduce.py --out outputs/reproduced`. Full column definitions, source attribution and limitations are in the [dataset card](https://huggingface.co/datasets/NYU-OLAB/MedDistractNotes).
