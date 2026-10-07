# Clinical reasoning benchmark

[MedDistractQA](https://huggingface.co/datasets/KrithikV/MedDistractQA) is already publicly available. The bundled `data/meddistractqa/meddistractqa_v2.parquet` combines its 1,273 clean MedQA test questions, two frozen distractor sentences per item, answer choices, answer keys and topic/competency labels. It also supplies the inputs used by the head-intervention tools.

```python
from llm_distract.meddistractqa.data import load_meddistractqa
items = load_meddistractqa()
```

To evaluate a new engine, install the relevant API or GPU extra and consult:

```bash
python -m llm_distract.meddistractqa.run_eval --help
```

Select a backend and run clean, nonliteral and bystander conditions with matched settings. API credentials come from environment variables. Each run writes its answers and a settings manifest under `outputs/qa/` by default. Use `--chat-template` for the native chat interface where supported, and set the token budget explicitly.

The answer parser, paired analysis utilities and model registry are supplied for reuse. Full historical generative-QA run archives are not bundled with this toolkit. The frozen CPU reproduction command covers the documentation, audio and head-intervention results associated with the linked preprint.

The combined QA table preserves the released distractors; generating new sentences will produce a different benchmark. Source MedQA content is MIT-licensed; MedDistractQA additions are CC BY 4.0. See `DATA_LICENSE.md`.
