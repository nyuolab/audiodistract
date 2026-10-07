# MedDistractQA inputs

`meddistractqa_v2.parquet` has 1,273 rows in MedQA-USMLE test order, with positional `item_id` 0–1272. It includes clean and distracted questions, `choice_A` through `choice_D`, `correct_answer`, the frozen `nonliteral_sentence` and `bystander_sentence`, and `medical_system` / `medical_competency` labels. `labels.csv` is the label-only companion.

Distractor sentences were generated with GPT-4o in February 2025; labels were generated with o3-mini. The distracted question inserts the sentence before the final sentence of the original question. Load the frozen table using `llm_distract.meddistractqa.data.load_meddistractqa()`; regeneration from source-generation files is an optional authoring workflow and is not needed for evaluation.

Public release: https://huggingface.co/datasets/KrithikV/MedDistractQA . Original MedQA: https://github.com/jind11/MedQA . See `../../DATA_LICENSE.md` for attribution and terms.
