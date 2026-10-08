# Attention-head interventions

The toolkit supports answer scoring, contrastive causal head gating, clean-to-distracted activation patching, suppression and attention summaries on MedDistractQA. Frozen measurements are bundled in `results/raw/mech_*.parquet`; the fixed QA table is in `data/meddistractqa/`.

```bash
python -m llm_distract.mechanism.analyze --out outputs/mechanism_analysis
```

The primary comparisons use 127 held-out items from a seed-0 split, excluded from gate training. Some older measurements also include all 1,273 items; aggregate tables identify these sensitivity columns.

## Run an intervention

Install the GPU extra and obtain access to the selected model. For example:

```bash
python -m llm_distract.mechanism.evaluate --model meta-llama/Llama-3.1-8B-Instruct --split heldout
python -m llm_distract.mechanism.train_cchg --model meta-llama/Llama-3.1-8B-Instruct --distractor nonliteral --seed 0
python -m llm_distract.mechanism.patch --model meta-llama/Llama-3.1-8B-Instruct --distractor nonliteral   --mask outputs/mech/cchg/meta-llama__Llama-3.1-8B-Instruct/nonliteral/seed0/mask.pt --split heldout
```

Other entry points are `mechanism.suppress`, `mechanism.attention` and `mechanism.seed_stability`; inspect their `--help` for controls and output paths. Model revisions and training hyperparameters are in `configs/default.yaml`.

The current frozen measurements use native chat templates, which are enabled by default. `MECH_CHAT_TEMPLATE=0` explicitly selects the older raw-prompt protocol. The answer is scored after an `Answer:` prefix. Keep the same formatting for fitting gates, evaluation and interventions, and preserve the run manifests.

The implemented objective minimizes the squared clean/distracted correct-answer log-probability difference plus a penalty of 5 times mean(1 − gate). Sigmoid gates start at 0.5; Adam runs 500 updates with a learning rate from 0.01 to 0.001. The default micro-batch of two with eight accumulation steps gives 16 pairs per update. Gates below 0.5 define selected heads. Patching aligns the shared prefix/suffix and leaves inserted tokens unpatched. Suppression scales selected heads; random controls exclude the selected set. Attention mass is descriptive, rather than a causal endpoint.

Model-specific memory use varies; Gemma's eager attention may require micro-batch one with gradient accumulation 16. Repeated seeds are supported, while the published main masks use seed zero. The training split, prompt format, model revision and gate parameters must match when comparing runs.
