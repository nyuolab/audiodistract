# Background speech evaluation

[MedDistractAudio](https://huggingface.co/datasets/NYU-OLAB/MedDistractAudio) contains 57 clean and 456 mixed recordings. Each PriMock57 foreground consultation has two donor segments at −20, −15, −10 and −5 dB. The primary reported level is −10 dB.

```bash
meddistract-download --dataset audio
python -m llm_distract.audio.analyze --out outputs/audio_analysis
```

The table download supplies pairs, saved Whisper large-v3 transcripts, generated notes, automated judgments and leakage measurements. Fetch the lossless audio separately:

```python
from datasets import load_dataset, Audio
recordings = load_dataset("NYU-OLAB/MedDistractAudio", "recordings", split="test")
recordings = recordings.cast_column("audio", Audio(decode=False))
```

Join on `item_id` and `distractor_type`; the clean recording uses `distractor_type="clean"`. The `audio` field holds embedded FLAC bytes and a filename. Decoding with `Audio(decode=True)` requires the audio backend supported by your installed `datasets` version. `duration_seconds` measures waveform length; `transcripts.seconds` measures historical ASR processing time.

## Generate a new experiment

```bash
python -m llm_distract.audio.data --download
python -m llm_distract.audio.mix --out outputs/audio/mixes
python -m llm_distract.audio.transcribe --manifest outputs/audio/mixes/manifest.parquet --out outputs/audio/transcripts.parquet
python -m llm_distract.audio.build_pairs --out data/audio/pairs.parquet
python -m llm_distract.audio.leakage --pairs data/audio/pairs.parquet
python -m llm_distract.audio.generate_notes --model meta-llama/Llama-3.1-8B-Instruct
python -m llm_distract.ambient.judge --notes outputs/audio/meta-llama__Llama-3.1-8B-Instruct/notes.parquet   --pairs data/audio/pairs.parquet --mode contamination_paired
```

Source download needs Git LFS. Transcription/generation needs the GPU dependencies and model access; judging needs the API dependencies and credentials. Use a separate checkout or data/output paths for a new experiment so the frozen inputs remain identifiable.

Mixing averages the original speaker channels into mono 16-kHz audio. A seeded selection chooses two other consultations; a keyword-ranked donor segment spans 5–15 seconds. One onset per foreground/donor pair serves every level. Background gain is relative to foreground RMS; mixtures are peak-normalized to 0.95. The distributed recordings reconstruct the saved recipe rather than rerunning segment selection. All 513 reconstructed PCM waveforms were verified against the original experiment WAVs and through a FLAC round trip.

The audio generator retains raw prompting and a 512-token default in `audio.note_max_new_tokens`; `--chat-template` and `--max-new-tokens` select alternatives for new runs. The original run manifests confirm raw prompts and a 512-token limit for all four models. `protocol/note_generation.json` in the Audio dataset records these settings and model revisions.

Two of 1,824 note pairs have unparseable contamination judgments and are excluded by the analysis. Bootstrap by foreground encounter. Controlled overlays of mock consultations and one ASR system do not establish performance under real room acoustics, overlapping speakers or clinical deployment.
