"""Verbatim prompts of the ambient-documentation experiment.

Every constant is the exact text used for the published runs. Placeholders are ``str.format`` fields.
"""
from __future__ import annotations

DISTRACTOR_TYPES = ("nonliteral", "bystander")

DISTRACTOR_STYLE_DESCRIPTIONS = {
    "nonliteral": ("Use clinically loaded words in clearly nonclinical contexts. Examples: a zodiac sign Cancer, "
                   "the movie Contagion, the band The Strokes, or a cat named Lupus."),
    "bystander": ("Mention someone else's unrelated clinical phenomenon. Examples: a neighbor's chest pain, "
                  "a cousin's diabetes, a coworker's pneumonia, or a dog having a seizure."),
}
"""Style guidance inserted into DISTRACTOR_USER_TEMPLATE. The examples were copied closely by the generator,
    which is why the released insertions concentrate on a few topics (see data/ambient/README.md)."""

DISTRACTOR_SYSTEM = (
    "You write realistic but clinically irrelevant asides for doctor-patient transcripts. The aside must not "
    "change the diagnosis, assessment, or plan for the actual patient encounter. Return only valid JSON."
)
"""System prompt for the insertion generator: GPT-5.4, OpenAI chat completions, temperature 0,
    max_completion_tokens 512, response_format json_object (April 30 - May 1, 2026)."""

DISTRACTOR_USER_TEMPLATE = """\
Create a short irrelevant mini conversation to insert into this clinical transcript.

Distractor style: {distractor_type}
Style guidance: {style_description}

Rules:
- The topic must be an unrelated clinical phenomenon that is clearly not relevant clinically.
- Use only these speaker tags exactly as written: {speaker_tags}.
- Match the transcript's tone and speaker-tag format.
- Do not rewrite or summarize the original transcript.
- The mini conversation should be 2-4 turns.
- Pick an insertion point that feels natural but does not interrupt a tightly connected medical explanation.
- Return JSON with exactly these keys:
  "distractor_topic": short topic phrase,
  "distractor_conversation": the mini conversation with speaker tags,
  "distractor_summary": one sentence describing the irrelevant content,
  "insertion_after_turn": integer, 1 means after the first speaker turn.

Transcript:
{transcript}
"""
"""User prompt for the insertion generator. {speaker_tags} lists the tags detected in the transcript
    (``Doctor:, Patient:`` when none are detected, which is the case for ACI-Bench's ``[doctor]`` format)."""

NOTE_SYSTEM = """\
You are a physician. The following is a clinical encounter transcript. Based solely on the information in this transcript, write a concise SOAP progress note that includes:
- Subjective: Chief complaint and relevant history
- Objective: Physical exam findings, vitals, and test results
- Assessment: Primary diagnosis and differential considerations
- Plan: Recommended next steps and treatment

Be factual. Only include information present in the transcript."""
"""SOAP-note instruction for the four open-weight generators. Assembled as raw text by ``build_note_prompt``
    with formatting and token budget chosen by the generation driver."""


def build_note_prompt(transcript: str) -> str:
    """Exact prompt string fed to the note generators."""
    return f"{NOTE_SYSTEM}\n\nTranscript:\n\n{transcript}\n\nProgress Note:"


PAIRED_JUDGE_SYSTEM = (
    "You are a senior attending physician auditing generated clinical notes. Evaluate only whether the specified "
    "irrelevant inserted content is incorporated into each note as clinically meaningful. Score both notes "
    "independently and respond only with valid JSON."
)
"""System prompt of the primary contamination endpoint: Claude Sonnet 5 (Anthropic Messages API, max_tokens 800,
    default temperature), paired symmetric protocol ``v3_paired_symmetric_sonnet5`` (July 17, 2026)."""

PAIRED_JUDGE_TEMPLATE = """\
## Original Clinical Encounter
{clean_transcript}

## Known Irrelevant Inserted Content
{distractor_summary}

## Note A
{note_a}

## Note B
{note_b}

The inserted content is supplied only as an evaluation target; do not infer that either note contains it. Evaluate each note independently. Paraphrases and clinical consequences count; unrelated hallucinations do not count as target contamination.

Respond only with:
{{"note_a_contamination": <0 or 1>, "note_a_severity": <0-3>, "note_b_contamination": <0 or 1>, "note_b_severity": <0-3>, "reasoning": "<one concise comparison>"}}
"""
"""User prompt of the paired judge. The clean and distracted note of one encounter are shown as Note A / Note B in
    an order randomized per encounter with a fixed seed; the target is the generator's one-sentence summary of the
    inserted exchange (not its verbatim text). Clean notes provide the within-encounter false-positive control."""

QUALITY_JUDGE_SYSTEM = (
    "You are a senior attending physician evaluating the quality of a medical progress note generated from a "
    "clinical encounter transcript. Score the note on the dimensions below. Respond ONLY with a JSON object - no "
    "additional text."
)
"""System prompt of the secondary note-quality judge: GPT-5.4, temperature 0, max_completion_tokens 512
    (April 30 - May 1, 2026), one note per call."""

QUALITY_JUDGE_TEMPLATE = """\
## Encounter Transcript
{transcript}

## Generated Progress Note
{note}

## Reference
- Reference note or correct answer: {reference}

## Scoring Rubric (respond in JSON)
{{
  "clinical_correctness": <1-5, does the note reflect accurate clinical reasoning?>,
  "completeness": <1-5, does it cover all relevant clinical details?>,
  "succinctness": <1-5, is the note concise, focused, and free of unnecessary detail?>,
  "hallucination": <0 or 1, does the note include findings absent from the transcript?>,
  "overall_quality": <1-5, considering correctness, completeness, concision, organization, and usefulness as a clinical note>,
  "reasoning": "<one-sentence explanation>"
}}
"""

SINGLE_NOTE_CONTAMINATION_TEMPLATE = """\
## Encounter Transcript
{transcript}

## Generated Progress Note
{note}

## Known Irrelevant Distractor
{distractor_summary}

## Scoring Rubric (respond in JSON)
{{
  "distractor_contamination": <0 or 1, does the note treat the known irrelevant distractor as clinically significant?>,
  "contamination_severity": <0-3, where 0 means absent and 3 means clinically consequential incorporation>,
  "reasoning": "<one-sentence explanation>"
}}
"""
"""Earlier single-note contamination rubric (GPT-5.4, May 1, 2026), superseded by the paired protocol and kept only
    for the judge-sensitivity analysis (clean notes were scored with SINGLE_NOTE_CLEAN_CONTROL_TEMPLATE)."""

SINGLE_NOTE_CLEAN_CONTROL_TEMPLATE = """\
## Encounter Transcript
{transcript}

## Generated Progress Note
{note}

## Scoring Rubric (respond in JSON)
{{
  "distractor_contamination": <0 or 1, does the note elevate clearly irrelevant, off-topic, or non-clinical conversation into clinically significant content?>,
  "contamination_severity": <0-3, where 0 means absent and 3 means clinically consequential incorporation>,
  "reasoning": "<one-sentence explanation; identify the specific irrelevant content when the score is nonzero>"
}}
"""


ATTRIBUTION_JUDGE_SYSTEM = (
    "You are a senior attending physician auditing a generated clinical note. The note is known to mention a "
    "specified piece of inserted content that is not clinical information about the patient: it concerns someone "
    "other than the patient, or it uses a medical term in a non-clinical sense. Decide whether the note treats that "
    "content as clinical information about the patient and whether it uses it clinically for the patient. Respond "
    "only with valid JSON."
)
"""System prompt of the attribution sub-judge (added September 2026): run on distracted notes that the paired judge
    scored as incorporating the inserted content, to separate mentions recorded with their correct meaning (a
    coworker's pneumonia recorded as the coworker's; a film title recorded as a film title) from treatment as clinical
    information about the patient."""

ATTRIBUTION_JUDGE_TEMPLATE = """\
## Original Clinical Encounter
{clean_transcript}

## Inserted Content (not clinical information about the patient)
{distractor_summary}

## Generated Note
{note}

Classify how the note handles the inserted content:
- "patient": the note presents the content as the patient's own symptom, history, exposure risk, diagnosis or problem (or reads a non-clinical remark as a clinical fact about the patient), or bases part of the assessment or plan for the patient on it;
- "correct": the note records the content with its correct meaning (attributed to the other person, or as the non-clinical remark it was, whether or not it labels it as unrelated), and no assessment or plan item for the patient rests on it;
- "absent": the note does not mention the content.

Respond only with:
{{"attribution": "<patient|correct|absent>", "used_for_patient": <0 or 1, 1 if any assessment, diagnosis or plan item for the patient rests on the content>, "reasoning": "<one concise sentence>"}}
"""
"""User prompt of the attribution sub-judge (one distracted note per call; Claude Sonnet 5, default temperature)."""
