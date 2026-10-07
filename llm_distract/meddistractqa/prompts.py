"""Verbatim prompts of the MedDistractQA pipeline.

Every constant is the exact template used for the published runs (copied from the original private code);
``{question}``, ``{choices}``, ``{context}`` and ``{clinical_topic}`` are ``str.format`` fields.
"""
from __future__ import annotations

PROMPT_VARIANTS = ("standard", "ignore", "structured")

PROMPT_STANDARD = '\n## Question\n{question}\n\n## Choices\n{choices}\n\n## Context\n{context}\n\n## Instructions\nYou are provided a medical multiple choice Question and potentially some medical Context. Solve the question in a step-by-step fashion and provide a single letter as the final answer in the end, formatted as "Therefore, the final answer is [X]."\n'
"""Zero-shot chain-of-thought evaluation prompt (prompt_eval_mod). Used for every clean, nonliteral and
    bystander run of all 35 engines (Feb-Apr 2025 open-weight/OpenAI/Anthropic runs; Mar-Apr 2026 Claude 4.x,
    Gemini 3.1 and GPT-5.4 runs). {context} is always empty in the paper runs."""

PROMPT_IGNORE = '\n## Question\n{question}\n\n## Choices\n{choices}\n\n## Context\n{context}\n\n## Instructions\nYou are provided a medical multiple choice Question and potentially some medical Context. Ignore irrelevent information given in the questions. Solve the question in a step-by-step fashion and provide a single letter as the final answer in the end, formatted as "Therefore, the final answer is [X]."\n'
"""Standard prompt plus the explicit instruction 'Ignore irrelevent information given in the questions.'
    (prompt_eval_mod_IIC_daniel; original spelling kept). Run on nonliteral-distracted questions for 26 engines,
    2025-03-30, max_tokens 1024."""

PROMPT_STRUCTURED = '\n\nYou are a medical expert tasked with answering multiple-choice medical questions. Your goal is to provide accurate answers based on the given information, while being cautious of extraneous or irrelevant details. Follow these steps carefully:\n\n1. First, review the medical context provided (if any):\n<medical_context>\n{context}\n</medical_context>\n\n2. Next, carefully read the question:\n<question>\n{question}\n</question>\n\n3. Now, examine the multiple choice options:\n<choices>\n{choices}\n</choices>\n\n4. Analyze the question and choices:\n   - Focus on the core medical concepts presented in the question.\n   - Identify any extraneous or irrelevant information and disregard it.\n   - Consider how the medical context (if provided) relates to the question.\n   - Evaluate each choice based on your medical knowledge and the given information.\n\n5. Use your inner monologue to reason through the question and choices:\n<inner_monologue>\n- Discuss the key medical concepts involved in the question.\n- Explain why certain information might be irrelevant or misleading.\n- Evaluate the validity of each choice, providing medical reasoning for why it may or may not be correct.\n- Consider any potential tricks or distractors in the question or choices.\n</inner_monologue>\n\n6. Provide your final answer:\n   - Based on your analysis and reasoning, select the most appropriate answer.\n   - Present your conclusion in the following format:\n     "Therefore, the final answer is [X]."\n   Where [X] is replaced by the single letter (A, B, C, D, etc.) corresponding to the correct choice.\n\nRemember to rely on established medical knowledge and the relevant information provided in the question and context. Do not be swayed by extraneous details or irrelevant information.\n'
"""Elaborate structured prompt with XML-style sections and an <inner_monologue> scratchpad
    (prompt_eval_mod_anthropic_prompt). Run on clean and nonliteral questions for 23 engines, 2025-03-27/28."""

GEN_NONLITERAL = '\nExample sentences (for other questions):\n\n1. "The patient\'s zodiac sign is Cancer."\n2. "The patient said their friend\'s business is hemorrhaging money after a bad investment."\n3. "The patient mentioned that their new car has a transmission fluid leak."\n4. "The patient described their favorite art exhibit as having vascular pattern in the sculptures."\n5. "The patient\'s professor delivered a lecture about the heart of ancient civilizations."\n\n## Medical Question: {question}\n\n## Clinical topic: {clinical_topic}\n\nGenerate one sentence using a medical term related to the clinical topic, where clinical terminology is used as nonmedical meaning. The statement should NOT affect the diagnostic process or decision-making for the medical question in any meaningful way. It should not convey any medical information or symptomatic information. Avoid including any information that could mislead clinical reasoning.\n\nOnly respond with the confounding sentence.\n'
"""Nonliteral-distractor generation prompt (prompt_gen_alpha_confounder). GPT-4o, temperature 0.6,
    max 200 output tokens, one call per MedQA test item, 2025-02-02. {clinical_topic} = full text of the first
    incorrect option in letter order (A if A is wrong, else B)."""

GEN_BYSTANDER = '\nExample sentences:\n\n1. "The patient\'s sister said her friend has high blood pressure and has been having seizures."\n2. "The patient\'s dog has diabetes and cannot see properly."\n3. "The patient\'s neighbor mentioned that their cat has been sneezing and refusing to eat."\n4. "The patient\'s cousin said her coworker\'s father has arthritis and struggles to walk long distances."\n5. "The patient\'s brother mentioned that his friend\'s goldfish has been swimming erratically and losing color."\n    \n\n## Medical Question: {question}\n\n## Clinical topic: {clinical_topic}\n\nGenerate one statement, where another person/animal is referenced with relation to a medical term as seen in the example sentences. The statement should also be related to the clinical topic. The statement should NOT affect the diagnostic process or decision-making for the medical question in any meaningful way. It should not convey any medical information or symptomatic information. Avoid including any information that could mislead clinical reasoning.\n\nOnly respond with the confounding sentence.\n'
"""Bystander-distractor generation prompt (prompt_gen_beta_confounder). GPT-4o, temperature 0.6,
    max 200 output tokens, 2025-02-03; same clinical-topic rule as GEN_NONLITERAL."""

LABEL_SYSTEM = '\n\n## Question\n{question}\n\n## Choices\n{choices}\n\nYou are a medical expert. Please select the topic of the question from the following list of topics:\n\n1. Human Development\n2. Immune System\n3. Blood & Lymphoreticular System\n4. Behavioral Health\n5. Nervous System & Special Senses\t\n6. Musculoskeletal System/Skin & Subcutaneous Tissue\t\n7. Cardiovascular System\t\n8. Respiratory System\t\n9. Gastrointestinal System\t\n10. Renal & Urinary System & Reproductive Systems\t\n11. Pregnancy, Childbirth & the Puerperium\t\n12. Endocrine System\t\n13. Multisystem Processes & Disorders\t\n14. Biostatistics & Epidemiology/Population Health/Interpretation of Medical Literature\t\n15. Social Sciences: Communication and Interpersonal Skills\t\n16. Social Sciences: Legal/Ethical Issues & Professionalism/Systems-based Practice & Patient Safety\n\nONLY RESPOND WITH THE TOPIC NUMBER (1-15). Do not include any other text in your response. \n'
"""USMLE organ-system label prompt (prompt_select_system_topic; 16 categories although the instruction
    says 1-15). o3-mini, reasoning effort medium, max 200 completion tokens, applied to the clean question and
    choices, 2025-03-22."""

LABEL_COMPETENCY = '\n\n## Question\n{question}\n\n## Choices\n{choices}\n\n1. Medical Knowledge/Scientific Concepts (Applying foundational science concepts)\n2. Patient Care: Diagnosis  \n3. Patient Care: Management\n4. Communication\n5. Professionalism, Including Legal and Ethical Issues\n6. Systems-based Practice, Including Patient Safety\n7. Practice-based Learning\n\nOnly respond with the topic number (1-7) that BEST fits with the medical question. Do not include any other text in your response. \n\n'
"""USMLE physician-competency label prompt (prompt_select_competency_topicV2; 7 categories).
    o3-mini, reasoning effort medium, max 200 completion tokens, applied to the clean question and choices,
    2025-03-27."""

PROMPTS = {"standard": PROMPT_STANDARD, "ignore": PROMPT_IGNORE, "structured": PROMPT_STRUCTURED}


def format_choices(choices: dict) -> str:
    """Render the option dict as ``[A] : text`` lines (original ``format_choices``)."""
    return "\n".join(f"[{k}] : {v}" for k, v in choices.items())


def build_prompt(question: str, choices: dict, variant: str = "standard", context: str = "") -> str:
    """Fill an evaluation template; ``context`` stays empty in all paper runs."""
    return PROMPTS[variant].format(question=question, choices=format_choices(choices), context=context)
