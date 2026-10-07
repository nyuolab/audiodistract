"""Second-generation inserted exchanges: stratified, rule-checked, diverse (September 2026).

The first insertion set collapsed onto a few scenarios (96% of bystander exchanges were a coworker's or neighbour's
pneumonia; three quarters of the nonliteral ones were the film Contagion or a cat named Lupus), and a sick contact is
something a clinician may legitimately record. This generator draws the content of every exchange from prespecified
lists so that it is irrelevant to the patient by construction:

* bystander: a named third party (relatives allowed) with a condition that is non-communicable, non-heritable and not
  environmental (injuries and elective procedures), and no bearing on the patient's care (no exposure, no shared
  medication, no symptom claim by the patient);
* nonliteral: a medical word used in a clearly non-clinical sense (a film, a pet's name, a band, an idiom), with the
  patient never claiming the condition.

Per encounter and family the generator samples a slot (relation x condition, or nonliteral item) so that every item
appears about equally often, asks GPT-5.4 to write a two-to-four-turn exchange in the transcript's speaker labels, and
validates the result with rules (speaker tags, required content, forbidden exposure/family/first-person-symptom
phrases, length); a failing exchange is regenerated with a different sample of the same stratum.

    python -m llm_distract.ambient.perturb_v2 --distractor bystander --out data/ambient/insertions_v2.parquet
    python -m llm_distract.ambient.perturb_v2 --distractor nonliteral --out data/ambient/insertions_v2.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from llm_distract.ambient.data import build_cohort, speaker_tags

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

RELATIVES = ["my sister", "my brother", "my aunt", "my uncle", "my cousin", "my mother-in-law"]
NON_RELATIVES = ["a coworker", "a neighbour", "a friend", "a friend's father", "my son's teacher", "our mail carrier", "someone from my gym",
                 "my barber", "a guy on my softball team", "my landlord"]
# body system -> (topic, keyword stems for validation, phrase for the writer, relatives_ok). Conditions are non-communicable;
# those with any heritable component are given only to non-relatives so that nothing reads as family history.
CONDITIONS = {
    "musculoskeletal": [
        ("knee replacement", ("knee replacement", "new knee"), "had a knee replacement six weeks ago and is doing physiotherapy twice a week", True),
        ("rotator cuff tear", ("rotator cuff",), "tore a rotator cuff playing tennis and is waiting for surgery", True),
        ("meniscus tear", ("meniscus",), "tore a meniscus and had an arthroscopy last month", True),
        ("sciatica", ("sciatica",), "has been off work with sciatica and was given naproxen and a physio referral", True),
        ("frozen shoulder", ("frozen shoulder",), "is getting steroid injections for a frozen shoulder", True),
        ("plantar fasciitis", ("plantar", "heel"), "has plantar fasciitis from running and now wears orthotics", True),
        ("wrist fracture", ("wrist",), "broke a wrist slipping on ice and has a cast for six weeks", True),
        ("hip replacement", ("hip replacement", "new hip"), "had a hip replacement and is walking with a cane for now", True),
        ("bunion surgery", ("bunion",), "had bunion surgery and has to wear a boot", True),
        ("whiplash", ("whiplash",), "has whiplash after a minor car accident and is doing neck exercises", True),
    ],
    "cardiovascular": [
        ("pacemaker", ("pacemaker",), "just got a pacemaker for a slow heart rhythm", False),
        ("coronary stent", ("stent",), "had a stent put in last month and is on cardiac rehab", False),
        ("varicose vein surgery", ("varicose",), "had varicose vein surgery and wears compression stockings", True),
        ("atrial fibrillation ablation", ("ablation", "fibrillation"), "had an ablation for atrial fibrillation", False),
        ("blood pressure medication change", ("blood pressure",), "had their blood pressure medication changed to amlodipine", False),
        ("heart valve repair", ("valve",), "had a heart valve repaired and is recovering well", False),
    ],
    "respiratory": [
        ("nasal polyp surgery", ("polyp",), "had nasal polyps removed and can finally breathe through the nose", True),
        ("deviated septum repair", ("septum",), "had a deviated septum repaired", True),
        ("collapsed lung after rib fracture", ("collapsed lung", "chest tube"), "had a collapsed lung after breaking a rib and needed a chest tube", True),
        ("sleep apnoea machine", ("apnea", "apnoea", "cpap"), "just started on a CPAP machine for sleep apnoea", False),
        ("asthma inhaler switch", ("inhaler", "asthma"), "was switched to a new asthma inhaler", False),
        ("vocal cord nodule", ("vocal cord", "nodule"), "had a vocal cord nodule removed and is on voice rest", True),
    ],
    "gastrointestinal": [
        ("gallbladder removal", ("gallbladder",), "had their gallbladder out laparoscopically", False),
        ("appendectomy", ("appendix", "appendectomy"), "had an appendectomy two weeks ago", True),
        ("hernia repair", ("hernia",), "just had an inguinal hernia repaired with mesh", True),
        ("colonoscopy with polyp removal", ("colonoscopy", "polyp"), "had a colonoscopy and a benign polyp removed", False),
        ("reflux medication", ("reflux", "omeprazole"), "was started on omeprazole for reflux", False),
        ("haemorrhoid surgery", ("hemorrhoid", "haemorrhoid"), "had haemorrhoid surgery and is complaining about the recovery", True),
    ],
    "neurological": [
        ("concussion from a bike fall", ("concussion",), "had a mild concussion after falling off a bike and is off screens for a week", True),
        ("carpal tunnel surgery", ("carpal tunnel",), "had carpal tunnel surgery on the right hand", True),
        ("migraine treatment", ("migraine",), "was started on a new preventive tablet for migraines", False),
        ("Bell's palsy", ("bell's palsy", "bells palsy", "facial"), "had Bell's palsy that is slowly resolving with steroids", False),
        ("sciatica", ("sciatica",), "has been off work with sciatica and was given naproxen and a physio referral", True),
        ("pinched nerve in the neck", ("pinched nerve", "neck"), "has a pinched nerve in the neck and is doing traction at physio", True),
    ],
    "genitourinary": [
        ("kidney stone", ("kidney stone",), "passed a kidney stone last week after two days in agony", False),
        ("vasectomy", ("vasectomy",), "had a vasectomy and took a week off", True),
        ("lithotripsy", ("lithotripsy",), "had lithotripsy for a stone that would not pass", False),
        ("enlarged prostate medication", ("prostate",), "was started on tamsulosin for an enlarged prostate", False),
        ("bladder sling surgery", ("bladder", "sling"), "had a bladder sling operation", False),
    ],
    "endocrine": [
        ("thyroid nodule removal", ("thyroid",), "had a benign thyroid nodule removed", False),
        ("type 2 diabetes medication change", ("diabetes", "metformin"), "had their diabetes tablets changed to metformin", False),
        ("gout flare", ("gout",), "had a gout flare after a barbecue and is on allopurinol now", False),
        ("vitamin D deficiency", ("vitamin d",), "was told they are low in vitamin D and takes drops", False),
        ("thyroid medication adjustment", ("levothyroxine", "thyroid"), "had their levothyroxine dose adjusted", False),
    ],
    "dermatological": [
        ("mole removal", ("mole",), "had a mole removed from the back and it was benign", True),
        ("eczema cream", ("eczema",), "was given a new steroid cream for eczema", False),
        ("psoriasis treatment", ("psoriasis",), "started a new treatment for psoriasis", False),
        ("cyst removal", ("cyst",), "had a sebaceous cyst removed from the neck", True),
        ("wart freezing", ("wart",), "had a plantar wart frozen off", True),
    ],
    "psychiatric": [
        ("anxiety counselling", ("anxiety", "counsel", "therap"), "started counselling for anxiety and says it helps", False),
        ("insomnia treatment", ("insomnia", "sleep"), "is doing a sleep programme for insomnia", False),
        ("ADHD medication", ("adhd",), "was started on medication for ADHD", False),
        ("grief counselling", ("grief", "counsel"), "is seeing a grief counsellor after losing a pet", True),
    ],
    "ent_ophthalmic_dental": [
        ("cataract surgery", ("cataract",), "had cataract surgery and can finally read the paper again", True),
        ("tonsillectomy", ("tonsil",), "had their tonsils out and lived on ice cream for a week", True),
        ("hearing aid fitting", ("hearing aid",), "just got fitted for hearing aids", False),
        ("root canal", ("root canal",), "needed a root canal and complained about the bill", True),
        ("LASIK", ("lasik", "laser eye"), "had laser eye surgery and no longer needs glasses", True),
        ("ear tubes", ("ear tube", "grommet"), "had ear tubes put in as a child", True),
    ],
    "hematologic_oncologic": [
        ("iron deficiency anaemia", ("iron", "anemia", "anaemia"), "was found to be low in iron and takes tablets now", False),
        ("basal cell skin cancer removal", ("basal cell",), "had a basal cell skin cancer removed from the nose", False),
        ("benign lymph node biopsy", ("lymph node", "biopsy"), "had a lymph node biopsy that came back benign", False),
        ("blood donation", ("blood donation", "donate"), "donates blood every few months and jokes about the biscuits", True),
    ],
    "obstetric_gynecologic": [
        ("fibroid surgery", ("fibroid",), "had fibroids removed and is recovering", False),
        ("endometriosis treatment", ("endometriosis",), "is being treated for endometriosis", False),
        ("pregnancy nausea", ("pregnan", "morning sickness"), "is pregnant and struggling with morning sickness", True),
        ("IUD fitting", ("iud", "coil"), "just had an IUD fitted", True),
    ],
}
CONDITIONS["general"] = sum(CONDITIONS.values(), [])
# nonliteral items -> the body systems whose visits they suit (empty = any)
NONLITERAL_ITEMS = [
    ("the film Contagion", ("contagion",), "the film Contagion, watched as a movie", ["respiratory", "general"]),
    ("a cat named Lupus", ("lupus",), "a cat named Lupus", ["musculoskeletal", "dermatological", "general"]),
    ("the band The Strokes", ("strokes",), "the band The Strokes", ["neurological", "cardiovascular", "general"]),
    ("the zodiac sign Cancer", ("cancer",), "the zodiac sign Cancer (astrology, not the disease)", ["hematologic_oncologic", "dermatological", "general"]),
    ("a stock market 'stroke'", ("stroke",), "a stock market that 'had a stroke' this week (a joke about prices)", ["neurological", "cardiovascular"]),
    ("a car with an 'arrhythmia'", ("arrhythmia",), "an old car whose engine has 'an arrhythmia' (it stalls)", ["cardiovascular"]),
    ("a sailboat named Vertigo", ("vertigo",), "a sailboat named Vertigo", ["neurological", "ent_ophthalmic_dental"]),
    ("the film Concussion", ("concussion",), "the film Concussion, discussed as a movie", ["neurological", "musculoskeletal"]),
    ("the album Hysteria", ("hysteria",), "the album Hysteria by Def Leppard", ["psychiatric", "general"]),
    ("a sourdough starter 'in remission'", ("remission",), "a sourdough starter that 'went into remission' (stopped rising)", ["hematologic_oncologic", "gastrointestinal"]),
    ("a novel with a 'fractured' plot", ("fractur",), "a novel with a 'fractured' plot", ["musculoskeletal"]),
    ("a horse named Migraine", ("migraine",), "a racehorse named Migraine", ["neurological"]),
    ("a coffee shop called Delirium", ("delirium",), "a coffee shop called Delirium", ["psychiatric", "neurological"]),
    ("a video game called Outbreak", ("outbreak",), "a video game called Outbreak", ["respiratory", "dermatological", "general"]),
    ("a puppy named Tremor", ("tremor",), "a puppy named Tremor", ["neurological", "endocrine"]),
    ("a garden 'in shock' after frost", ("shock",), "a garden that 'went into shock' after a late frost", ["cardiovascular", "general"]),
    ("a laptop with 'chronic fatigue'", ("fatigue",), "an old laptop with 'chronic fatigue' (the battery dies)", ["endocrine", "hematologic_oncologic", "psychiatric"]),
    ("a football team's 'hernia' of a defence", ("hernia",), "a football team whose defence 'has a hernia' (it keeps giving way)", ["gastrointestinal", "musculoskeletal"]),
    ("a plant nursery called Colic", ("colic",), "a plant nursery called Colic", ["gastrointestinal", "obstetric_gynecologic"]),
    ("a rowing boat named Cataract", ("cataract",), "a rowing boat named Cataract", ["ent_ophthalmic_dental"]),
    ("a bakery called Sweet Tooth Cavity", ("cavity",), "a bakery called Cavity", ["ent_ophthalmic_dental"]),
    ("a band called Kidney Stone", ("kidney stone",), "a garage band called Kidney Stone", ["genitourinary"]),
    ("a racehorse named Palpitation", ("palpitation",), "a racehorse named Palpitation", ["cardiovascular"]),
    ("a sourdough 'going into labour'", ("labour", "labor"), "a bread dough that 'went into labour' overnight (it rose)", ["obstetric_gynecologic"]),
]
STYLES = ["small talk while the clinician types", "a light joke", "a passing remark while changing subject", "a comment about the weekend",
          "an aside while the patient settles in"]

SYSTEM = ("You write short, realistic asides for doctor-patient transcripts. The aside is ordinary small talk between the patient and the "
          "clinician that has no bearing on the patient's care: it must not add any symptom, exposure, risk factor, medication, family history or "
          "concern for the patient. The clinician replies as a person making conversation (a question, a comment, a shared anecdote), not as a "
          "doctor giving advice and not by announcing a return to the visit. Return only valid JSON.")
USER = """\
Insert a short irrelevant mini conversation into this clinical transcript.

Content to use (use it, and nothing clinical beyond it): {content}
Style: {style}
Distractor type: {distractor_type}

Rules:
- {family_rule}
- The patient, or the family member or colleague accompanying them, brings it up: the first turn of the aside is theirs, and the clinician never introduces it.
- The patient must not report, suspect or worry about any symptom, condition, exposure or risk of their own, and must not ask for advice about it.
- Do not use the words contagious, caught, exposed, spread, infection, genetic, hereditary, "runs in the family", "me too" or "same thing".
- Use only these speaker tags exactly as written: {speaker_tags}.
- Match the transcript's tone and speaker-tag format. Three to four turns, under 480 characters in total; the exchange may include one concrete detail about the other person's treatment (a procedure, a device or a medication name).
- The clinician does not comment on the patient's own health, does not offer advice, and does not say anything like "back to you" or "let's get back to".
- Do not rewrite or summarize the original transcript.
- Pick an insertion point that feels natural and does not interrupt a tightly connected medical explanation.
- Return JSON with exactly these keys:
  "distractor_topic": short topic phrase,
  "distractor_conversation": the mini conversation with speaker tags,
  "distractor_summary": one sentence describing the irrelevant content and who it is about,
  "insertion_after_turn": integer, 1 means after the first speaker turn.

Transcript:
{transcript}
"""
FAMILY_RULE = {
    "bystander": "The condition belongs to the named third party only, it is already being handled by their own doctors, and nothing about it applies to the patient.",
    "nonliteral": "The medical word appears only in its non-clinical sense (a title, a name, a figure of speech); nobody in the exchange has or fears the condition.",
}
FORBIDDEN = re.compile(r"\b(contagious|caught|catch(ing)?|exposed|exposure|spread(ing)?|infect\w*|genetic|hereditary|runs in (the|our|my) family|me too|same thing|"
                       r"should I|do I need|could I have|worried (that )?I|I (have|had|get|got|feel|felt)( a| an| the| some| my| that)? [a-z]* ?(pain|ache|sore|symptom|flare|swelling|numb|stiff))\b", re.I)
PATIENT_BODY = re.compile(r"\b(my|our) (own )?(knee|shoulder|wrist|back|ankle|elbow|foot|feet|heel|neck|tonsils|appendix|hand|hip|chest|heart|stomach|belly|gut|head|eye|ear|throat|skin|bladder|kidney|thyroid|nose|sinus|lung|breathing|blood pressure|sugar|period|pregnancy|mood|sleep|nerve|migraine|headache)s?\b", re.I)


def content_for(distractor_type: str, k: int, rng: random.Random, system: str = "general") -> tuple[str, tuple, str]:
    """Deterministic, balanced slot assignment within the encounter's body system: item k cycles through the system's list
    in a seeded shuffled order; relatives are used only for conditions flagged relatives_ok."""
    if distractor_type == "bystander":
        conds = CONDITIONS.get(system, CONDITIONS["general"])[:]
        rng.shuffle(conds)
        topic, stems, phrase, relatives_ok = conds[k % len(conds)]
        pool = (RELATIVES + NON_RELATIVES) if relatives_ok else NON_RELATIVES
        relation = pool[(k // len(conds) + k) % len(pool)]
        return f"{topic} ({relation}; {system})", stems, f"{relation} {phrase} (said by the patient about that person)"
    items = [it for it in NONLITERAL_ITEMS if system in it[3]] or NONLITERAL_ITEMS
    items = items[:]
    rng.shuffle(items)
    topic, stems, desc, _ = items[k % len(items)]
    return f"{topic} ({system})", stems, desc


def validate(conv: str, distractor_type: str, stems: tuple, tags: tuple) -> str | None:
    lines = [ln.strip() for ln in conv.splitlines() if ln.strip()]
    if not 3 <= len(lines) <= 4:
        return "turn count"
    if not all(ln.startswith(tags) for ln in lines):
        return "speaker tag"
    if lines[0].lower().startswith(("doctor", "[doctor")) and any(not t.lower().startswith(("doctor", "[doctor")) for t in tags):
        return "clinician opens the aside"
    if len(conv) > 560:
        return "too long"
    if re.search(r"\b(back to (you|your|the visit|where we were)|let'?s get back|anyway,? (so|about) your|returning to)\b", conv, re.I):
        return "redirect cue"
    low = conv.lower()
    if not any(s.lower() in low for s in stems):
        return "assigned content missing"
    if FORBIDDEN.search(conv):
        return "forbidden phrase: " + FORBIDDEN.search(conv).group(0)
    patient_lines = [ln for ln in lines if ln.lower().startswith(("patient", "[patient", "guest", "[guest"))]
    if distractor_type == "bystander" and any(PATIENT_BODY.search(ln) for ln in patient_lines):
        return "patient refers to own body part"
    if distractor_type == "nonliteral" and re.search(r"\b(i|i'm|i've|i have|i had)\b[^.]{0,30}\b(stroke|cancer|lupus|migraine|concussion|arrhythmia|vertigo|hysteria|delirium|tremor|outbreak|shock|fractur\w*)\b", " ".join(patient_lines), re.I):
        return "patient claims the condition"
    return None


def generate_one(transcript: str, distractor_type: str, k: int, model: str, seed: int = 0, retries: int = 4, system: str = "general") -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tags = tuple(f"{t}:" for t in speaker_tags(transcript))
    style_rng = random.Random(f"{seed}-{distractor_type}-{k}")
    last = ""
    for attempt in range(retries * 3):
        # after ``retries`` failures move to the next slot of the family (a different condition or item) so that one
        # stubborn combination cannot stall the run; the assigned content is recorded with the exchange
        rng = random.Random(f"{seed}-{distractor_type}-{system}")
        topic, stems, content = content_for(distractor_type, k + (attempt // retries) * 7, rng, system)
        style = style_rng.choice(STYLES)
        prompt = USER.format(content=content, style=style, distractor_type=distractor_type, family_rule=FAMILY_RULE[distractor_type],
                             speaker_tags=", ".join(tags), transcript=transcript)
        r = client.chat.completions.create(model=model, temperature=0.7 if attempt else 0.3, max_completion_tokens=512,
                                           response_format={"type": "json_object"},
                                           messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        text = r.choices[0].message.content or ""
        try:
            j = json.loads(text) if text.strip().startswith("{") else json.loads(JSON_RE.search(text).group())
            conv = "\n".join(ln.strip() for ln in str(j["distractor_conversation"]).splitlines() if ln.strip())
            problem = validate(conv, distractor_type, stems, tags)
            if problem:
                last = problem
                continue
            return {"distractor_topic": topic, "distractor_conversation": conv, "distractor_summary": str(j["distractor_summary"]).strip(),
                    "insert_after_turn": max(1, int(j.get("insertion_after_turn", 1))), "assigned_content": content, "style": style,
                    "attempts": attempt + 1, "body_system": system}
        except Exception as exc:  # noqa: BLE001
            last = f"parse: {exc}"
            time.sleep(1.5 ** attempt)
    raise RuntimeError(f"no valid exchange after {retries * 3} attempts ({last})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="data/ambient/sources")
    ap.add_argument("--distractor", required=True, choices=["nonliteral", "bystander"])
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--systems", default=None, help="encounter_systems.parquet from label_systems: adjacent design (same body system as the visit)")
    args = ap.parse_args()
    cohort = build_cohort(args.sources).reset_index(drop=True)
    if args.limit:
        cohort = cohort.head(args.limit)
    out = Path(args.out)
    rows = pd.read_parquet(out).to_dict("records") if out.exists() else []
    done = {(r["source_dataset"], str(r["item_id"]), r["distractor_type"]) for r in rows}
    systems = {}
    if args.systems:
        sysdf = pd.read_parquet(args.systems)
        systems = {(a, str(b)): c for a, b, c in zip(sysdf["source_dataset"], sysdf["item_id"], sysdf["body_system"])}
    counters: dict = {}
    todo = []
    for _, r in cohort.iterrows():
        key = (r["source_dataset"], str(r["item_id"]))
        system = systems.get(key, "general")
        if system == "general" and args.systems:
            system = random.Random(f"{args.seed}-{key}").choice([x for x in CONDITIONS if x != "general"])  # no clinical content: any system
        k = counters.get(system, 0)
        counters[system] = k + 1  # balanced within each system
        if (key[0], key[1], args.distractor) not in done:
            todo.append((k, r, system))

    def work(item):
        k, r, system = item
        try:
            g = generate_one(r["clean_transcript"], args.distractor, k, args.model, args.seed, system=system)
        except RuntimeError as e:
            print(f"  FAILED {r['source_dataset']} {r['item_id']}: {e}", flush=True)
            return None
        return {"source_dataset": r["source_dataset"], "item_id": str(r["item_id"]), "distractor_type": args.distractor, **g,
                "generation_model": args.model, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for i, rec in enumerate(pool.map(work, todo), 1):
            if rec is not None:
                rows.append(rec)
            if i % 25 == 0 or i == len(todo):
                pd.DataFrame(rows).to_parquet(out, index=False)
                print(f"  {i}/{len(todo)}", flush=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    df = pd.DataFrame(rows)
    fam = df[df.distractor_type == args.distractor]
    print(f"{len(fam)} {args.distractor} insertions -> {out} | distinct topics {fam.distractor_topic.nunique()} | mean attempts {fam.attempts.mean():.2f}")


if __name__ == "__main__":
    main()
