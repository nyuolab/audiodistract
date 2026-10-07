"""Matched single-note control for the frozen paired contamination judge.

Both conditions receive the same clean encounter and insertion target. Only one
generated note is shown. This does not regenerate notes or replace primary scores.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

KEYS = ['source_dataset', 'item_id', 'model', 'distractor_type', 'condition']
PROTOCOL = 'v4_single_note_matched_target'
MODEL = 'claude-sonnet-5'
SYSTEM = (
    'You are a senior attending physician auditing a generated clinical note. Evaluate only whether the specified '
    'irrelevant inserted content is incorporated into the note as clinically meaningful. Score the note '
    'independently and respond only with valid JSON.'
)
TEMPLATE = '''## Original Clinical Encounter
{clean_transcript}

## Known Irrelevant Inserted Content
{distractor_summary}

## Generated Note
{note}

The inserted content is supplied only as an evaluation target; do not infer that the note contains it. Evaluate the note independently. Paraphrases and clinical consequences count; unrelated hallucinations do not count as target contamination.

Respond only with:
{{"contamination": <0 or 1>, "severity": <0-3>, "reasoning": "<one concise explanation>"}}
'''


def prepare(notes: pd.DataFrame, pairs: pd.DataFrame):
    notes = notes.copy()
    pairs = pairs.copy()
    for df in [notes, pairs]:
        df['item_id'] = df['item_id'].astype(str)
    if notes.duplicated(KEYS).any():
        raise ValueError('Duplicate note keys')
    if not set(notes['condition']).issubset({'clean', 'distracted'}):
        raise ValueError('Unknown condition')
    lookup = ['source_dataset', 'item_id', 'distractor_type']
    merged = notes.merge(pairs[lookup + ['clean_transcript', 'distractor_summary']],
                         on=lookup, how='left', validate='many_to_one')
    if merged[['clean_transcript', 'distractor_summary', 'note']].isna().any().any():
        raise ValueError('Missing source transcript, target or note')
    jobs, metadata = [], []
    for row in merged.sort_values(KEYS).to_dict('records'):
        key = {k: str(row[k]) for k in KEYS}
        cid = 'n_' + hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:40]
        user = TEMPLATE.format(**row)
        params = {'model': MODEL, 'max_tokens': 800, 'system': SYSTEM,
                  'messages': [{'role': 'user', 'content': user}]}
        jobs.append({'custom_id': cid, 'params': params})
        metadata.append({**key, 'custom_id': cid,
                         'input_sha256': hashlib.sha256(user.encode()).hexdigest()})
    assert len({j['custom_id'] for j in jobs}) == len(jobs)
    return jobs, metadata


def parse_result(record):
    result = record['result']
    if result['type'] != 'succeeded':
        return {'parse_error': True, 'api_result_type': result['type'],
                'judge_raw': '', 'contamination': None, 'severity': None}
    message = result['message']
    raw = ''.join(c.get('text', '') for c in message['content'] if c['type'] == 'text')
    usage = message.get('usage', {})
    row = {'judge_raw': raw, 'api_result_type': 'succeeded',
           'response_model': message['model'], 'finish_reason': message['stop_reason'],
           'input_tokens': usage.get('input_tokens'), 'output_tokens': usage.get('output_tokens')}
    try:
        repair = None
        try:
            data = json.loads(raw[raw.index('{'):raw.rindex('}') + 1])
        except (ValueError, json.JSONDecodeError):
            # Recover only an otherwise complete JSON object lacking its final
            # brace. Values and rationale must already exist in the raw reply.
            candidate = raw.strip()
            if not candidate.startswith('{') or not candidate.endswith('"'):
                raise ValueError('Incomplete JSON payload')
            data = json.loads(candidate + '}')
            repair = 'appended_missing_closing_brace'
        c, s = data['contamination'], data['severity']
        assert type(c) is int and c in (0, 1)
        assert type(s) is int and 0 <= s <= 3
        assert isinstance(data['reasoning'], str)
        row.update(contamination=c, severity=s, judge_reasoning=data['reasoning'], parse_error=False,
                   parse_repair=repair)
    except (ValueError, KeyError, AssertionError, TypeError):
        row.update(contamination=None, severity=None, parse_error=True)
    return row
