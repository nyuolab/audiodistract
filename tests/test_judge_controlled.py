import pandas as pd
import pytest
from llm_distract.ambient.judge_controlled import prepare, parse_result


def test_same_target_and_clean_encounter_for_both_conditions_and_models():
    notes = pd.DataFrame([dict(source_dataset='s', item_id='1', distractor_type='bystander',
                               model=m, condition=c, note=n)
                          for m in ['model_a', 'model_b']
                          for c, n in [('clean', 'CONTROL_NOTE'), ('distracted', 'DISTRACTED_NOTE')]])
    pairs = pd.DataFrame([dict(source_dataset='s', item_id='1', distractor_type='bystander',
                               clean_transcript='CLEAN_SOURCE', distracted_transcript='DO_NOT_SHOW',
                               distractor_summary='EXACT_TARGET')])
    jobs, meta = prepare(notes, pairs)
    assert len(jobs) == len({r['custom_id'] for r in meta}) == 4
    for job in jobs:
        text = job['params']['messages'][0]['content']
        assert 'CLEAN_SOURCE' in text and 'EXACT_TARGET' in text
        assert 'DO_NOT_SHOW' not in text and 'model_a' not in text and 'model_b' not in text
        assert ('CONTROL_NOTE' in text) != ('DISTRACTED_NOTE' in text)
    with pytest.raises(ValueError, match='Duplicate'):
        prepare(pd.concat([notes, notes]), pairs)
    with pytest.raises(ValueError, match='Missing'):
        prepare(notes, pairs.assign(item_id='2'))


@pytest.mark.parametrize('raw', ['{"contamination": 2, "severity": 0, "reasoning": "x"}',
                                  '{"contamination": true, "severity": 0, "reasoning": "x"}',
                                  'truncated response'])
def test_invalid_scores_are_missing_not_zero(raw):
    record = {'result': {'type': 'succeeded', 'message': {'content': [{'type': 'text', 'text': raw}],
              'model': 'claude-sonnet-5', 'stop_reason': 'end_turn', 'usage': {}}}}
    row = parse_result(record)
    assert row['parse_error'] and row['contamination'] is None


def test_only_missing_final_json_brace_is_recovered_without_changing_values():
    raw = '{"contamination": 1, "severity": 3, "reasoning": "Complete rationale."'
    record = {'result': {'type': 'succeeded', 'message': {'content': [{'type': 'text', 'text': raw}],
              'model': 'claude-sonnet-5', 'stop_reason': 'end_turn', 'usage': {}}}}
    row = parse_result(record)
    assert not row['parse_error']
    assert row['contamination'] == 1 and row['severity'] == 3
    assert row['judge_raw'] == raw and row['judge_reasoning'] == 'Complete rationale.'
    assert row['parse_repair'] == 'appended_missing_closing_brace'
