"""Compare matched single-note judgments with the frozen primary paired scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_distract.ambient.judge_controlled import KEYS, PROTOCOL
from llm_distract.common.style import MODEL_SHORT, FRONTIER_MODELS, set_style, PALETTE, NATURE_DOUBLE

N_BOOT = 5000
SEED = 20261002


def estimate(frame, col):
    cells = frame.groupby(['source_dataset', 'item_id'])[col].agg(['sum', 'count'])
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(cells), (N_BOOT, len(cells)))
    values = cells['sum'].to_numpy()[indices].sum(1) / cells['count'].to_numpy()[indices].sum(1)
    low, high = np.percentile(values * 100, [2.5, 97.5])
    return {'estimate': float(frame[col].mean() * 100), 'ci_low': float(low), 'ci_high': float(high)}


def agreement(frame):
    a, b = frame['paired'], frame['single']
    observed = float((a == b).mean())
    expected = float(a.mean() * b.mean() + (1-a.mean()) * (1-b.mean()))
    return {'n_notes': len(frame), 'agreement_pct': observed * 100,
            'kappa': (observed-expected)/(1-expected) if expected < 1 else None,
            'both_negative': int(((a == 0) & (b == 0)).sum()),
            'both_positive': int(((a == 1) & (b == 1)).sum()),
            'paired_only': int(((a == 1) & (b == 0)).sum()),
            'single_only': int(((a == 0) & (b == 1)).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=ROOT/'results/judge_control')
    ap.add_argument('--out', type=Path, default=ROOT/'results/judge_control_v4')
    args = ap.parse_args()
    primary_path = ROOT/'results/raw_v3/amb_judgments.parquet'
    single_path = args.run/'judgments.parquet'
    original, single = pd.read_parquet(primary_path), pd.read_parquet(single_path)
    assert len(original) == len(single) == 18432
    assert set(single.judge_protocol) == {PROTOCOL}
    assert not single.parse_error.any(), 'Resolve parse/API failures before analysis; never replace with zeros'
    for frame in [original, single]:
        frame['item_id'] = frame.item_id.astype(str)
        assert not frame.duplicated(KEYS).any()
    merged = original[KEYS+['contamination_v3']].merge(single[KEYS+['contamination']], on=KEYS,
                                                        validate='one_to_one', how='outer', indicator=True)
    assert merged['_merge'].eq('both').all()
    merged = merged.drop(columns='_merge').rename(columns={'contamination_v3': 'paired', 'contamination': 'single'})
    assert not merged[['paired', 'single']].isna().any().any()
    merged['model_name'] = merged.model.map(MODEL_SHORT)
    merged['cohort'] = np.where(merged.model_name.isin(FRONTIER_MODELS), 'frontier', 'open-weight')
    merged['single_minus_paired'] = merged.single-merged.paired
    args.out.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out/'per_note_comparison.parquet', index=False)
    rate_rows, agreement_rows, contrasts = [], [], []
    groups = [('cohort', c, g) for c, g in merged.groupby('cohort')]
    groups += [('model', c, g) for c, g in merged.groupby('model_name')]
    summary = {}
    for level, name, group in groups:
        agreement_rows.append({'level': level, 'name': name, 'condition': 'all', **agreement(group)})
        for condition, g in group.groupby('condition'):
            agreement_rows.append({'level': level, 'name': name, 'condition': condition, **agreement(g)})
        wide = group.pivot(index=KEYS[:-1], columns='condition', values=['single', 'paired'])
        wide.columns = ['_'.join(c) for c in wide.columns]
        wide = wide.reset_index()
        assert len(wide) == len(group)/2
        for protocol in ['paired', 'single']:
            wide[protocol+'_delta'] = wide[protocol+'_distracted']-wide[protocol+'_clean']
            for endpoint in ['clean', 'distracted', 'delta']:
                col = protocol+'_'+endpoint
                rate_rows.append({'level': level, 'name': name, 'protocol': protocol,
                                  'endpoint': endpoint, 'n_pairs': len(wide),
                                  'n_flagged': int(wide[col].sum()) if endpoint != 'delta' else None,
                                  **estimate(wide, col)})
        wide['contrast'] = wide.single_delta-wide.paired_delta
        contrast = {'level': level, 'name': name, 'endpoint': 'single minus paired increase', **estimate(wide, 'contrast')}
        contrasts.append(contrast)
        if level == 'cohort':
            summary[name] = {'n_pairs': len(wide), 'agreement': agreement(group),
                'by_condition': {c: agreement(g) for c, g in group.groupby('condition')},
                'paired': {e: estimate(wide, 'paired_'+e) for e in ['clean', 'distracted', 'delta']},
                'single': {e: estimate(wide, 'single_'+e) for e in ['clean', 'distracted', 'delta']},
                'single_minus_paired_delta': contrast}
    rates = pd.DataFrame(rate_rows)
    rates.to_csv(args.out/'rates.csv', index=False)
    pd.DataFrame(agreement_rows).to_csv(args.out/'agreement.csv', index=False)
    pd.DataFrame(contrasts).to_csv(args.out/'protocol_contrasts.csv', index=False)
    result = {'n_notes': len(merged), 'n_encounters': 576, 'n_boot': N_BOOT, 'seed': SEED,
              'protocol': json.loads((args.run/'protocol.json').read_text()),
              'completion': json.loads((args.run/'completion.json').read_text()),
              'cohorts': summary,
              'hashes': {'paired': hashlib.sha256(primary_path.read_bytes()).hexdigest(),
                         'single': hashlib.sha256(single_path.read_bytes()).hexdigest()}}
    (args.out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

    import matplotlib.pyplot as plt
    set_style()
    plt.rcParams['svg.fonttype'] = 'none'
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE*.85, 2.65), gridspec_kw={'wspace': .6})
    for ax, endpoint, ylabel in zip(axes, ['delta', 'clean'], ['Distracted minus clean flag rate (pp)', 'Clean notes flagged (%)']):
        for i, protocol in enumerate(['paired', 'single']):
            values = rates[(rates.level == 'cohort') & (rates.protocol == protocol) & (rates.endpoint == endpoint)].set_index('name').loc[['frontier', 'open-weight']]
            means = values.estimate.to_numpy()
            pos = np.array([0, 1]) + (i-.5)*.32
            ax.bar(pos, means, width=.3, color=PALETTE['slate' if i == 0 else 'grey'],
                   label='Paired' if i == 0 else 'Single note',
                   yerr=[means-values.ci_low.to_numpy(), values.ci_high.to_numpy()-means], capsize=2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Frontier', 'Open-weight'])
        ax.set_ylabel(ylabel); ax.set_ylim(0, 100); ax.set_yticks([0, 20, 40, 60, 80, 100])
        if endpoint == 'clean':
            for x, cohort in enumerate(['frontier', 'open-weight']):
                cells = rates[(rates.level == 'cohort') & (rates.name == cohort) & (rates.endpoint == 'clean')].set_index('protocol')
                labels = [f"{label}: {int(cells.loc[protocol, 'n_flagged'])}/4,608"
                          for protocol, label in [('paired', 'Paired'), ('single', 'Single')]]
                ax.text(x, 8, '\n'.join(labels), ha='center', va='bottom', fontsize=5.5)
    axes[0].legend(frameon=False, fontsize=6, loc='upper left')
    for label, ax in zip(['a', 'b'], axes):
        ax.text(-.18, 1.08, label, transform=ax.transAxes, fontweight='bold')
    fig.subplots_adjust(bottom=.2, top=.9)
    out = args.out / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(out/('ed_amb_judges.'+suffix), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
