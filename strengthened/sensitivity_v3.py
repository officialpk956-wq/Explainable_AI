"""Collect the four predeclared sensitivities into one comparable table.

python -m strengthened.sensitivity_v3

Each run is compared against the main run restricted to the same seeds, so the
contrast isolates the design change rather than the grid size.
"""
import json
from pathlib import Path

import pandas as pd

from .experiment import ROOT, dump
from .experiment_v3 import OUT, config

PK = ['target', 'seed', 'teacher', 'tuning', 'model']
METRICS = ['auc', 'average_precision', 'f1_macro']
HEADLINE = [('shap_sum', 'uniform_sum'), ('select_shap', 'select_random'),
            ('ps_aug', 'p_aug'), ('ps_aug', 'pr_aug'), ('pr_aug', 'p_aug'),
            ('pn_aug', 'p_aug'), ('original', 'prevalence'),
            ('permuted_original', 'original')]
RUNS = [('teacher_grouped', [101, 102, 103], 'Project-grouped teacher cross-fitting'),
        ('capacity', [101, 102, 103], '128 trees instead of 32'),
        ('control_rand1', [101], 'Control draws re-seeded (offset 1000)'),
        ('control_rand2', [101], 'Control draws re-seeded (offset 2000)')]


def equal_target_mean(scores, arm, control, tuning='independent'):
    s = scores[scores.tuning == tuning].groupby(PK + ['arm'], as_index=False)[METRICS].mean()
    m = s[s.arm == arm].merge(s[s.arm == control], on=PK, suffixes=('_a', '_b'),
                              validate='one_to_one')
    if not len(m):
        return None
    return float((m.auc_a - m.auc_b).groupby(m.target).mean().mean())


def collect():
    base = pd.read_csv(OUT / 'scores.csv')
    rows = []
    for folder, seeds, label in RUNS:
        path = OUT.parent / folder / 'scores.csv'
        if not path.exists():
            continue
        alt = pd.read_csv(path)
        ref = base[base.seed.isin(seeds)]
        for arm, control in HEADLINE:
            a, b = equal_target_mean(ref, arm, control), equal_target_mean(alt, arm, control)
            if a is None or b is None:
                continue
            rows.append({'sensitivity': folder, 'label': label, 'seeds': len(seeds),
                         'arm': arm, 'control': control, 'main': a, 'alternative': b,
                         'shift': b - a,
                         'sign_preserved': bool(a == 0 or b == 0 or (a > 0) == (b > 0))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'sensitivity.csv', index=False)

    # control-draw spread at a fixed source sample: the three control runs share seed 101
    spread = []
    fixed = [('full', 101), ('control_rand1', 101), ('control_rand2', 101)]
    frames = []
    for folder, seed in fixed:
        p = OUT / 'scores.csv' if folder == 'full' else OUT.parent / folder / 'scores.csv'
        if p.exists():
            frames.append(pd.read_csv(p).query('seed == @seed'))
    if len(frames) == 3:
        for arm, control in HEADLINE:
            vals = [equal_target_mean(f, arm, control) for f in frames]
            if any(v is None for v in vals):
                continue
            spread.append({'arm': arm, 'control': control, 'draws': len(vals),
                           'min': min(vals), 'max': max(vals), 'spread': max(vals) - min(vals)})
    pd.DataFrame(spread).to_csv(OUT / 'control_draw_spread.csv', index=False)

    dump(OUT / 'sensitivity_summary.json', {
        'runs': [r[0] for r in RUNS if (OUT.parent / r[0] / 'scores.csv').exists()],
        'comparisons': len(df),
        'sign_reversals': int((~df.sign_preserved).sum()),
        'reversed': df[~df.sign_preserved][['sensitivity', 'arm', 'control', 'main',
                                            'alternative']].to_dict('records')})
    print(df[['sensitivity', 'arm', 'control', 'main', 'alternative', 'shift',
              'sign_preserved']].to_string(index=False))
    return df


if __name__ == '__main__':
    collect()
