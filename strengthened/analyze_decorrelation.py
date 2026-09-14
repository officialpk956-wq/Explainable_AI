"""Analyze the decorrelation sensitivity against the predeclared thresholds in
protocol_decorrelation.json. Run only after experiment_decorrelation.py's
manifest.json says status: complete.

python -m strengthened.analyze_decorrelation
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import digest
from .experiment_decorrelation import OUT, SPEC, config, fingerprint
from .experiment_v3 import OUT as V3_OUT

PK = ['target', 'seed', 'teacher', 'tuning', 'model']
COMPARISONS = [('shap_sum', 'uniform_sum'), ('shap_sum', 'shuffled_sum'),
              ('select_shap', 'select_random')]


def validate(cfg):
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['fingerprint'] != fingerprint(cfg):
        raise RuntimeError('decorrelation run is not complete or no longer matches the code')
    keys = PK + ['arm', 'draw']
    expected = {(t, s, te, r, m, a, 0) for t, s, te, r, m in
               product(cfg['projects'], cfg['seeds'], cfg['teachers'],
                      cfg['tuning_regimes'], cfg['models'])
               for a in cfg['arms'] + ['prevalence']}
    scores = pd.read_csv(OUT / 'scores.csv')
    got = set(map(tuple, scores[keys].to_numpy()))
    if got != expected:
        raise RuntimeError(f'incomplete grid: missing {len(expected - got)}, '
                           f'unexpected {len(got - expected)}')
    reconstructed = 0
    for target, seed, teacher in product(cfg['projects'], cfg['seeds'], cfg['teachers']):
        stem = OUT / f'{target}__{seed}__{teacher}'
        done = json.loads(Path(str(stem) + '.done.json').read_text())
        for n, h in done['hashes'].items():
            assert digest(OUT / n) == h, f'{n} corrupted'
        with np.load(str(stem) + '.npz', allow_pickle=False) as saved:
            sub = scores[(scores.target == target) & (scores.seed == seed) &
                        (scores.teacher == teacher)]
            for r in sub.itertuples():
                from .experiment import metric
                prob = saved['__'.join([r.tuning, r.model, r.arm, str(r.draw)])]
                for k, v in metric(saved['labels'], prob).items():
                    assert np.isclose(v, getattr(r, k), rtol=0, atol=1e-12)
                reconstructed += 1
    return scores, reconstructed


def equal_target_mean(scores, arm, control, tuning='independent'):
    s = scores[scores.tuning == tuning]
    m = s[s.arm == arm].merge(s[s.arm == control], on=PK, suffixes=('_a', '_b'),
                              validate='one_to_one')
    per_target = (m.auc_a - m.auc_b).groupby(m.target).mean()
    return per_target


def run():
    cfg = config()
    scores, reconstructed = validate(cfg)
    v3_scores = pd.read_csv(V3_OUT / 'scores.csv')

    rows = []
    for arm, control in COMPARISONS:
        decorrelated = equal_target_mean(scores, arm, control)
        original = equal_target_mean(v3_scores, arm, control)
        drops = {t: float(decorrelated.drop(t).mean()) for t in decorrelated.index}
        worst = max(drops, key=lambda t: abs(drops[t] - decorrelated.mean()))
        rows.append({
            'arm': arm, 'control': control,
            'decorrelated_mean': float(decorrelated.mean()),
            'decorrelated_target_min': float(decorrelated.min()),
            'decorrelated_target_max': float(decorrelated.max()),
            'original_54feature_mean': float(original.mean()),
            'shift_from_original': float(decorrelated.mean() - original.mean()),
            'sign_matches_original': bool(np.sign(decorrelated.mean()) ==
                                          np.sign(original.mean())
                                          or decorrelated.mean() == 0),
            'loto_min': min(drops.values()), 'loto_max': max(drops.values()),
            'most_influential_omission': worst,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'decorrelation_effects.csv', index=False)

    th = SPEC['predeclared_thresholds']
    ceiling = 0.02
    verdict_rows = []
    for r in df.itertuples():
        unchanged = abs(r.decorrelated_mean) < ceiling and r.sign_matches_original
        verdict_rows.append({'arm': r.arm, 'control': r.control,
                             'decorrelated_mean': r.decorrelated_mean,
                             'conclusion_unchanged': unchanged})
    sel = df[(df.arm == 'select_shap')].iloc[0]
    zero_signal_offset = 0.006056
    select_shap_extends = abs(sel.decorrelated_mean) <= zero_signal_offset

    verdict = {
        'reconstructed_score_records': reconstructed,
        'per_comparison': verdict_rows,
        'select_shap_zero_signal_extension_supported': bool(select_shap_extends),
        'select_shap_decorrelated_mean': float(sel.decorrelated_mean),
        'zero_signal_offset_reference': zero_signal_offset,
        'overall': 'conclusion unchanged for all three comparisons' if all(
            v['conclusion_unchanged'] for v in verdict_rows) else
            'at least one comparison is flagged as decorrelation-sensitive; see per_comparison',
    }
    (OUT / 'verdict.json').write_text(json.dumps(verdict, indent=2), encoding='utf-8')

    print(df.to_string(index=False))
    print()
    print(json.dumps(verdict, indent=2))
    return verdict


if __name__ == '__main__':
    run()
