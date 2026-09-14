"""Analyze the LIME-attribution run against the original SHAP/tree-teacher v3
effects. Run only after experiment_lime_teacher.py's manifest.json says
status: complete.

python -m strengthened.analyze_lime_teacher
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import digest, metric
from .experiment_lime_teacher import OUT, COMPARISONS, config
from .experiment_v3 import OUT as V3_OUT

PK = ['target', 'seed', 'teacher', 'tuning', 'model']
# maps this run's lime_* arm names to the tree/SHAP run's shap_* names for comparison
SHAP_EQUIVALENT = {'lime_sum': 'shap_sum', 'shuffled_lime': 'shuffled_sum',
                   'select_lime': 'select_shap', 'uniform_sum': 'uniform_sum',
                   'select_random': 'select_random'}


def validate(cfg):
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise RuntimeError('lime-teacher run is not complete')
    stored = manifest['fingerprint']
    keys = ['target', 'seed', 'teacher', 'tuning', 'model', 'arm', 'draw']
    expected = {(t, s, 'RandomForest', r, m, a, 0) for t, s, r, m in
               product(cfg['projects'], cfg['seeds'], cfg['tuning_regimes'], cfg['models'])
               for a in cfg['arms']}
    scores = pd.read_csv(OUT / 'scores.csv')
    got = set(map(tuple, scores[keys].to_numpy()))
    if got != expected:
        raise RuntimeError(f'incomplete grid: missing {len(expected - got)}, '
                           f'unexpected {len(got - expected)}')
    reconstructed = 0
    for target, seed in product(cfg['projects'], cfg['seeds']):
        stem = OUT / f'{target}__{seed}__RandomForest'
        done = json.loads(Path(str(stem) + '.done.json').read_text())
        if done['fingerprint'] != stored:
            raise RuntimeError(f'{stem.name}: checkpoint fingerprint disagrees with manifest')
        for n, h in done['hashes'].items():
            assert digest(OUT / n) == h, f'{n} corrupted'
        with np.load(str(stem) + '.npz', allow_pickle=False) as saved:
            sub = scores[(scores.target == target) & (scores.seed == seed)]
            for r in sub.itertuples():
                prob = saved['__'.join([r.tuning, r.model, r.arm, str(r.draw)])]
                for k, v in metric(saved['labels'], prob).items():
                    assert np.isclose(v, getattr(r, k), rtol=0, atol=1e-12)
                reconstructed += 1
    return scores, reconstructed


def equal_target_mean(scores, arm, control, tuning='independent'):
    s = scores[scores.tuning == tuning]
    m = s[s.arm == arm].merge(s[s.arm == control], on=PK, suffixes=('_a', '_b'),
                              validate='one_to_one')
    return (m.auc_a - m.auc_b).groupby(m.target).mean()


def run():
    cfg = config()
    scores, reconstructed = validate(cfg)
    v3_scores = pd.read_csv(V3_OUT / 'scores.csv')

    rows = []
    for arm, control in COMPARISONS:
        lime = equal_target_mean(scores, arm, control)
        shap_arm, shap_control = SHAP_EQUIVALENT[arm], SHAP_EQUIVALENT[control]
        shap = equal_target_mean(v3_scores, shap_arm, shap_control)
        drops = {t: float(lime.drop(t).mean()) for t in lime.index}
        shap_drops = {t: float(shap.drop(t).mean()) for t in shap.index}
        rows.append({
            'arm': arm, 'control': control,
            'lime_mean': float(lime.mean()), 'lime_target_min': float(lime.min()),
            'lime_target_max': float(lime.max()),
            'shap_equivalent': f'{shap_arm} vs {shap_control}',
            'shap_tree_teacher_mean': float(shap.mean()),
            'sign_matches_shap': bool(np.sign(lime.mean()) == np.sign(shap.mean())
                                      or lime.mean() == 0),
            'lime_loto_min': min(drops.values()), 'lime_loto_max': max(drops.values()),
            'lime_loto_always_positive': bool(min(drops.values()) > 0),
            'shap_loto_min': min(shap_drops.values()), 'shap_loto_max': max(shap_drops.values()),
            'shap_loto_always_positive': bool(min(shap_drops.values()) > 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'lime_teacher_effects.csv', index=False)

    verdict = {
        'reconstructed_score_records': reconstructed,
        'per_comparison': rows,
        'mean_sign_matches_shap_for_all_comparisons': bool(df.sign_matches_shap.all()),
        'loto_robust_for_all_comparisons': bool(df.lime_loto_always_positive.all()),
        'comparisons_not_loto_robust': df[~df.lime_loto_always_positive][
            ['arm', 'control']].to_dict('records'),
        'note': ('Descriptive comparison: same RandomForest teacher as the main v3 run, '
                'attribution mechanism swapped from shap.TreeExplainer to LIME. No '
                'preregistered threshold -- first measurement of this comparison, '
                'exploratory like v3 itself. BreakDown was not implemented: no maintained '
                'Python package was installable in this environment.'),
    }
    (OUT / 'verdict.json').write_text(json.dumps(verdict, indent=2), encoding='utf-8')

    print(df.to_string(index=False))
    print()
    print(json.dumps({k: v for k, v in verdict.items() if k != 'per_comparison'}, indent=2))
    return verdict


if __name__ == '__main__':
    run()
