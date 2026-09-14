"""Analyze the linear-teacher run against the original (tree-teacher) v3
effects. Run only after experiment_linear_teacher.py's manifest.json says
status: complete.

python -m strengthened.analyze_linear_teacher
"""
import json
from itertools import product

import numpy as np
import pandas as pd

from .experiment import digest, metric
from .experiment_linear_teacher import OUT, COMPARISONS, config
from .experiment_v3 import OUT as V3_OUT

PK = ['target', 'seed', 'teacher', 'tuning', 'model']


def validate(cfg):
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise RuntimeError('linear-teacher run is not complete')
    stored = manifest['fingerprint']
    keys = ['target', 'seed', 'teacher', 'tuning', 'model', 'arm', 'draw']
    expected = {(t, s, 'LinearLogit', r, m, a, 0) for t, s, r, m in
               product(cfg['projects'], cfg['seeds'], cfg['tuning_regimes'], cfg['models'])
               for a in cfg['arms']}
    scores = pd.read_csv(OUT / 'scores.csv')
    got = set(map(tuple, scores[keys].to_numpy()))
    if got != expected:
        raise RuntimeError(f'incomplete grid: missing {len(expected - got)}, '
                           f'unexpected {len(got - expected)}')
    reconstructed = 0
    for target, seed in product(cfg['projects'], cfg['seeds']):
        stem = OUT / f'{target}__{seed}__LinearLogit'
        from pathlib import Path
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
        linear = equal_target_mean(scores, arm, control)
        tree = equal_target_mean(v3_scores, arm, control)
        drops = {t: float(linear.drop(t).mean()) for t in linear.index}
        tree_drops = {t: float(tree.drop(t).mean()) for t in tree.index}
        rows.append({
            'arm': arm, 'control': control,
            'linear_teacher_mean': float(linear.mean()),
            'linear_teacher_target_min': float(linear.min()),
            'linear_teacher_target_max': float(linear.max()),
            'tree_teacher_mean': float(tree.mean()),
            'sign_matches_tree_teacher': bool(np.sign(linear.mean()) == np.sign(tree.mean())
                                              or linear.mean() == 0),
            'linear_loto_min': min(drops.values()), 'linear_loto_max': max(drops.values()),
            'linear_loto_always_positive': bool(min(drops.values()) > 0),
            'tree_loto_min': min(tree_drops.values()), 'tree_loto_max': max(tree_drops.values()),
            'tree_loto_always_positive': bool(min(tree_drops.values()) > 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'linear_teacher_effects.csv', index=False)

    verdict = {
        'reconstructed_score_records': reconstructed,
        'per_comparison': rows,
        'mean_sign_matches_tree_teacher_for_all_comparisons': bool(
            df.sign_matches_tree_teacher.all()),
        'loto_robust_for_all_comparisons': bool(df.linear_loto_always_positive.all()),
        'comparisons_not_loto_robust': df[~df.linear_loto_always_positive][
            ['arm', 'control']].to_dict('records'),
        'note': ('Descriptive replication check against the tree-teacher v3 main run. '
                'No preregistered threshold: this is the first measurement of this '
                'comparison, exploratory like v3 itself. The mean-sign check alone '
                'overstates replication for select_shap vs select_random: its LOTO range '
                'crosses zero under the linear teacher (tree teacher does not), so that '
                'comparison is not robustly positive even though its mean is.'),
    }
    (OUT / 'verdict.json').write_text(json.dumps(verdict, indent=2), encoding='utf-8')

    print(df.to_string(index=False))
    print()
    print(json.dumps({k: v for k, v in verdict.items() if k != 'per_comparison'}, indent=2))
    return verdict


if __name__ == '__main__':
    run()
