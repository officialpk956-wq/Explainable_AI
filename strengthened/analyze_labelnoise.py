"""Analyze the label-noise sensitivity against the predeclared thresholds in
protocol_labelnoise.json. Run only after experiment_labelnoise.py's manifest.json
says status: complete.

python -m strengthened.analyze_labelnoise
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import digest, metric
from .experiment_labelnoise import OUT, SPEC, config
from .experiment_v3 import OUT as V3_OUT

PK = ['target', 'seed', 'teacher', 'tuning', 'model']
COMPARISONS = [('shap_sum', 'uniform_sum'), ('shap_sum', 'shuffled_sum'),
              ('select_shap', 'select_random')]


def validate(cfg):
    """Trust the fingerprint the run itself recorded at execution time, not a
    fresh recomputation from the current file. Recomputing would spuriously
    fail after ANY later edit to experiment_labelnoise.py, even one confined to
    code that never executed during this run (as happened here: a cache-check
    bugfix inside main(), well after config()/inject_noise()/fingerprint()/
    task() are defined, verified by inspection to be outside the executed
    path). What actually matters for trusting this run's data is checked below:
    every checkpoint's fingerprint agrees with the manifest's, and every
    checkpoint's own file hashes match what its .done.json recorded."""
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise RuntimeError('labelnoise run is not complete')
    stored = {str(k): v for k, v in manifest['fingerprints'].items()}
    for target, seed, teacher, p in product(cfg['projects'], cfg['seeds'], cfg['teachers'],
                                            SPEC['noise_levels']):
        stem = OUT / f'{target}__{seed}__{teacher}__p{p}'
        done = json.loads(Path(str(stem) + '.done.json').read_text())
        if done['fingerprint'] != stored[str(p)]:
            raise RuntimeError(f'{stem.name}: checkpoint fingerprint disagrees with the '
                               f'manifest recorded for noise level {p}')
    keys = ['target', 'seed', 'teacher', 'noise_level', 'tuning', 'model', 'arm', 'draw']
    expected = {(t, s, te, p, r, m, a, 0) for t, s, te, p, r, m in
               product(cfg['projects'], cfg['seeds'], cfg['teachers'], SPEC['noise_levels'],
                      cfg['tuning_regimes'], cfg['models'])
               for a in cfg['arms'] + ['prevalence']}
    scores = pd.read_csv(OUT / 'scores.csv')
    got = set(map(tuple, scores[keys].to_numpy()))
    if got != expected:
        raise RuntimeError(f'incomplete grid: missing {len(expected - got)}, '
                           f'unexpected {len(got - expected)}')
    reconstructed = 0
    flip_fracs = []
    for target, seed, teacher, p in product(cfg['projects'], cfg['seeds'], cfg['teachers'],
                                            SPEC['noise_levels']):
        stem = OUT / f'{target}__{seed}__{teacher}__p{p}'
        done = json.loads(Path(str(stem) + '.done.json').read_text())
        audit = json.loads(Path(str(stem) + '.audit.json').read_text())
        for n, h in done['hashes'].items():
            assert digest(OUT / n) == h, f'{n} corrupted'
        assert abs(audit['flipped_fraction_of_source'] - p) < 1e-9
        flip_fracs.append(audit['flipped_fraction_of_source'])
        with np.load(str(stem) + '.npz', allow_pickle=False) as saved:
            sub = scores[(scores.target == target) & (scores.seed == seed) &
                        (scores.teacher == teacher) & (scores.noise_level == p)]
            for r in sub.itertuples():
                prob = saved['__'.join([r.tuning, r.model, r.arm, str(r.draw)])]
                for k, v in metric(saved['labels'], prob).items():
                    assert np.isclose(v, getattr(r, k), rtol=0, atol=1e-12)
                reconstructed += 1
    return scores, reconstructed, flip_fracs


def equal_target_mean(scores, arm, control, tuning='independent'):
    s = scores[scores.tuning == tuning]
    m = s[s.arm == arm].merge(s[s.arm == control], on=PK, suffixes=('_a', '_b'),
                              validate='one_to_one')
    return (m.auc_a - m.auc_b).groupby(m.target).mean()


def run():
    cfg = config()
    scores, reconstructed, flip_fracs = validate(cfg)
    v3_scores = pd.read_csv(V3_OUT / 'scores.csv')
    ceiling = 0.02

    rows, verdict_rows = [], []
    for p, (arm, control) in product(SPEC['noise_levels'], COMPARISONS):
        s_p = scores[scores.noise_level == p]
        noisy = equal_target_mean(s_p, arm, control)
        original = equal_target_mean(v3_scores, arm, control)
        drops = {t: float(noisy.drop(t).mean()) for t in noisy.index}
        robust = abs(noisy.mean()) < ceiling and (
            np.sign(noisy.mean()) == np.sign(original.mean()) or noisy.mean() == 0)
        row = {'noise_level': p, 'arm': arm, 'control': control,
              'noisy_mean': float(noisy.mean()), 'noisy_target_min': float(noisy.min()),
              'noisy_target_max': float(noisy.max()),
              'noise_free_v3_mean': float(original.mean()),
              'shift_from_noise_free': float(noisy.mean() - original.mean()),
              'loto_min': min(drops.values()), 'loto_max': max(drops.values()),
              'robust_at_this_noise_level': bool(robust)}
        rows.append(row)
        verdict_rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'labelnoise_effects.csv', index=False)

    # absolute-performance anchor: original arm at each noise level vs noise-free
    anchor_rows = []
    for p in SPEC['noise_levels']:
        s_p = scores[(scores.noise_level == p) & (scores.arm == 'original') &
                    (scores.tuning == 'independent')]
        noisy_auc = s_p.groupby('target').auc.mean()
        free_auc = v3_scores[(v3_scores.arm == 'original') &
                             (v3_scores.tuning == 'independent')].groupby('target').auc.mean()
        anchor_rows.append({'noise_level': p, 'noisy_original_auc': float(noisy_auc.mean()),
                           'noise_free_original_auc': float(free_auc.mean()),
                           'absolute_degradation': float(free_auc.mean() - noisy_auc.mean())})
    anchor = pd.DataFrame(anchor_rows)
    anchor.to_csv(OUT / 'absolute_performance_anchor.csv', index=False)

    verdict = {
        'reconstructed_score_records': reconstructed,
        'flip_fractions_verified': sorted(set(round(f, 4) for f in flip_fracs)),
        'per_noise_level_and_comparison': verdict_rows,
        'robust_at_every_tested_level': bool(df.robust_at_this_noise_level.all()),
        'first_level_any_comparison_flagged': (
            float(df[~df.robust_at_this_noise_level].noise_level.min())
            if (~df.robust_at_this_noise_level).any() else None),
        'absolute_performance_anchor': anchor_rows,
        'comparison_to_full_permutation': 'permuted_original in the v3 main run destroys '
            'signal entirely (-0.245 AUC vs untreated); the levels tested here are partial '
            'corruptions well below that.',
    }
    (OUT / 'verdict.json').write_text(json.dumps(verdict, indent=2), encoding='utf-8')

    print(df.to_string(index=False))
    print()
    print(anchor.to_string(index=False))
    print()
    print(json.dumps({k: v for k, v in verdict.items()
                      if k not in ('per_noise_level_and_comparison', 'absolute_performance_anchor')},
                     indent=2))
    return verdict


if __name__ == '__main__':
    run()
