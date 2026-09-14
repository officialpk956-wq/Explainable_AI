"""Validate the v2 execution and derive its descriptive tables.

python -m strengthened.analyze_v2

Every reported number is reconstructed from the saved target predictions. The
aggregation is fixed by protocol_v2.json: average repeats, teachers and learners
inside each target, then take the equal-target mean.

Leave-one-target-out aggregation sensitivity drops one target from that mean. It
does not retrain anything and does not remove the dropped project from the source
pools of the remaining targets, so it is a sensitivity of the aggregate, not a
project-exclusion experiment. The resulting spread is not a confidence interval.
"""
from itertools import product, combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .experiment import load, digest, metric, dump
from .experiment_v2 import OUT, config, fingerprint

# (treatment, control, what the pair isolates)
COMPARISONS = [
    ('shap_sum', 'uniform_sum', 'attribution weighting beyond a matched-magnitude constant'),
    ('shap_sum', 'shuffled_sum', 'attribution weighting beyond a matched multiplier distribution'),
    ('select_shap', 'select_random', 'attribution selection beyond an equal-sized random subset'),
    ('ps_aug', 'p_aug', 'attribution columns beyond the prediction channel'),
    ('ps_aug', 'pn_aug', 'attribution columns beyond dimension-matched uninformative columns'),
    ('pn_aug', 'p_aug', 'the generic penalty for added uninformative dimensions'),
    ('p_aug', 'original', 'the prediction channel alone'),
    ('coral', 'original', 'a documented adaptation comparator (transductive)'),
    ('permuted_original', 'original', 'loss of all source label signal'),
    ('original', 'prevalence', 'source-only learning against a featureless reference'),
]
METRICS = ['auc', 'average_precision', 'f1_macro']


def check_oof(rep, n):
    seen = []
    for f in rep['oof']:
        tr, va = set(f['train']), set(f['validation'])
        assert not tr & va and tr | va == set(range(n))
        seen.extend(f['validation'])
    assert sorted(seen) == list(range(n))
    np.testing.assert_allclose(sorted(rep['weights']), sorted(rep['shuffled_weights']))
    assert len(rep['selected']) == len(rep['random_selected']) == int(np.ceil(len(rep['weights']) / 2))
    assert rep['augmentation_columns']['ps_aug'] == rep['augmentation_columns']['pn_aug']


def validate(cfg, scores):
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['smoke']:
        raise RuntimeError('A complete non-smoke execution is required')
    if manifest['fingerprint'] != fingerprint(cfg):
        raise RuntimeError('Execution fingerprint no longer matches the current code and protocol')
    keys = ['target', 'seed', 'teacher', 'tuning', 'model', 'arm']
    expected = set(product(cfg['projects'], cfg['seeds'], cfg['teachers'],
                           cfg['tuning_regimes'], cfg['models'], cfg['arms']))
    got = set(map(tuple, scores[keys].to_numpy()))
    if got != expected or scores.duplicated(keys).any():
        raise RuntimeError(f'Incomplete score grid: missing {len(expected - got)}, '
                           f'unexpected {len(got - expected)}, duplicates '
                           f'{int(scores.duplicated(keys).sum())}')

    reconstructed = 0
    diagnostics, weights = [], []
    for target, seed, teacher in product(cfg['projects'], cfg['seeds'], cfg['teachers']):
        stem = str(OUT / f'{target}__{seed}__{teacher}')
        done = json.loads(Path(stem + '.done.json').read_text())
        audit = json.loads(Path(stem + '.audit.json').read_text())
        assert done['fingerprint'] == manifest['fingerprint']
        for n, h in done['hashes'].items():
            assert digest(OUT / n) == h, f'{n} corrupted'

        assert set(audit['source_indices']) == set(cfg['projects']) - {target}
        labels, groups = [], []
        for source, idx in audit['source_indices'].items():
            _, y, _ = load(source, cfg)
            assert len(idx) == len(set(idx)) == min(len(y), cfg['source_cap_per_project'])
            labels.extend(y[idx]); groups.extend([source] * len(idx))
        np.testing.assert_array_equal(labels, audit['source_labels'])
        assert groups == audit['source_groups']

        g = np.asarray(groups)
        permuted = np.asarray(audit['permuted_source_labels'])
        for project in np.unique(g):
            m = g == project
            assert permuted[m].sum() == np.asarray(labels)[m].sum(), 'permutation altered counts'

        covered = []
        for fold in audit['tuning']:
            tr, va = np.array(fold['train']), np.array(fold['validation'])
            assert not set(tr) & set(va) and not set(g[tr]) & set(g[va])
            assert set(tr) | set(va) == set(range(len(g)))
            covered.extend(va); check_oof(fold['representation'], len(tr))
        assert sorted(covered) == list(range(len(g)))
        check_oof(audit['final_representation'], len(g))

        for key, c in audit['chosen'].items():
            regime, m, a = key.split('__')
            tuned = a if regime == 'independent' else 'original'
            means = [np.mean(audit['candidate_auc']['__'.join([m, tuned, str(i)])]) for i in range(2)]
            assert c == int(np.argmax(means)), f'stored choice disagrees with validation scores: {key}'

        sub = scores[(scores.target == target) & (scores.seed == seed) & (scores.teacher == teacher)]
        with np.load(stem + '.npz', allow_pickle=False) as saved:
            _, y, ids = load(target, cfg)
            np.testing.assert_array_equal(saved['labels'], y)
            np.testing.assert_array_equal(saved['target_ids'], ids)
            prevalence = set()
            for r in sub.itertuples():
                prob = saved['__'.join([r.tuning, r.model, r.arm])]
                for k, v in metric(y, prob).items():
                    assert np.isclose(v, getattr(r, k), rtol=0, atol=1e-12), f'{k} mismatch'
                if r.arm == 'prevalence':
                    prevalence.add(round(float(prob[0]), 12))
                    np.testing.assert_allclose(prob, prob[0])
                reconstructed += 1
            assert len(prevalence) == 1, 'featureless reference differs across learners'
            assert abs(prevalence.pop() - audit['source_positive_rate']) < 1e-12

        rep = audit['final_representation']
        w = np.asarray(rep['weights'])
        assert np.ptp(w) > 1e-10, 'attribution weights are constant'
        diagnostics.append({'target': target, 'seed': seed, 'teacher': teacher,
                            'redundancy': rep['source_redundancy'], 'shift': rep['target_median_shift'],
                            'active_features': len(w), 'seconds': audit['seconds']})
        vector = np.zeros(len(cfg['features'])); vector[rep['keep']] = w
        weights.append({'target': target, 'seed': seed, 'teacher': teacher,
                        **dict(zip(cfg['features'], vector))})
    return reconstructed, pd.DataFrame(diagnostics), pd.DataFrame(weights)


def paired(scores, arm, control, metric_name):
    keys = ['target', 'seed', 'teacher', 'tuning', 'model']
    m = scores[scores.arm == arm].merge(scores[scores.arm == control], on=keys,
                                        suffixes=('_a', '_b'), validate='one_to_one')
    m['delta'] = m[metric_name + '_a'] - m[metric_name + '_b']
    return m


def analyze():
    cfg = config()
    scores = pd.read_csv(OUT / 'scores.csv')
    reconstructed, diagnostics, weights = validate(cfg, scores)

    cells, effects, summary, loto, aug = [], [], [], [], []
    for (arm, control, isolates), metric_name in product(COMPARISONS, METRICS):
        p = paired(scores, arm, control, metric_name)
        for key, d in p.groupby(['target', 'teacher', 'tuning', 'model']):
            effects.append(dict(zip(['target', 'teacher', 'tuning', 'model'], key),
                                arm=arm, control=control, metric=metric_name,
                                mean_difference=float(d.delta.mean()),
                                repeat_sd=float(d.delta.std()),
                                repeat_min=float(d.delta.min()), repeat_max=float(d.delta.max()),
                                repeats=len(d)))
        for regime, dr in p.groupby('tuning'):
            # protocol aggregation: within target first, then equal-target mean
            per_target = dr.groupby('target').delta.mean()
            full = float(per_target.mean())
            drops = {t: float(per_target.drop(t).mean()) for t in per_target.index}
            worst = max(drops, key=lambda t: abs(drops[t] - full))
            summary.append({'arm': arm, 'control': control, 'metric': metric_name,
                            'tuning': regime, 'isolates': isolates, 'mean_difference': full,
                            'target_min': float(per_target.min()), 'target_max': float(per_target.max()),
                            'targets_positive': int((per_target > 0).sum()), 'targets': len(per_target),
                            'cells': len(dr), 'cells_negative': int((dr.delta < 0).sum()),
                            'mean_repeat_sd': float(dr.groupby(
                                ['target', 'teacher', 'model']).delta.std().mean())})
            row = {'arm': arm, 'control': control, 'metric': metric_name, 'tuning': regime,
                   'all_targets': full, 'loto_min': min(drops.values()),
                   'loto_max': max(drops.values()), 'most_influential_omission': worst,
                   'largest_shift': float(drops[worst] - full),
                   'sign_stable': bool(np.sign(min(drops.values())) == np.sign(max(drops.values()))
                                       == np.sign(full) or full == 0)}
            row.update({'without_' + t: v for t, v in drops.items()})
            loto.append(row)
        cells.append({'arm': arm, 'control': control, 'metric': metric_name,
                      'cells': len(p), 'cells_negative': int((p.delta < 0).sum()),
                      'cells_positive': int((p.delta > 0).sum()),
                      'targets_all_negative_after_averaging': int(
                          (p.groupby(['target', 'tuning']).delta.mean() < 0).all()),
                      'every_individual_cell_negative': int((p.delta < 0).all())})

    # augmentation is the co-headline: report it along every declared dimension
    for arm, control in [('ps_aug', 'p_aug'), ('ps_aug', 'pn_aug'), ('pn_aug', 'p_aug')]:
        for metric_name in METRICS:
            p = paired(scores, arm, control, metric_name)
            for dim in ['target', 'model', 'teacher', 'tuning']:
                for level, d in p.groupby(dim):
                    aug.append({'arm': arm, 'control': control, 'metric': metric_name,
                                'dimension': dim, 'level': level,
                                'mean_difference': float(d.delta.mean()),
                                'negative_share': float((d.delta < 0).mean()), 'cells': len(d)})

    for name, frame in [('effects', effects), ('summary', summary), ('loto', loto),
                        ('augmentation', aug), ('cell_signs', cells)]:
        pd.DataFrame(frame).to_csv(OUT / f'{name}.csv', index=False)
    diagnostics.to_csv(OUT / 'diagnostics.csv', index=False)
    weights.to_csv(OUT / 'weights.csv', index=False)

    stability = []
    for target, d in weights.groupby('target'):
        for t, group in d.groupby('teacher'):
            v = group[cfg['features']].to_numpy()
            stability.append({'target': target, 'comparison': f'{t} across seeds',
                              'mean_spearman': float(np.mean(
                                  [spearmanr(a, b).statistic for a, b in combinations(v, 2)]))})
        means = d.groupby('teacher')[cfg['features']].mean()
        stability.append({'target': target, 'comparison': 'RandomForest vs ExtraTrees teacher',
                          'mean_spearman': float(spearmanr(means.loc['RandomForest'],
                                                           means.loc['ExtraTrees']).statistic)})
    pd.DataFrame(stability).to_csv(OUT / 'stability.csv', index=False)

    quality = []
    for project in cfg['projects']:
        x, y, _ = load(project, cfg)
        quality.append({'project': project, 'rows': len(y), 'buggy': int(y.sum()),
                        'prevalence': float(y.mean()), 'static_features': x.shape[1],
                        'repeated_metric_rows': int(pd.DataFrame(x).duplicated(keep=False).sum())})
    pd.DataFrame(quality).to_csv(OUT / 'datasets.csv', index=False)

    dump(OUT / 'validation.json', {
        'valid': True, 'score_records_reconstructed': reconstructed,
        'checkpoints': len(cfg['projects']) * len(cfg['seeds']) * len(cfg['teachers']),
        'targets': len(cfg['projects']), 'seeds': len(cfg['seeds']),
        'learners': len(cfg['models']), 'arms': len(cfg['arms']),
        'grid_complete_and_unique': True, 'target_label_access': 'scoring only',
        'coral_reads_unlabelled_target_covariates': True,
        'source_project_folds_verified': True, 'oof_coverage_verified': True,
        'per_arm_parameter_choices_verified': True,
        'augmentation_control_dimension_matched': True,
        'permutation_preserves_project_class_counts': True,
        'featureless_reference_learner_independent': True,
        'total_compute_minutes': round(float(diagnostics.seconds.sum()) / 60, 1)})

    s = pd.DataFrame(summary)
    print(s[(s.metric == 'auc') & (s.tuning == 'independent')][
        ['arm', 'control', 'mean_difference', 'target_min', 'target_max',
         'targets_positive']].to_string(index=False))
    return s


if __name__ == '__main__':
    analyze()
