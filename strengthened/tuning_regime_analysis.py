"""Post-hoc tuning-regime robustness analyses, no new training.

python -m strengthened.tuning_regime_analysis

Tantithamthavorn, McIntosh, Hassan & Matsumoto (TSE 2019) show automated
parameter optimization substantially shifts variable-importance rankings (as
few as 28% of top variables stay top-ranked) and ask, as future work, whether
tuning choice changes which learner looks best, whether it matters more for
ensembles, how consistent the winning hyperparameter is across datasets (a
transferability proxy), and what tuning costs. This answers all four from data
strengthened.experiment_v3 already computed and stored -- no new model fit.
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .experiment import ROOT, dump
from .experiment_v3 import OUT, config

DEST = OUT / 'tuning_regime_analysis'
ENSEMBLE = {'RandomForest', 'ExtraTrees', 'XGBoost'}
NON_ENSEMBLE = {'LogisticRegression', 'SVCRBF'}
COMPARISONS = [('shap_sum', 'uniform_sum'), ('shap_sum', 'shuffled_sum'),
              ('select_shap', 'select_random'), ('ps_aug', 'p_aug')]


def learner_rank_stability(effects):
    """(a) Does independent-vs-shared tuning reorder which learner shows the
    largest effect? Spearman rank correlation of per-learner effect size
    between the two regimes, per comparison."""
    rows = []
    for arm, control in COMPARISONS:
        d = effects[(effects.metric == 'auc') & (effects.arm == arm) &
                    (effects.control == control)]
        per_learner = d.groupby(['tuning', 'model']).mean_difference.mean().unstack('tuning')
        if {'independent', 'shared_baseline'} - set(per_learner.columns):
            continue
        rho = spearmanr(per_learner['independent'], per_learner['shared_baseline']).statistic
        rows.append({'arm': arm, 'control': control,
                    'learner_rank_spearman_independent_vs_shared': float(rho),
                    'n_learners': len(per_learner)})
    return pd.DataFrame(rows)


def ensemble_split(effects):
    """(b) Does the regime choice move ensembles more than non-ensembles?"""
    rows = []
    for arm, control in COMPARISONS:
        d = effects[(effects.metric == 'auc') & (effects.arm == arm) &
                    (effects.control == control)]
        piv = d.groupby(['model', 'tuning']).mean_difference.mean().unstack('tuning')
        if {'independent', 'shared_baseline'} - set(piv.columns):
            continue
        piv['abs_shift'] = (piv['independent'] - piv['shared_baseline']).abs()
        for family, members in [('ensemble', ENSEMBLE), ('non_ensemble', NON_ENSEMBLE)]:
            sub = piv[piv.index.isin(members)]
            if len(sub):
                rows.append({'arm': arm, 'control': control, 'family': family,
                            'mean_abs_regime_shift': float(sub.abs_shift.mean()),
                            'learners': ';'.join(sub.index)})
    return pd.DataFrame(rows)


def candidate_consistency(cfg):
    """(c) Across all 140 checkpoints, how consistently does independent
    per-arm tuning pick the same candidate index for a given (model, arm)?
    A modal share near 1.0 means the hyperparameter choice transfers across
    seeds and projects; near 0.5 means it is close to a coin flip."""
    manifest = json.loads((OUT / 'manifest.json').read_text())
    tallies = {}
    for target, seed, teacher in product(cfg['projects'], cfg['seeds'], cfg['teachers']):
        audit = json.loads((OUT / f'{target}__{seed}__{teacher}.audit.json').read_text())
        for key, c in audit['chosen'].items():
            regime, model, arm = key.split('__')
            if regime != 'independent':
                continue
            tallies.setdefault((model, arm), []).append(c)
    rows = []
    for (model, arm), choices in tallies.items():
        vals, counts = np.unique(choices, return_counts=True)
        modal_share = counts.max() / counts.sum()
        rows.append({'model': model, 'arm': arm, 'checkpoints': len(choices),
                    'modal_candidate': int(vals[np.argmax(counts)]),
                    'modal_share': float(modal_share)})
    return pd.DataFrame(rows).sort_values('modal_share')


def divergence_and_cost(cfg):
    """(d) Wall-clock: per-checkpoint total time already logged, plus the
    divergence rate between independent and shared_baseline choices, which
    estimates the fraction of (model, arm) final fits that need a second,
    extra model fit to support both regimes (a fit is duplicated only when
    the two regimes disagree on which candidate to use)."""
    seconds, divergences, total_pairs = [], 0, 0
    for target, seed, teacher in product(cfg['projects'], cfg['seeds'], cfg['teachers']):
        audit = json.loads((OUT / f'{target}__{seed}__{teacher}.audit.json').read_text())
        seconds.append(audit['seconds'])
        by_arm_model = {}
        for key, c in audit['chosen'].items():
            regime, model, arm = key.split('__')
            by_arm_model.setdefault((model, arm), {})[regime] = c
        for (model, arm), d in by_arm_model.items():
            if {'independent', 'shared_baseline'} <= set(d):
                total_pairs += 1
                divergences += int(d['independent'] != d['shared_baseline'])
    return {
        'mean_seconds_per_checkpoint': float(np.mean(seconds)),
        'total_checkpoint_hours': float(np.sum(seconds) / 3600),
        'model_arm_pairs_checked': total_pairs,
        'divergent_pairs': divergences,
        'divergence_rate': divergences / total_pairs if total_pairs else float('nan'),
        'interpretation': ('divergence_rate estimates the fraction of (learner, arm) final '
                           'fits that require an extra model fit to support both tuning '
                           'regimes in the same run, since a fit is duplicated only when '
                           'independent and shared-baseline tuning disagree on the '
                           'candidate index; it is not a directly re-timed ablation.'),
    }


def run():
    cfg = config()
    DEST.mkdir(parents=True, exist_ok=True)
    effects = pd.read_csv(OUT / 'effects.csv')

    rank = learner_rank_stability(effects)
    rank.to_csv(DEST / 'learner_rank_stability.csv', index=False)

    split = ensemble_split(effects)
    split.to_csv(DEST / 'ensemble_vs_nonensemble.csv', index=False)

    consistency = candidate_consistency(cfg)
    consistency.to_csv(DEST / 'candidate_consistency.csv', index=False)

    cost = divergence_and_cost(cfg)
    dump(DEST / 'cost_and_divergence.json', cost)

    verdict = {
        'a_learner_rank_stability_median_spearman': float(rank[
            'learner_rank_spearman_independent_vs_shared'].median()) if len(rank) else None,
        'b_ensemble_vs_nonensemble_mean_shift': split.groupby(
            'family').mean_abs_regime_shift.mean().to_dict() if len(split) else {},
        'c_candidate_consistency_median_modal_share': float(consistency.modal_share.median())
            if len(consistency) else None,
        'c_least_consistent_model_arm_pairs': consistency.head(5).to_dict('records'),
        'd_cost_and_divergence': cost,
    }
    dump(DEST / 'verdict.json', verdict)

    print('(a) median learner-rank Spearman (independent vs shared):',
          verdict['a_learner_rank_stability_median_spearman'])
    print('(b) mean |regime shift| by family:', verdict['b_ensemble_vs_nonensemble_mean_shift'])
    print('(c) median modal candidate share:', verdict['c_candidate_consistency_median_modal_share'])
    print('(d) mean seconds/checkpoint:', cost['mean_seconds_per_checkpoint'],
          '| divergence rate:', cost['divergence_rate'])
    print(f'\nwrote {DEST}')
    return verdict


if __name__ == '__main__':
    run()
