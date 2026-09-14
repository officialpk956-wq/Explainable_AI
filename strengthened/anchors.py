"""Put the attribution effects on a scale a reader can act on.

python -m strengthened.anchors

A difference of 0.003 ROC-AUC means nothing on its own. This expresses each
attribution effect as a fraction of quantities the same harness already measured:
the total learnable signal on this benchmark, the cost of destroying the source
labels, and what doubling the training data buys. All anchors come from completed
runs; nothing new is fitted here.
"""
import json

import pandas as pd

from .experiment import ROOT, dump
from .experiment_v2 import OUT as V2
from .experiment_v3 import OUT as V3, config

PK = ['target', 'seed', 'teacher', 'tuning', 'model']
M = ['auc', 'average_precision', 'f1_macro']


def equal_target_mean(scores, arm, control, tuning='independent'):
    s = scores[scores.tuning == tuning].groupby(PK + ['arm'], as_index=False)[M].mean()
    m = s[s.arm == arm].merge(s[s.arm == control], on=PK, suffixes=('_a', '_b'),
                              validate='one_to_one')
    if not len(m):
        return None
    return float((m.auc_a - m.auc_b).groupby(m.target).mean().mean())


def arm_mean(scores, arm, tuning='independent'):
    s = scores[(scores.tuning == tuning) & (scores.arm == arm)]
    return float(s.groupby('target').auc.mean().mean())


def build():
    cfg = config()
    v3 = pd.read_csv(V3 / 'scores.csv')
    anchors = {}

    # 1. the whole learnable signal on this benchmark
    total = equal_target_mean(v3, 'original', 'prevalence')
    anchors['total_learnable_signal'] = {
        'value': total,
        'meaning': 'untreated source-only learner minus a featureless source-rate predictor'}

    # 2. what destroying the source labels costs
    anchors['source_label_signal'] = {
        'value': abs(equal_target_mean(v3, 'permuted_original', 'original')),
        'meaning': 'cost of permuting source labels within project, preserving class counts'}

    # 3. what doubling the training data buys, from the v2 source-cap sensitivity
    cap = V2.parent / 'cap800' / 'scores.csv'
    if cap.exists():
        big = pd.read_csv(cap)
        small = pd.read_csv(V2 / 'scores.csv')
        shared_models = sorted(set(big.model.unique()))
        small = small[small.seed.isin(sorted(big.seed.unique())) & small.model.isin(shared_models)]
        anchors['doubling_training_data'] = {
            'value': arm_mean(big, 'original') - arm_mean(small, 'original'),
            'meaning': 'untreated learner at 800 versus 400 source rows per project, '
                       'same seeds and learners'}

    # 4. the attribution operations themselves
    bl = pd.read_csv(V3 / 'by_learner.csv')
    scale_only = bl[(bl.metric == 'auc') & (bl.tuning == 'independent') &
                    (bl.model == 'SCALE-SENSITIVE SUBSET')]
    effects = {
        'weighting_scale_sensitive_learners': float(
            scale_only[(scale_only.arm == 'shap_sum') &
                       (scale_only.control == 'uniform_sum')].mean_difference.iloc[0]),
        'selection_vs_random': equal_target_mean(v3, 'select_shap', 'select_random'),
        'augmentation_vs_prediction': equal_target_mean(v3, 'ps_aug', 'p_aug'),
        'row_correspondence': equal_target_mean(v3, 'ps_aug', 'pr_aug')}

    rows = []
    for name, effect in effects.items():
        row = {'effect': name, 'roc_auc': effect}
        for anchor, meta in anchors.items():
            base = meta['value']
            row[f'pct_of_{anchor}'] = 100 * effect / base if base else float('nan')
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(V3 / 'anchors.csv', index=False)

    # 5. what the attribution pipeline costs to run
    diag = pd.read_csv(V3 / 'diagnostics.csv')
    cost = {'mean_seconds_per_checkpoint': float(diag.seconds.mean()),
            'total_compute_minutes': float(diag.seconds.sum()) / 60,
            'note': ('A checkpoint fits every arm, so the marginal cost of the attribution '
                     'arms is a fraction of this. The teacher and its cross-fitted attributions '
                     'are the added step a practitioner would pay for.')}

    dump(V3 / 'anchors.json', {'anchors': anchors, 'effects': effects, 'cost': cost})
    print('ANCHORS (ROC-AUC)')
    for k, meta in anchors.items():
        print(f"  {k:<32}{meta['value']:+.4f}   {meta['meaning']}")
    print()
    print(table.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    return table


if __name__ == '__main__':
    build()
