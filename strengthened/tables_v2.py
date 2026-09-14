"""LaTeX tables for the v2 manuscript.

Tables are written into paper/emse/tables/ for the local build. The submission
packager flattens them, because EMSE does not accept subfolders in an upload.
"""
from pathlib import Path

import pandas as pd

DEST = Path(__file__).resolve().parents[1] / 'paper/emse'
WIDE_FIRST_COLUMN = {'data', 'baseline', 'targeteffects', 'arms', 'loto', 'controls',
                     'effects', 'augmentation', 'secondary'}


def escape(s):
    return (str(s).replace('&', r'\&').replace('_', r'\_').replace('%', r'\%'))


def table(name, caption, headers, rows, note=None):
    cols = ('l' + 'Y' * (len(headers) - 1)) if name in WIDE_FIRST_COLUMN else 'Y' * len(headers)
    body = [r'\begin{table}[!htbp]', r'\centering\small', r'\setlength{\tabcolsep}{3pt}',
            rf'\caption{{{caption}}}\label{{tab:{name}}}',
            rf'\begin{{tabularx}}{{\linewidth}}{{{cols}}}', r'\toprule',
            ' & '.join(map(escape, headers)) + r' \\', r'\midrule']
    body += [' & '.join(map(escape, r)) + r' \\' for r in rows]
    body += [r'\bottomrule', r'\end{tabularx}']
    if note:
        body.append(rf'\par\smallskip\footnotesize {note}')
    body.append(r'\end{table}')
    (DEST / 'tables').mkdir(parents=True, exist_ok=True)
    (DEST / 'tables' / f'{name}.tex').write_text('\n'.join(body), encoding='utf-8')
    return rf'\input{{tables/{name}.tex}}'


def num(v, places=4):
    return f'{v:+.{places}f}'


def corpus_table(datasets):
    return table('data',
                 'External corpus. Prevalence is the fraction of files labelled defective under '
                 'the RealBug response. All inputs are the same 54 static code metrics',
                 ['Release', 'Files', 'Defective', 'Prevalence'],
                 [[r.project, str(r.rows), str(r.buggy), f'{r.prevalence:.3f}']
                  for r in datasets.itertuples()])


def arms_table(cfg):
    defs = cfg['arm_definitions']
    role = {'original': 'Baseline', 'shap_sum': 'Treatment', 'uniform_sum': 'Matched control',
            'shuffled_sum': 'Matched control', 'select_shap': 'Treatment',
            'select_random': 'Matched control', 'p_aug': 'Stacking control',
            'ps_aug': 'Treatment', 'pn_aug': 'Width control', 'pr_aug': 'Structure control',
            'coral': 'Comparator',
            'prevalence': 'Harness control', 'permuted_original': 'Harness control'}
    rows = []
    for a in cfg['arms']:
        text = defs[a].replace('NEW. ', '')
        rows.append([a.replace('_', ' '), role[a], text.split('.')[0] + '.'])
    return table('arms',
                 'Complete treatment, control and comparator design. Four arms are new in this '
                 'study: a featureless reference, a source-label permutation, a documented '
                 'adaptation comparator and a dimension-matched augmentation control',
                 ['Arm', 'Role', 'Representation'], rows)


def controls_table(summary):
    """The harness-sensitivity ladder, in one place."""
    rows = []
    for arm, control, label in [
            ('original', 'prevalence', 'Untreated vs featureless reference'),
            ('permuted_original', 'original', 'Permuted source labels vs untreated'),
            ('coral', 'original', 'CORAL vs untreated (transductive)')]:
        d = summary[(summary.metric == 'auc') & (summary.tuning == 'independent') &
                    (summary.arm == arm) & (summary.control == control)]
        if not len(d):
            continue
        r = d.iloc[0]
        rows.append([label, num(r.mean_difference), num(r.target_min), num(r.target_max),
                     f'{int(r.targets_positive)}/{int(r.targets)}'])
    return table('controls',
                 'Harness controls under independent tuning, equal-target means of paired '
                 'ROC-AUC differences. These establish that the evaluation registers removal '
                 'and addition of signal. They do not establish sensitivity to effects near zero, '
                 'and they are not equivalence or power evidence',
                 ['Contrast', 'Mean', 'Min target', 'Max target', 'Targets positive'], rows)


def effects_table(summary, comparisons):
    rows = []
    for arm, control, label in comparisons:
        ind = summary[(summary.metric == 'auc') & (summary.tuning == 'independent') &
                      (summary.arm == arm) & (summary.control == control)]
        sha = summary[(summary.metric == 'auc') & (summary.tuning == 'shared_baseline') &
                      (summary.arm == arm) & (summary.control == control)]
        if not len(ind):
            continue
        i, s = ind.iloc[0], sha.iloc[0]
        rows.append([label, num(i.mean_difference), num(s.mean_difference),
                     num(i.target_min), num(i.target_max),
                     f'{int(i.targets_positive)}/{int(i.targets)}'])
    return table('effects',
                 'Paired ROC-AUC effects, equal-target means. Indep.: independent per-arm tuning; '
                 'Shared: the arm-agnostic untreated choice. Both regimes receive equal '
                 'two-candidate budgets. The target range is the spread of seven target means '
                 'under independent tuning; it is not an uncertainty interval',
                 ['Comparison', 'Indep.', 'Shared', 'Min target', 'Max target',
                  'Targets +'], rows)


def loto_table(loto, comparisons):
    rows = []
    for arm, control, label in comparisons:
        d = loto[(loto.metric == 'auc') & (loto.tuning == 'independent') &
                 (loto.arm == arm) & (loto.control == control)]
        if not len(d):
            continue
        r = d.iloc[0]
        rows.append([label, num(r.all_targets), num(r.loto_min), num(r.loto_max),
                     r.most_influential_omission.split('-')[0].title(),
                     'yes' if r.sign_stable else 'no'])
    return table('loto',
                 'Leave-one-target-out aggregation sensitivity. Each estimate recomputes the '
                 'equal-target mean with one project omitted from the aggregate. Nothing is '
                 'retrained and the omitted project remains in the source pool of the other '
                 'six, so this is a sensitivity of the aggregate, not a project-exclusion '
                 'experiment. The range is not a confidence interval',
                 ['Comparison', 'All seven', 'Min', 'Max', 'Most influential omission',
                  'Sign stable'], rows)


def augmentation_table(summary, cells):
    rows = []
    for arm, control, label in [
            ('ps_aug', 'p_aug', 'Attribution columns beyond prediction'),
            ('ps_aug', 'pn_aug', 'Attribution columns beyond matched noise'),
            ('pn_aug', 'p_aug', 'Matched noise columns beyond prediction')]:
        d = summary[(summary.metric == 'auc') & (summary.tuning == 'independent') &
                    (summary.arm == arm) & (summary.control == control)]
        c = cells[(cells.metric == 'auc') & (cells.arm == arm) & (cells.control == control)]
        if not len(d):
            continue
        r, cc = d.iloc[0], c.iloc[0]
        rows.append([label, num(r.mean_difference), num(r.target_min), num(r.target_max),
                     'yes' if cc.targets_all_negative_after_averaging else 'no',
                     f'{int(cc.cells_negative)}/{int(cc.cells)}'])
    return table('augmentation',
                 'Augmentation decomposed against both controls. The final column counts '
                 'individual experimental cells, which is a weaker and more informative statement '
                 'than the target-averaged column beside it',
                 ['Contrast', 'Mean', 'Min target', 'Max target', 'Negative in every target',
                  'Negative cells'], rows)


def secondary_table(summary, comparisons):
    rows = []
    for arm, control, label in comparisons:
        vals = []
        for m in ['average_precision', 'f1_macro']:
            d = summary[(summary.metric == m) & (summary.tuning == 'independent') &
                        (summary.arm == arm) & (summary.control == control)]
            vals.append(num(d.iloc[0].mean_difference) if len(d) else '--')
        rows.append([label] + vals)
    return table('secondary',
                 'Secondary metrics under independent tuning, equal-target means. Average '
                 'precision is prevalence dependent and macro F1 uses a fixed 0.5 threshold, so '
                 'the three metrics answer different questions and need not agree',
                 ['Comparison', 'Average precision', 'Macro F1'], rows)


def baseline_table(scores, cfg):
    b = scores[(scores.arm == 'original') & (scores.tuning == 'independent')]
    b = b.groupby(['target', 'model'])[['auc', 'average_precision', 'f1_macro']].mean()
    abbrev = {'RandomForest': 'RF', 'LogisticRegression': 'LR',
              'ExtraTrees': 'ET', 'XGBoost': 'XGB', 'SVCRBF': 'SVC'}
    headers = ['Release'] + [f'{abbrev[m]} AUC' for m in cfg['models']]
    rows = []
    for t in cfg['projects']:
        rows.append([t] + [f'{b.loc[(t, m), "auc"]:.3f}' for m in cfg['models']])
    return table('baseline',
                 'Untreated absolute ROC-AUC by held-out project and downstream learner, '
                 'averaged over teachers and source subsamples. RF: RandomForest; LR: logistic '
                 'regression; ET: ExtraTrees; XGB: XGBoost',
                 headers, rows)
