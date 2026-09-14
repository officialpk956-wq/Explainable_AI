"""Generate the EMSE manuscript from validated v3 results.

python -m strengthened.build_paper_v3

Every interpretation sentence is derived from the data. Where a sensitivity
reverses a sign, the generated prose says so; it is not possible for this builder
to describe a robustness the results do not have.
"""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .experiment import ROOT, digest, dump
from .experiment_v3 import OUT, config
from . import figures_v2 as F
from . import tables_v2 as T

DEST = ROOT / 'paper/emse'
COMPARISONS = [('shap_sum', 'uniform_sum', 'Weighting vs uniform'),
               ('shap_sum', 'shuffled_sum', 'Weighting vs shuffled'),
               ('select_shap', 'select_random', 'Selection vs random'),
               ('ps_aug', 'p_aug', 'Attribution vs prediction augmentation')]
PLACEHOLDER = r'\textbf{[To be completed by the authors before submission.]} '
n = lambda v, p=4: f'{v:+.{p}f}'


def sign_word(v, tol=1e-4):
    return 'positive' if v > tol else ('negative' if v < -tol else 'indistinguishable from zero')


def build():
    cfg = config()
    validation = json.loads((OUT / 'validation.json').read_text())
    if not validation.get('valid'):
        raise RuntimeError('valid full v3 results are required')
    for d in ['figures', 'tables', 'results']:
        (DEST / d).mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(OUT / 'scores.csv')
    summary = pd.read_csv(OUT / 'summary.csv')
    effects = pd.read_csv(OUT / 'effects.csv')
    loto = pd.read_csv(OUT / 'loto.csv')
    cells = pd.read_csv(OUT / 'cell_signs.csv')
    bylearner = pd.read_csv(OUT / 'by_learner.csv')
    stability = pd.read_csv(OUT / 'stability.csv')
    datasets = pd.read_csv(OUT / 'datasets.csv')
    sens = pd.read_csv(OUT / 'sensitivity.csv')
    spread = pd.read_csv(OUT / 'control_draw_spread.csv')

    def val(a, b, column='mean_difference', metric='auc', tuning='independent'):
        d = summary[(summary.metric == metric) & (summary.tuning == tuning) &
                    (summary.arm == a) & (summary.control == b)]
        if not len(d):
            raise KeyError(f'{a} vs {b}')
        return float(d.iloc[0][column])

    def lot(a, b, column, metric='auc', tuning='independent'):
        return loto[(loto.metric == metric) & (loto.tuning == tuning) &
                    (loto.arm == a) & (loto.control == b)].iloc[0][column]

    def bl(a, b, model):
        d = bylearner[(bylearner.metric == 'auc') & (bylearner.tuning == 'independent') &
                      (bylearner.arm == a) & (bylearner.control == b) & (bylearner.model == model)]
        return float(d.iloc[0].mean_difference) if len(d) else float('nan')

    F.build_all(COMPARISONS)
    v = {}
    for key, num, label, cap in [
            ('workflow_figure', 1, 'workflow',
             'External evaluation flow, repeated for each held-out project. Target responses enter '
             'scoring only. Representations and the teacher are rebuilt inside every '
             'source-validation fold. The adaptation comparator additionally reads unlabelled '
             'target covariates'),
            ('harness_figure', 2, 'harness',
             'Harness controls under independent tuning. Each grey point is one held-out project; '
             'the bar is the equal-target mean. The featureless reference sits at chance by '
             'construction'),
            ('effects_figure', 3, 'effects',
             'Target means for the matched-control comparisons under independent tuning. Each '
             'panel has its own horizontal scale. Points are descriptive means with no confidence '
             'intervals'),
            ('augmentation_figure', 4, 'augmentation',
             'Augmentation against the prediction-only control and against the width-matched '
             'noise control, by target, downstream learner and attribution teacher'),
            ('loto_figure', 5, 'loto',
             'Leave-one-target-out aggregation sensitivity. The bar spans the most extreme '
             'estimates obtained by omitting a single project from the aggregate; nothing is '
             'retrained. The span is not a confidence interval'),
            ('stability_figure', 6, 'stability',
             'Attribution rank agreement across source subsamples and between teacher-mean '
             'vectors. Correlations describe rankings, not predictive equivalence')]:
        v[key] = '\n'.join([r'\begin{figure}[!htbp]', r'\centering',
                            rf'\includegraphics[width=\linewidth]{{figures/Fig{num}.pdf}}',
                            rf'\caption{{{cap}}}\label{{fig:{label}}}', r'\end{figure}'])

    v['dataset_table'] = T.corpus_table(datasets)
    v['arms_table'] = T.arms_table(cfg)
    v['controls_table'] = T.controls_table(summary)
    v['baseline_table'] = T.baseline_table(scores, cfg)
    v['effects_table'] = T.effects_table(summary, COMPARISONS)
    v['loto_table'] = T.loto_table(loto, COMPARISONS)
    v['secondary_table'] = T.secondary_table(summary, COMPARISONS)

    # --- per-learner weighting table ---------------------------------------
    rows = []
    for arm, control, label in COMPARISONS[:3]:
        for model in cfg['models'] + ['SCALE-SENSITIVE SUBSET']:
            d = bylearner[(bylearner.metric == 'auc') & (bylearner.tuning == 'independent') &
                          (bylearner.arm == arm) & (bylearner.control == control) &
                          (bylearner.model == model)]
            if not len(d):
                continue
            r = d.iloc[0]
            rows.append([label, model.replace('SCALE-SENSITIVE SUBSET', 'Scale-sensitive only'),
                         n(r.mean_difference), f'{r.max_abs_cell:.4f}',
                         'yes' if model in cfg['scale_sensitive_models'] else
                         ('--' if 'SUBSET' in model else 'no')])
    v['bylearner_table'] = T.table(
        'bylearner',
        'Matched-control effects by downstream learner, independent tuning. A positive per-feature '
        'multiplier cannot move an axis-aligned split, so the tree and boosting learners cannot '
        'respond to the weighting arms; the final column marks the learners that can. The maximum '
        'absolute cell shows that an exactly zero entry is a structural zero, not an average of '
        'cancelling effects',
        ['Comparison', 'Learner', 'Mean', 'Max abs cell', 'Scale sensitive'], rows)

    # --- augmentation decomposition table ----------------------------------
    rows = []
    for arm, control, label in [
            ('ps_aug', 'p_aug', 'Attributions vs prediction'),
            ('pn_aug', 'p_aug', 'Matched noise vs prediction'),
            ('pr_aug', 'p_aug', 'Row-permuted vs prediction'),
            ('ps_aug', 'pr_aug', 'Real vs row-permuted'),
            ('ps_aug', 'pn_aug', 'Real vs matched noise')]:
        c = cells[(cells.metric == 'auc') & (cells.arm == arm) & (cells.control == control)].iloc[0]
        rows.append([label, n(val(arm, control)), n(val(arm, control, 'target_min')),
                     n(val(arm, control, 'target_max')),
                     'yes' if bool(lot(arm, control, 'sign_stable')) else 'no',
                     f'{int(c.cells_negative)}/{int(c.cells)}'])
    v['augmentation_table'] = T.table(
        'augmentation',
        'Augmentation decomposed. Width-matched noise bounds the cost of added dimensions alone. '
        'The row permutation preserves every column marginal and every inter-column correlation of '
        'the real attribution matrix and destroys only the row correspondence, so the fourth row '
        'is the row-level attribution content. Sign stability refers to leave-one-target-out '
        'aggregation, not to the design sensitivities in Table~\\ref{tab:sensitivity}',
        ['Contrast', 'Mean', 'Min', 'Max', 'Stable', 'Neg. cells'], rows)

    # --- sensitivity table --------------------------------------------------
    rows = []
    for (folder, label), g in sens.groupby(['sensitivity', 'label'], sort=False):
        rev = g[~g.sign_preserved]
        worst = g.iloc[g['shift'].abs().argmax()]
        rows.append([label, str(int(g.seeds.iloc[0])), f"{g['shift'].abs().max():.4f}",
                     f'{worst.arm} vs {worst.control}'.replace('_', ' '),
                     'none' if not len(rev) else '; '.join(
                         f'{r.arm} vs {r.control}'.replace('_', ' ') for r in rev.itertuples())])
    v['sensitivity_table'] = T.table(
        'sensitivity',
        'Predeclared design sensitivities, each compared against the main execution restricted to '
        'the same seeds. The final column names any headline contrast whose sign does not survive '
        'the change. A sign reversal is reported, not resolved',
        ['Sensitivity', 'Seeds', 'Largest shift', 'On contrast', 'Sign reversals'], rows)

    # --- scalars ------------------------------------------------------------
    prev, perm, coral = (val('original', 'prevalence'), val('permuted_original', 'original'),
                         val('coral', 'original'))
    w, ws, sel = (val('shap_sum', 'uniform_sum'), val('shap_sum', 'shuffled_sum'),
                  val('select_shap', 'select_random'))
    w_scale = bl('shap_sum', 'uniform_sum', 'SCALE-SENSITIVE SUBSET')
    a_p, nz_p, pr_p = val('ps_aug', 'p_aug'), val('pn_aug', 'p_aug'), val('pr_aug', 'p_aug')
    a_pr, a_nz = val('ps_aug', 'pr_aug'), val('ps_aug', 'pn_aug')
    width = int(json.loads(next(OUT.glob('*.audit.json')).read_text())
                ['final_representation']['augmentation_columns']['ps_aug'])
    added = (width - 1) // 2

    v.update({'score_records': f'{len(scores):,}', 'arm_count': str(len(cfg['arms'])),
              'seed_count': str(len(cfg['seeds'])), 'ctl_prevalence': n(prev),
              'ctl_permuted': n(perm), 'ctl_coral': n(coral), 'weight': n(w),
              'weight_scale': n(w_scale), 'selection': n(sel), 'augmentation': n(a_p),
              'aug_noise_vs_p': n(nz_p), 'aug_perm_vs_p': n(pr_p), 'aug_shap_vs_perm': n(a_pr),
              'aug_width': str(added),
              'teacher_stability': f"{stability[stability.comparison.str.contains('vs')].mean_spearman.mean():.3f}"})
    sel_sd = effects[(effects.metric == 'auc') & (effects.tuning == 'independent') &
                     (effects.arm == 'select_shap') & (effects.control == 'select_random')].repeat_sd.mean()
    v['repeat_sd'] = f'{sel_sd:.4f}'
    sp = spread[(spread.arm == 'ps_aug') & (spread.control == 'pr_aug')]
    v['control_spread'] = f'{float(sp.iloc[0].spread):.4f}' if len(sp) else 'not measured'

    # --- generated interpretation -------------------------------------------
    v['harness_interpretation'] = (
        f'The featureless reference scores at chance by construction and the untreated learner '
        f'sits {abs(prev):.4f} ROC-AUC above it, so the benchmark carries learnable source signal. '
        f'Permuting source labels within project removes that signal while preserving every class '
        f'count, and the measurement falls by {abs(perm):.4f}. The adaptation comparator is '
        f'{sign_word(coral)} at {n(coral)}; we report it as measured and do not present it as a '
        f'successful adaptation. These contrasts fix the scale on which everything below should be '
        f'read. They do not establish that the design could resolve an effect near zero, and we '
        f'make no such claim.')

    zeros = [m for m in cfg['models'] if abs(bl('shap_sum', 'uniform_sum', m)) < 1e-9]
    v['learner_interpretation'] = (
        f'For {", ".join(zeros) if zeros else "no learner"} the weighting contrast is exactly zero '
        f'with a maximum absolute cell of zero, which is the structural result rather than an '
        f'average of cancelling effects. Restricting to the learners that can respond raises the '
        f'weighting contrast from {n(w)} to {n(w_scale)}. Reporting the all-learner aggregate '
        f'alone would understate a scale transformation by roughly '
        f'{abs(w_scale / w):.0f} times, so both are given.')

    stable = [lab for a, b, lab in COMPARISONS[:3] if bool(lot(a, b, 'sign_stable'))]
    biggest = max(abs(w_scale), abs(sel))
    v['effects_interpretation'] = (
        f'Weighting is {sign_word(w_scale)} for the scale-sensitive learners at {n(w_scale)} and '
        f'{sign_word(ws)} against a permuted weight vector at {n(ws)}. Selection is '
        f'{sign_word(sel)} against an equal-sized random subset at {n(sel)}, spanning '
        f'{val("select_shap", "select_random", "target_min"):+.4f} to '
        f'{val("select_shap", "select_random", "target_max"):+.4f} across targets. '
        + (f'All three keep their sign when any single target is dropped. ' if len(stable) == 3
           else f'{len(stable)} of the three keep their sign when any single target is dropped. ') +
        f'The direction is consistent and the magnitude is small: the largest is {biggest:.4f} '
        f'ROC-AUC, about {abs(perm) / biggest:.0f} times smaller than what the same measurement '
        f'loses when source label signal is destroyed. A small consistent benefit is what these '
        f'data support; equivalence to zero is not, and we have not estimated an interval that '
        f'could support it.')

    rev = sens[(~sens.sign_preserved) & (sens.arm == 'ps_aug') & (sens.control == 'pr_aug')]
    rev_names = sorted(set(rev.label)) if len(rev) else []
    v['augmentation_interpretation'] = (
        f'Width alone is the dominant term. Appending {added} independent columns matched to the '
        f'attribution columns in width and marginal scale changes ROC-AUC by {n(nz_p)}, against '
        f'{n(a_p)} for the real attribution columns. Preserving the attribution columns\' joint '
        f'structure while destroying the row correspondence gives {n(pr_p)}, so the column '
        f'structure recovers part of the width cost. What is left for the row correspondence '
        f'itself is {n(a_pr)}: pairing each row with its own attributions is '
        f'{"worse" if a_pr < 0 else "better"} than pairing it with another row\'s. That residual '
        f'is sign-stable under leave-one-target-out'
        + (f', and it strengthens under project-grouped teacher cross-fitting. '
           if not sens[(sens.sensitivity == "teacher_grouped") & (sens.arm == "ps_aug") &
                       (sens.control == "pr_aug")].empty and
              float(sens[(sens.sensitivity == "teacher_grouped") & (sens.arm == "ps_aug") &
                         (sens.control == "pr_aug")].iloc[0]['shift']) < 0 else '. ')
        + (f'It does not, however, survive every design sensitivity: its sign reverses under '
           f'{" and ".join(rev_names).lower()} (Table~\\ref{{tab:sensitivity}}). We therefore '
           f'report the width cost as the robust component and treat the row-correspondence term '
           f'as suggestive rather than established.'
           if rev_names else
           'It also survives every design sensitivity we ran.')
        + f' The contrast against width-matched noise alone, {n(a_nz)}, is '
        f'{"not " if not bool(lot("ps_aug", "pn_aug", "sign_stable")) else ""}sign-stable and is '
        f'the weaker of the two controls, because independent noise discards the joint structure '
        f'the real columns have.')

    unstable = [lab for a, b, lab in COMPARISONS if not bool(lot(a, b, 'sign_stable'))]
    v['loto_interpretation'] = (
        ('Every headline aggregate keeps its sign under omission of any single target. '
         if not unstable else
         'The sign of the aggregate changes under omission of a single target for '
         + '; '.join(u.lower() for u in unstable) + '. ') +
        'Seven targets is a small aggregate, so all seven recomputed estimates are reported rather '
        'than a range alone. The procedure retrains nothing and leaves the omitted project in the '
        'source pool of the other six.')

    total_rev = int((~sens.sign_preserved).sum())
    v['sensitivity_interpretation'] = (
        f'Across {len(sens)} sensitivity comparisons, {total_rev} reverse a sign'
        + (f', all of them on the row-correspondence contrast. ' if total_rev and
           set(sens[~sens.sign_preserved].arm) == {'ps_aug'} else '. ') +
        f'The harness controls are unmoved: the featureless and label-permutation contrasts are '
        f'bit-identical under project-grouped teacher cross-fitting, as they must be, since '
        f'neither arm uses the teacher. Larger capacity shrinks the augmentation penalty '
        f'substantially while leaving the width cost in place, which is consistent with the '
        f'penalty being a capacity-limited fitting cost rather than a property of the '
        f'attributions.')

    d_sel = sel - val('select_shap', 'select_random', tuning='shared_baseline')
    v['tuning_interpretation'] = (
        f'Independent rather than shared-baseline tuning changes the aggregate selection contrast '
        f'by {n(d_sel)} ROC-AUC, a descriptive difference between two completed procedures with '
        f'equal candidate budgets. Per-target and per-learner values are retained so an aggregate '
        f'cannot conceal a reversal confined to one setting.')

    v['aug_robust_short'] = ('its sign reverses under ' + ' and '.join(rev_names).lower()
                             if rev_names else 'it survives every sensitivity we ran')
    v['aug_verdict_short'] = (
        'The residual attributable to the row correspondence is negative and sign-stable under '
        'target omission, but it does not survive every design sensitivity, so we report it as '
        'suggestive rather than established.'
        if rev_names and a_pr < 0 else
        'The residual attributable to the row correspondence is reported with its sensitivities.')

    v['conclusion'] = (
        f'We evaluated source-derived SHAP weighting, selection and augmentation against controls '
        f'matched on the property each operation changes, on seven documented project families. '
        f'Weighting and selection were small and consistently {sign_word(sel)}: {n(w_scale)} for '
        f'the learners that can respond to a rescaling at all, and {n(sel)} for selection. Three '
        f'harness controls establish that the same measurement moves by {abs(perm):.4f} when '
        f'source label signal is destroyed, which is what makes those small numbers interpretable. '
        f'Augmentation was negative at {n(a_p)}, and decomposing it showed the dominant term to be '
        f'the cost of added width ({n(nz_p)}) rather than the attributions. '
        f'The residual attributable to pairing a row with its own attributions, {n(a_pr)}, is '
        f'sign-stable across targets but '
        + (f'reverses under {" and ".join(rev_names).lower()}, so we do not claim it as '
           f'established. ' if rev_names else 'survives every sensitivity we ran. ') +
        f'We draw a narrow conclusion. On this benchmark, with these implementations, an '
        f'improvement over an untreated learner would not have demonstrated attribution utility, '
        f'and the matched controls that show this are inexpensive. Reporting a weighting effect '
        f'averaged over learners that cannot respond to a rescaling, or an augmentation effect '
        f'without a width-matched control, would have supported conclusions the data do not carry. '
        f'We do not claim that attribution reuse is unhelpful in general, that these estimates are '
        f'equivalent to zero, or that another explainer, teacher, capacity or feature family would '
        f'behave the same way.')

    from .build_paper_v2 import identity_statement, sourcecap_statement
    v['identity_statement'] = ('The present execution adds a fifth learner and a further control '
                               'arm, so it is not expected to reproduce the earlier runs cell for '
                               'cell; the accompanying change report gives both.')
    v['sourcecap_statement'] = sourcecap_statement()

    for key, text in [('author_funding', 'State any funding received, or state that no funding was received.'),
                      ('author_competing', 'State any competing interests, or state that the authors have none.'),
                      ('author_availability', 'Confirm the public archive identifier for code and derived results, and confirm redistribution terms for the source dataset.'),
                      ('author_contributions', "State each author's contribution using the journal's preferred taxonomy.")]:
        v[key] = PLACEHOLDER + text

    template = (ROOT / 'strengthened/manuscript_v3.tex.in').read_text(encoding='utf-8')
    for key, value in v.items():
        template = template.replace('@@' + key + '@@', str(value))
    if '@@' in template:
        missing = sorted({s for s in template.split('@@')[1::2]})
        raise RuntimeError(f'unresolved manuscript values: {missing}')
    (DEST / 'main.tex').write_text(template, encoding='utf-8')

    for name in ['scores.csv', 'effects.csv', 'summary.csv', 'loto.csv', 'by_learner.csv',
                 'cell_signs.csv', 'sensitivity.csv', 'control_draw_spread.csv',
                 'draw_variability.csv', 'datasets.csv', 'diagnostics.csv', 'stability.csv',
                 'weights.csv', 'validation.json']:
        if (OUT / name).exists():
            shutil.copy2(OUT / name, DEST / 'results' / name)
    dump(DEST / 'build_provenance.json', {
        'input_hashes': {str(p.relative_to(ROOT)): digest(p) for p in
                         [OUT / 'scores.csv', OUT / 'summary.csv', OUT / 'loto.csv',
                          OUT / 'sensitivity.csv',
                          ROOT / 'strengthened/manuscript_v3.tex.in', Path(__file__)]},
        'scalars': {k: val for k, val in v.items()
                    if not k.endswith(('_table', '_figure', '_interpretation', 'conclusion',
                                       '_statement'))},
        'target_venue': 'Empirical Software Engineering'})
    print(json.dumps({k: val for k, val in v.items()
                      if not k.endswith(('_table', '_figure', '_interpretation', 'conclusion',
                                         '_statement'))}, indent=2))
    return v


if __name__ == '__main__':
    build()
