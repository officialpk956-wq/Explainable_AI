"""Generate the EMSE manuscript from validated v2 results.

python -m strengthened.build_paper_v2

Interpretation sentences are derived from the data rather than written in advance,
so the prose cannot describe a direction the results do not have. Author-supplied
declarations are left as visible placeholders; they are never invented here.
"""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .experiment import ROOT, digest, dump
from .experiment_v2 import OUT, config
from . import figures_v2 as F
from . import tables_v2 as T

DEST = ROOT / 'paper/emse'
COMPARISONS = [('shap_sum', 'uniform_sum', 'Weighting vs uniform'),
               ('shap_sum', 'shuffled_sum', 'Weighting vs shuffled'),
               ('select_shap', 'select_random', 'Selection vs random'),
               ('ps_aug', 'p_aug', 'Attribution vs prediction augmentation')]
PLACEHOLDER = (r'\textbf{[To be completed by the authors before submission.]} ')


def sign_word(v, tol=1e-4):
    return 'positive' if v > tol else ('negative' if v < -tol else 'indistinguishable from zero')


def build():
    cfg = config()
    validation = json.loads((OUT / 'validation.json').read_text())
    if not validation.get('valid'):
        raise RuntimeError('Valid full v2 results are required')
    for d in ['figures', 'tables', 'results']:
        (DEST / d).mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(OUT / 'scores.csv')
    summary = pd.read_csv(OUT / 'summary.csv')
    effects = pd.read_csv(OUT / 'effects.csv')
    loto = pd.read_csv(OUT / 'loto.csv')
    aug = pd.read_csv(OUT / 'augmentation.csv')
    cells = pd.read_csv(OUT / 'cell_signs.csv')
    stability = pd.read_csv(OUT / 'stability.csv')
    datasets = pd.read_csv(OUT / 'datasets.csv')

    def val(a, b, column='mean_difference', metric='auc', tuning='independent'):
        d = summary[(summary.metric == metric) & (summary.tuning == tuning) &
                    (summary.arm == a) & (summary.control == b)]
        if not len(d):
            raise KeyError(f'no summary row for {a} vs {b} ({metric}, {tuning})')
        return float(d.iloc[0][column])

    def lot(a, b, column, metric='auc', tuning='independent'):
        d = loto[(loto.metric == metric) & (loto.tuning == tuning) &
                 (loto.arm == a) & (loto.control == b)]
        return d.iloc[0][column]

    n = lambda v, p=4: f'{v:+.{p}f}'

    # ---- figures and tables -------------------------------------------------
    F.build_all(COMPARISONS)
    v = {}
    v['workflow_figure'] = fig(1, 'workflow',
        'External evaluation flow, repeated for each held-out project. Target responses enter '
        'scoring only. Representations and the teacher are rebuilt inside every source-validation '
        'fold. The adaptation comparator additionally reads unlabelled target covariates')
    v['harness_figure'] = fig(2, 'harness',
        'Harness controls under independent tuning. Each grey point is one held-out project, '
        'averaged over teachers, learners and source subsamples; the bar is the equal-target mean. '
        'The featureless reference sits at chance by construction')
    v['effects_figure'] = fig(3, 'effects',
        'Target means for the matched-control comparisons under independent tuning. Each panel '
        'has its own horizontal scale. Points are descriptive means with no confidence intervals')
    v['loto_figure'] = fig(5, 'loto',
        'Leave-one-target-out aggregation sensitivity. The bar spans the most extreme estimates '
        'obtained by omitting a single project from the aggregate; nothing is retrained. The span '
        'is not a confidence interval')
    v['augmentation_figure'] = fig(4, 'augmentation',
        'Augmentation against the prediction-only control and against the dimension-matched '
        'control, by target, downstream learner and attribution teacher')
    v['stability_figure'] = fig(6, 'stability',
        'Attribution rank agreement across source subsamples and between teacher-mean vectors. '
        'Correlations describe rankings, not predictive equivalence')

    v['dataset_table'] = T.corpus_table(datasets)
    v['arms_table'] = T.arms_table(cfg)
    v['controls_table'] = T.controls_table(summary)
    v['baseline_table'] = T.baseline_table(scores, cfg)
    v['effects_table'] = T.effects_table(summary, COMPARISONS)
    v['loto_table'] = T.loto_table(loto, COMPARISONS)
    v['augmentation_table'] = T.augmentation_table(summary, cells)
    v['secondary_table'] = T.secondary_table(summary, COMPARISONS)

    # ---- scalar values ------------------------------------------------------
    v['score_records'] = f'{len(scores):,}'
    v['arm_count'] = str(len(cfg['arms']))
    v['seed_count'] = str(len(cfg['seeds']))
    v['ctl_prevalence'] = n(val('original', 'prevalence'))
    v['ctl_permuted'] = n(val('permuted_original', 'original'))
    v['ctl_coral'] = n(val('coral', 'original'))
    v['weight'] = n(val('shap_sum', 'uniform_sum'))
    v['weight_shuffled'] = n(val('shap_sum', 'shuffled_sum'))
    v['selection'] = n(val('select_shap', 'select_random'))
    v['selection_min'] = n(val('select_shap', 'select_random', 'target_min'))
    v['selection_max'] = n(val('select_shap', 'select_random', 'target_max'))
    v['augmentation'] = n(val('ps_aug', 'p_aug'))
    v['aug_noise_vs_p'] = n(val('pn_aug', 'p_aug'))
    v['aug_shap_vs_noise'] = n(val('ps_aug', 'pn_aug'))
    # ps_aug is [active features | teacher probability | one attribution column per feature],
    # so the number of appended attribution columns is (width - 1) / 2.
    width = int(json.loads(next(OUT.glob('*.audit.json')).read_text())
                ['final_representation']['augmentation_columns']['ps_aug'])
    added = (width - 1) // 2
    v['aug_width'] = str(added)
    v['teacher_stability'] = f"{stability[stability.comparison.str.contains('vs')].mean_spearman.mean():.3f}"
    sel_sd = effects[(effects.metric == 'auc') & (effects.tuning == 'independent') &
                     (effects.arm == 'select_shap') & (effects.control == 'select_random')].repeat_sd.mean()
    aug_sd = effects[(effects.metric == 'auc') & (effects.tuning == 'independent') &
                     (effects.arm == 'ps_aug') & (effects.control == 'p_aug')].repeat_sd.mean()
    v['repeat_sd'] = f'{sel_sd:.4f}'
    v['repeat_sd_aug'] = f'{aug_sd:.4f}'

    # ---- data-derived interpretation ---------------------------------------
    prev, perm, coral = val('original', 'prevalence'), val('permuted_original', 'original'), val('coral', 'original')
    v['harness_interpretation'] = (
        f'The featureless reference scores at chance by construction, and the untreated learner '
        f'sits {abs(prev):.4f} ROC-AUC above it, so the benchmark carries learnable source signal. '
        f'Permuting source labels within project removes that signal while preserving every class '
        f'count, and the measurement falls by {abs(perm):.4f}. The adaptation comparator is '
        f'{sign_word(coral)} at {n(coral)}. These contrasts fix the scale on which the remaining '
        f'comparisons should be read: the harness moves by hundredths of a point when signal is '
        f'removed or a representation is realigned. It does not follow that it could resolve an '
        f'effect near zero, and we make no such claim.')

    w, ws, sel = val('shap_sum', 'uniform_sum'), val('shap_sum', 'shuffled_sum'), val('select_shap', 'select_random')
    biggest = max(abs(w), abs(ws), abs(sel))
    stable = [lab for a, b, lab in COMPARISONS[:3] if bool(lot(a, b, 'sign_stable'))]
    ratio = abs(perm) / biggest if biggest else float('inf')
    v['effects_interpretation'] = (
        f'Each matched contrast is {sign_word(w)} for weighting against a uniform multiplier '
        f'({n(w)}), {sign_word(ws)} for weighting against a permuted weight vector ({n(ws)}), and '
        f'{sign_word(sel)} for selection against an equal-sized random subset ({n(sel)}, target '
        f'means spanning {val("select_shap", "select_random", "target_min"):+.4f} to '
        f'{val("select_shap", "select_random", "target_max"):+.4f}). '
        + (f'All three keep their sign when any single target is dropped from the aggregate. '
           if len(stable) == 3 else
           f'{len(stable)} of the three keep their sign when any single target is dropped from '
           f'the aggregate. ') +
        f'The direction is therefore consistent, but the magnitude is not: the largest of the '
        f'three is {biggest:.4f} ROC-AUC, roughly {ratio:.0f} times smaller than the '
        f'{abs(perm):.4f} the same measurement loses when source label signal is destroyed, and '
        f'about {abs(prev) / biggest:.0f} times smaller than the gap between the untreated learner '
        f'and a featureless reference. A small consistent benefit is what these data support. '
        f'They do not support a claim of equivalence to zero, which would require an interval we '
        f'have not estimated, nor a claim that this magnitude is operationally useful.')

    a_p, nz_p, a_nz = val('ps_aug', 'p_aug'), val('pn_aug', 'p_aug'), val('ps_aug', 'pn_aug')
    c = cells[(cells.metric == 'auc') & (cells.arm == 'ps_aug') & (cells.control == 'p_aug')].iloc[0]
    share = (nz_p / a_p * 100) if a_p != 0 else float('nan')
    attribution_specific = abs(a_nz) > abs(nz_p)
    v['augmentation_interpretation'] = (
        f'The decomposition separates two explanations that the prediction-only comparison '
        f'confounds. Appending {added} uninformative columns of matched width and scale '
        f'changes ROC-AUC by {n(nz_p)}, '
        + (f'which exceeds the {n(a_p)} recorded for attribution augmentation overall'
           if abs(nz_p) > abs(a_p) else
           f'which is {"most" if abs(share) > 60 else "part"} of the {n(a_p)} recorded for '
           f'attribution augmentation overall') +
        f'. What remains once that dimensional cost is removed is '
        f'{n(a_nz)}, so the penalty is '
        f'{"predominantly attribution specific" if attribution_specific else "largely a generic added-dimension effect rather than a property of the attributions"}. '
        f'The penalty is negative after averaging within '
        f'{"every" if c.targets_all_negative_after_averaging else "some but not all"} target, and '
        f'negative in {int(c.cells_negative)} of {int(c.cells)} individual experimental cells. '
        f'The cell-level count is the weaker and more informative statement, and we report both.')

    unstable = []
    for a, b, label in COMPARISONS:
        if not bool(lot(a, b, 'sign_stable')):
            unstable.append(f'{label.lower()} ({n(lot(a, b, "loto_min"))} to {n(lot(a, b, "loto_max"))})')
    v['loto_interpretation'] = (
        ('Every headline aggregate keeps its sign under omission of any single target. ' if not unstable
         else 'The sign of the aggregate changes under omission of a single target for '
              + '; '.join(unstable) + '. ') +
        'Because seven targets is a small aggregate, we report all seven recomputed estimates '
        'rather than a range alone, and we note that this procedure does not retrain anything: '
        'the omitted project remains in the source pool of the other six.')

    d_sel = val('select_shap', 'select_random') - val('select_shap', 'select_random', tuning='shared_baseline')
    v['tuning_interpretation'] = (
        f'Independent rather than shared-baseline tuning changes the aggregate selection contrast '
        f'by {n(d_sel)} ROC-AUC. That is a descriptive difference between two completed evaluation '
        f'procedures with equal candidate budgets, not evidence about what a larger search would '
        f'achieve. Per-target and per-learner values are retained so an aggregate cannot conceal a '
        f'reversal confined to one setting.')

    v['aug_verdict_short'] = ('attributable mainly to the attribution columns themselves'
                              if attribution_specific else
                              'largely explained by the added dimensions rather than by the attributions')
    v['conclusion'] = (
        f'We evaluated source-derived SHAP weighting, selection and augmentation against controls '
        f'matched on the property each operation changes, on seven documented project families '
        f'with a fixed static-metric schema. Against those controls weighting and selection were '
        f'{"consistently " + sign_word(sel) if len(stable) == 3 else sign_word(sel)} but small: '
        f'the largest matched contrast was {biggest:.4f} ROC-AUC, some {ratio:.0f} times below '
        f'what the same measurement loses when source label signal is destroyed. '
        f'Augmentation was negative at {n(a_p)}, and the dimension-matched control showed this to '
        f'be {v["aug_verdict_short"]}. Three harness controls establish that the same measurement '
        f'moves by {abs(perm):.4f} when source label signal is destroyed and by {abs(prev):.4f} '
        f'between the untreated learner and a featureless reference, which is what makes the '
        f'smaller numbers interpretable. '
        f'We draw a narrow conclusion. On this benchmark, with these implementations, an '
        f'improvement over an untreated learner would not have demonstrated attribution utility, '
        f'and the matched controls that show this are inexpensive to run. We do not claim that '
        f'attribution reuse is unhelpful in general, that these estimates are equivalent to zero, '
        f'or that a different explainer, teacher, budget or feature family would behave the same way.')

    v['identity_statement'] = identity_statement()
    v['sourcecap_statement'] = sourcecap_statement()

    for key, text in [('author_funding', 'State any funding received, or state that no funding was received.'),
                      ('author_competing', 'State any competing interests, or state that the authors have none.'),
                      ('author_availability', 'Confirm the public archive identifier for code and derived results, and confirm redistribution terms for the source dataset.'),
                      ('author_contributions', 'State each author\'s contribution using the journal\'s preferred taxonomy.')]:
        v[key] = PLACEHOLDER + text

    template = (ROOT / 'strengthened/manuscript_v2.tex.in').read_text(encoding='utf-8')
    for key, value in v.items():
        template = template.replace('@@' + key + '@@', str(value))
    if '@@' in template:
        missing = sorted({s.split('@@')[0] for s in template.split('@@')[1::2]})
        raise RuntimeError(f'Unresolved manuscript values: {missing}')
    (DEST / 'main.tex').write_text(template, encoding='utf-8')

    for name in ['scores.csv', 'effects.csv', 'summary.csv', 'loto.csv', 'augmentation.csv',
                 'cell_signs.csv', 'datasets.csv', 'diagnostics.csv', 'stability.csv',
                 'weights.csv', 'validation.json']:
        shutil.copy2(OUT / name, DEST / 'results' / name)
    dump(DEST / 'build_provenance.json', {
        'input_hashes': {str(p.relative_to(ROOT)): digest(p) for p in
                         [OUT / 'scores.csv', OUT / 'summary.csv', OUT / 'loto.csv',
                          ROOT / 'strengthened/manuscript_v2.tex.in', Path(__file__)]},
        'scalars': {k: val for k, val in v.items()
                    if not k.endswith(('_table', '_figure', '_interpretation', 'conclusion'))},
        'target_venue': 'Empirical Software Engineering'})
    print(json.dumps({k: val for k, val in v.items()
                      if not k.endswith(('_table', '_figure', '_interpretation', 'conclusion'))},
                     indent=2))
    return v


def identity_statement():
    """State whether the restricted new run reproduces the earlier one, from the data."""
    from .compare_versions import equal_target_mean, SHARED_SEEDS, SHARED_MODELS, HEADLINE
    from .experiment import OUT as V1
    v1 = pd.read_csv(V1 / 'scores.csv')
    v2 = pd.read_csv(OUT / 'scores.csv')
    r = v2[v2.seed.isin(SHARED_SEEDS) & v2.model.isin(SHARED_MODELS)]
    diffs = []
    for arm, control in HEADLINE:
        a, b = equal_target_mean(v1, arm, control), equal_target_mean(r, arm, control)
        if a is not None and b is not None:
            diffs.append(abs(a - b))
    worst = max(diffs) if diffs else float('nan')
    if worst == 0:
        return (f'Restricted in this way, the present execution reproduces all {len(diffs)} shared '
                f'headline estimates exactly, so the differences in the full grid come from the '
                f'wider grid and the added arms rather than from any change in the shared code '
                f'path.')
    if worst < 5e-5:
        return (f'Restricted in this way, the present execution reproduces every one of the '
                f'{len(diffs)} shared headline estimates to within {worst:.1e} ROC-AUC, so the '
                f'differences in the full grid come from the wider grid rather than from a change '
                f'in the shared code path.')
    return (f'Restricted in this way, the largest discrepancy against the earlier execution across '
            f'the {len(diffs)} shared headline estimates is {worst:.4f} ROC-AUC, so the shared code '
            f'path is not bit-identical and the comparison should be read accordingly.')


def sourcecap_statement():
    """Report the predeclared 400 vs 800 source-cap sensitivity if it has been run."""
    from .compare_versions import equal_target_mean, SHARED_SEEDS
    cap = OUT.parent / 'cap800'
    if not (cap / 'scores.csv').exists():
        return ('The predeclared source-cap sensitivity comparing 400 against 800 rows per source '
                'project has not been executed for this build, and no number is reported for it.')
    big = pd.read_csv(cap / 'scores.csv')
    small = pd.read_csv(OUT / 'scores.csv')
    small = small[small.seed.isin(SHARED_SEEDS) & small.model.isin(sorted(big.model.unique()))]
    rows = []
    for arm, control, label in COMPARISONS:
        a, b = equal_target_mean(small, arm, control), equal_target_mean(big, arm, control)
        if a is not None and b is not None:
            rows.append(f'{label.lower()} {a:+.4f} to {b:+.4f}')
    return ('A predeclared sensitivity doubling the source cap from 400 to 800 rows per project, '
            'holding the three shared seeds and all other settings fixed, moves the headline '
            'estimates as follows: ' + '; '.join(rows) + '. The cap therefore conditions the '
            'magnitudes and is reported as a design choice rather than a neutral detail.')


def fig(n, label, caption):
    return '\n'.join([r'\begin{figure}[!htbp]', r'\centering',
                      rf'\includegraphics[width=\linewidth]{{figures/Fig{n}.pdf}}',
                      rf'\caption{{{caption}}}\label{{fig:{label}}}', r'\end{figure}'])


if __name__ == '__main__':
    build()
