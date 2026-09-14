"""Two one-sided tests (TOST) equivalence bound, synthetic benchmark only.

python -m strengthened.tost_synthetic

Tantithamthavorn, McIntosh, Hassan & Matsumoto (TSE 2017) show validation
estimates carry substantial resampling variance even at comparable dataset
scales -- an eyeballed "the effect is small" is not the same as a formally
bounded one. The real cross-project targets share source projects (row-level
non-independence), which is exactly why the manuscript reports no significance
test on them. The synthetic benchmark does not have that problem: each (seed,
target) cell is drawn from an independently generated dataset (see
strengthened.synthetic_power.generate), so a formal equivalence test is valid
there in a way it is not on the real targets. This applies TOST only to that
benchmark, never to the real one.

Margin: 0.02 ROC-AUC, identical to the ceiling already predeclared in
PREREGISTRATION.md before this script was written -- reusing an
already-fixed number rather than choosing a new one after seeing these results.
"""
import json
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .experiment import ROOT, dump
from .synthetic_power import OUT as SYN_OUT

DEST = SYN_OUT / 'tost'
MARGIN = 0.02          # reused from PREREGISTRATION.md, not chosen post hoc
ALPHA = 0.05
PAIRS = [('shap_sum', 'uniform_sum'), ('shap_sum', 'shuffled_sum'),
         ('select_shap', 'select_random')]


def tost(diffs, margin, alpha=ALPHA):
    """Schuirmann's two one-sided t-tests. Equivalence at `margin` declared
    only if BOTH one-sided nulls (diff <= -margin, diff >= +margin) are
    rejected at `alpha`."""
    n = len(diffs)
    if n < 2:
        raise ValueError('TOST needs at least 2 independent units')
    mean, se = float(np.mean(diffs)), float(np.std(diffs, ddof=1) / np.sqrt(n))
    if se == 0:
        p_lower = p_upper = 0.0 if abs(mean) < margin else 1.0
    else:
        df = n - 1
        t_lower = (mean - (-margin)) / se
        t_upper = (mean - margin) / se
        p_lower = float(1 - student_t.cdf(t_lower, df))   # H0: diff <= -margin
        p_upper = float(student_t.cdf(t_upper, df))       # H0: diff >= +margin
    p_tost = max(p_lower, p_upper)
    return {'n': n, 'mean': mean, 'se': se, 'p_lower': p_lower, 'p_upper': p_upper,
           'p_tost': p_tost, 'equivalent_at_margin': bool(p_tost < alpha)}


def paired_diffs(scores, arm, control, alpha, unit_cols):
    s = scores[scores.alpha == alpha]
    keys = ['target', 'seed', 'teacher', 'tuning', 'model']
    m = s[s.arm == arm].merge(s[s.arm == control], on=keys, suffixes=('_a', '_b'),
                              validate='one_to_one')
    m['delta'] = m.auc_a - m.auc_b
    # collapse teacher/tuning/model within each independent unit first, so the
    # TOST sample size reflects independently generated data, not repeated
    # measurements of the same draw
    return m.groupby(unit_cols).delta.mean()


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(SYN_OUT / 'scores.csv')
    alphas = sorted(scores.alpha.unique())

    rows = []
    for (arm, control), alpha in product(PAIRS, alphas):
        # primary unit: (seed, target) cell -- independent draws share only the
        # informative-feature indices and base coefficient direction for a
        # given seed, never data rows; much weaker dependence than the real
        # benchmark's shared training pool.
        cell = paired_diffs(scores, arm, control, alpha, ['seed', 'target'])
        r = tost(cell.to_numpy(), MARGIN)
        r.update({'arm': arm, 'control': control, 'alpha': alpha, 'unit': 'seed_target_cell'})
        rows.append(r)

        # stricter unit: seed-level mean (n=3), where independence is
        # unambiguous, reported alongside for transparency about the
        # dependence structure of the primary unit.
        seed_level = paired_diffs(scores, arm, control, alpha, ['seed', 'target']) \
            .groupby('seed').mean()
        r2 = tost(seed_level.to_numpy(), MARGIN)
        r2.update({'arm': arm, 'control': control, 'alpha': alpha, 'unit': 'seed_level_n3'})
        rows.append(r2)

    df = pd.DataFrame(rows)
    df.to_csv(DEST / 'tost_results.csv', index=False)

    primary = df[df.unit == 'seed_target_cell']
    verdict = {
        'margin': MARGIN, 'alpha': ALPHA,
        'margin_source': 'PREREGISTRATION.md predeclared 0.02 ROC-AUC ceiling, fixed '
                         'before this script existed',
        'weighting_equivalent_at_every_alpha': bool(
            primary[primary.arm == 'shap_sum'].equivalent_at_margin.all()),
        'selection_equivalent_at_every_alpha': bool(
            primary[primary.arm == 'select_shap'].equivalent_at_margin.all()),
        'per_alpha_summary': primary[['arm', 'control', 'alpha', 'mean', 'p_tost',
                                     'equivalent_at_margin']].to_dict('records'),
        'note': ('Equivalence at a 0.02 margin says the effect is formally indistinguishable '
                'from a magnitude no reviewer already treats as material on this benchmark; '
                'it is not evidence the effect is exactly zero, and it is computed only on '
                'the synthetic benchmark, never on the real cross-project targets.'),
    }
    dump(DEST / 'verdict.json', verdict)

    print(f"weighting equivalent at every alpha (margin={MARGIN}): "
          f"{verdict['weighting_equivalent_at_every_alpha']}")
    print(f"selection equivalent at every alpha (margin={MARGIN}): "
          f"{verdict['selection_equivalent_at_every_alpha']}")
    print(primary[['arm', 'control', 'alpha', 'n', 'mean', 'p_tost',
                   'equivalent_at_margin']].to_string(index=False))
    print(f'\nwrote {DEST}')
    return verdict


if __name__ == '__main__':
    run()
