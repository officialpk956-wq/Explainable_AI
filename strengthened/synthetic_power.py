"""Does the matched-control battery have the power to detect a real attribution benefit?

python -m strengthened.synthetic_power --workers 6

The harness controls in the main study show that the design registers the loss of
source label signal. They do not show that the *matched* controls could detect a
genuine attribution benefit if one existed, so a near-zero
select_shap-versus-select_random result is still open to the objection that the
comparison is simply insensitive.

This closes that. Synthetic projects are generated in which a known subset of
features drives the label, so a competent attribution method must identify that
subset and attribution-guided selection must beat random selection. Signal
strength is swept, including a zero-signal level at which the battery must report
nothing. The same fit_predict used on the real benchmark is called here, so the
result describes the actual design rather than a reimplementation of it.

The output is a detection curve: the effect the battery recovers at each true
signal strength, against the effect observed on the real benchmark.
"""
import argparse
import concurrent.futures
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .experiment import ROOT, dump, metric
from .experiment_v3 import config as v3_config, fit_predict

OUT = ROOT / 'outputs/revised/synthetic_power'
N_PROJECTS = 7
N_FEATURES = 54
N_INFORMATIVE = 10
# six source projects at this size give 2,400 training rows, matching the real
# study's source cap exactly, so the power estimate describes that design
N_ROWS = 400
N_FACTORS = 8
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]
SEEDS = [101, 102, 103]
ARMS = ['original', 'shap_sum', 'uniform_sum', 'shuffled_sum', 'select_shap', 'select_random']
TARGET_PREVALENCE = 0.25


def generate(alpha, seed):
    """Seven correlated-feature projects whose labels depend only on a known subset.

    Real code metrics are strongly collinear, so features are drawn from a factor
    model rather than independently; independent features would make attribution
    guided selection trivially easy and overstate the battery's power.
    """
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(N_FACTORS, N_FEATURES))
    informative = np.sort(rng.choice(N_FEATURES, N_INFORMATIVE, replace=False))
    beta = np.zeros(N_FEATURES)
    beta[informative] = rng.normal(0, 1, N_INFORMATIVE)

    projects = []
    for p in range(N_PROJECTS):
        prng = np.random.default_rng(seed * 1000 + p)
        # project-specific covariate shift and a mild coefficient perturbation,
        # so this is a transfer problem rather than one pooled population
        shift = prng.normal(0, .4, N_FEATURES)
        scale = np.exp(prng.normal(0, .2, N_FEATURES))
        factors = prng.normal(size=(N_ROWS, N_FACTORS))
        x = (factors @ loadings + prng.normal(0, .6, (N_ROWS, N_FEATURES))) * scale + shift
        b = beta * np.exp(prng.normal(0, .15, N_FEATURES))
        # fixed intercept, so class balance does not drift with signal strength;
        # otherwise alpha would change prevalence and effect size together
        z = x @ b
        logit = alpha * (z - z.mean()) / max(np.std(z), 1e-9) + np.log(
            TARGET_PREVALENCE / (1 - TARGET_PREVALENCE))
        y = (prng.random(N_ROWS) < 1 / (1 + np.exp(-logit))).astype(int)
        if min(np.bincount(y, minlength=2)) < 20:      # keep every project usable
            y[prng.choice(N_ROWS, 20, replace=False)] = 1 - y[0]
        projects.append((x, y))
    return projects, informative


def cell(args):
    alpha, seed, target, teacher, cfg = args
    with threadpool_limits(limits=1):
        projects, informative = generate(alpha, seed)
        xs, ys, gg = [], [], []
        for i, (x, y) in enumerate(projects):
            if i == target:
                continue
            xs.append(x); ys.append(y); gg.extend([f'p{i}'] * len(y))
        x = np.concatenate(xs); y = np.concatenate(ys); g = np.asarray(gg)
        xt, yt = projects[target]
        pred, audit = fit_predict(x, y, g, xt, cfg, seed, teacher)
        rows = []
        for key, prob in pred.items():
            regime, m, arm, draw = key.split('__')
            rows.append({'alpha': alpha, 'seed': seed, 'target': f'p{target}',
                         'teacher': teacher, 'tuning': regime, 'model': m, 'arm': arm,
                         **metric(yt, prob)})
        # did the teacher actually find the informative features?
        w = np.asarray(audit['final_representation']['weights'])
        keep = np.asarray(audit['final_representation']['keep'])
        rank = {f: i for i, f in enumerate(keep[np.argsort(-w, kind='stable')])}
        hits = sum(1 for f in informative if rank.get(f, 10 ** 9) < N_INFORMATIVE)
        return rows, {'alpha': alpha, 'seed': seed, 'target': f'p{target}', 'teacher': teacher,
                      'informative_recovered': hits, 'informative_total': N_INFORMATIVE}


def run(workers):
    cfg = v3_config(arms=ARMS, control_draws=1)
    cfg['multi_draw_arms'] = []
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(a, s, t, n, cfg) for a in ALPHAS for s in SEEDS
            for t in range(N_PROJECTS) for n in cfg['teachers']]
    rows, recovery = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for r, rec in ex.map(cell, jobs):
            rows.extend(r); recovery.append(rec)
            if len(recovery) % 14 == 0:
                print(f'  {len(recovery)}/{len(jobs)} cells', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'scores.csv', index=False)
    pd.DataFrame(recovery).to_csv(OUT / 'recovery.csv', index=False)
    return pd.DataFrame(rows), pd.DataFrame(recovery)


def collect(scores=None, recovery=None):
    scores = pd.read_csv(OUT / 'scores.csv') if scores is None else scores
    recovery = pd.read_csv(OUT / 'recovery.csv') if recovery is None else recovery
    PK = ['alpha', 'seed', 'target', 'teacher', 'tuning', 'model']
    M = ['auc', 'average_precision', 'f1_macro']
    pairs = [('select_shap', 'select_random'), ('shap_sum', 'uniform_sum'),
             ('shap_sum', 'shuffled_sum')]
    rows = []
    s = scores[scores.tuning == 'independent']
    scale_sensitive = v3_config()['scale_sensitive_models']
    for arm, control in pairs:
        for alpha, d in s.groupby('alpha'):
            m = d[d.arm == arm].merge(d[d.arm == control], on=PK, suffixes=('_a', '_b'),
                                      validate='one_to_one')
            per_target = (m.auc_a - m.auc_b).groupby(m.target).mean()
            sub = m[m.model.isin(scale_sensitive)]
            rows.append({'arm': arm, 'control': control, 'alpha': alpha,
                         'mean_difference': float(per_target.mean()),
                         'target_min': float(per_target.min()),
                         'target_max': float(per_target.max()),
                         'targets_positive': int((per_target > 0).sum()),
                         'scale_sensitive_only': float(
                             (sub.auc_a - sub.auc_b).groupby(sub.target).mean().mean())
                         if len(sub) else float('nan'),
                         'baseline_auc': float(d[d.arm == 'original'].auc.mean())})
    curve = pd.DataFrame(rows)
    curve.to_csv(OUT / 'detection_curve.csv', index=False)

    rec = recovery.groupby('alpha').informative_recovered.mean()
    sel = curve[(curve.arm == 'select_shap') & (curve.control == 'select_random')].set_index('alpha')
    null_effect = float(sel.loc[0.0, 'mean_difference']) if 0.0 in sel.index else float('nan')
    verdict = {
        'alphas': ALPHAS, 'informative_features': N_INFORMATIVE, 'features': N_FEATURES,
        'projects': N_PROJECTS, 'seeds': SEEDS,
        'zero_signal_effect': null_effect,
        'zero_signal_recovered': float(rec.get(0.0, float('nan'))),
        'detection_curve': {str(a): float(sel.loc[a, 'mean_difference'])
                            for a in sel.index},
        'recovered_informative': {str(a): float(rec.get(a, float('nan'))) for a in sel.index},
        'note': ('At alpha=0 the label does not depend on the features, so attribution guided '
                 'selection must show no advantage; a non-zero effect there would indicate the '
                 'battery is biased rather than insensitive. At larger alpha the attributions '
                 'identify the informative subset and the battery must register the advantage. '
                 'The largest effect the battery recovers bounds what it is capable of showing '
                 'on this design.'),
        'completed_utc': datetime.now(timezone.utc).isoformat()}
    dump(OUT / 'verdict.json', verdict)
    print(curve.to_string(index=False))
    print()
    print(json.dumps(verdict, indent=2))
    return curve


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--collect-only', action='store_true')
    a = ap.parse_args()
    if a.collect_only:
        collect()
    else:
        s, r = run(a.workers)
        collect(s, r)
