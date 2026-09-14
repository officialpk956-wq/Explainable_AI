"""LIME instability against surrogate fit, on the external benchmark.

python -m strengthened.experiment_lime --workers 8

Measured explanation instability mixes three sources of variation. This separates
them and reports each: the explainer's own sampling randomness, the randomness of
the model being explained, and the choice of explained instances. The headline
correlation uses explainer randomness alone, because that is what the stability
literature reports; mixing the others in would confound the confound.

The target response is never used for fitting, instance selection or explanation.
"""
import argparse
import concurrent.futures
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .experiment import ROOT, digest, dump, load
from .experiment_v3 import learner

OUT = ROOT / 'outputs/revised/lime_confound'


def config():
    here = Path(__file__).parent
    cfg = json.loads((here / 'protocol_lime.json').read_text())
    cfg['features'] = json.loads((here / 'protocol.json').read_text())['features']
    cfg['candidates'] = json.loads((here / 'protocol_v3.json').read_text())['candidates']
    cfg['downstream_trees'] = 32
    cfg['svc_calibration_folds'] = 3
    return cfg


def source_matrix(target, cfg, seed=101):
    xx, yy = [], []
    for name in cfg['projects']:
        if name == target:
            continue
        x, y, _ = load(name, cfg)
        ix = np.arange(len(y))
        if len(ix) > cfg['source_cap_per_project']:
            ix, _ = train_test_split(ix, train_size=cfg['source_cap_per_project'],
                                     stratify=y, random_state=seed)
        xx.append(x[ix]); yy.append(y[ix])
    return np.concatenate(xx), np.concatenate(yy)


def prepared(target, cfg, seed=101):
    """Impute on source only; the target contributes covariates, never labels."""
    xs, ys = source_matrix(target, cfg, seed)
    xt, yt, ids = load(target, cfg)
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(xs)
    a, b = imp.transform(xs), imp.transform(xt)
    keep = np.flatnonzero(np.std(a, axis=0) > 0)
    return a[:, keep], ys, b[:, keep], yt, ids, keep


def instance_indices(ids, cfg, which=0):
    """Deterministic, response independent: evenly spaced after sorting by identifier."""
    order = np.argsort(ids, kind='stable')
    n = cfg['n_instances']
    total = cfg['n_instance_sets']
    pos = np.linspace(0, len(order) - 1, n * total).astype(int)
    return order[pos[which::total]]


def vectors_and_fit(model, explainer, x, rows, cfg, lime_seed):
    """One attribution vector and one surrogate R^2 per requested row."""
    explainer.random_state = np.random.RandomState(lime_seed)
    vecs, fits = [], []
    for r in rows:
        e = explainer.explain_instance(x[r], model.predict_proba,
                                       num_features=x.shape[1],
                                       num_samples=cfg['lime_num_samples'],
                                       labels=(1,))
        v = np.zeros(x.shape[1])
        for idx, w in e.local_exp[1]:
            v[idx] = w
        norm = np.linalg.norm(v)
        vecs.append(v / norm if norm > 0 else v)
        fits.append(float(e.score))
    return np.asarray(vecs), float(np.mean(fits))


def sigma_bar(stack):
    """stack: (variants, instances, features) -> mean per-feature SD, averaged over instances."""
    return float(np.mean(np.std(stack, axis=0)))


def cell(target, model_name, cfg):
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('error', category=ConvergenceWarning)
    start = time.time()
    a, ys, b, yt, ids, keep = prepared(target, cfg)
    explainer = LimeTabularExplainer(a, mode='classification', discretize_continuous=False,
                                     feature_names=[str(i) for i in keep],
                                     random_state=np.random.RandomState(0))
    base = learner(model_name, cfg['model_candidate_index'], cfg, 101)
    base.fit(a, ys)
    rows = instance_indices(ids, cfg, which=0)

    # (1) explainer randomness: one model, one instance set, many LIME seeds
    stack, fits = [], []
    for s in range(cfg['n_explainer_seeds']):
        v, f = vectors_and_fit(base, explainer, b, rows, cfg, 1000 + s)
        stack.append(v); fits.append(f)
    sigma_explainer = sigma_bar(np.asarray(stack))
    r2 = float(np.mean(fits))

    # (2) model-training randomness: one LIME seed, one instance set, refit the model
    stack = []
    for s in range(cfg['n_model_seeds']):
        m = learner(model_name, cfg['model_candidate_index'], cfg, 200 + s)
        m.fit(a, ys)
        v, _ = vectors_and_fit(m, explainer, b, rows, cfg, 1000)
        stack.append(v)
    sigma_model = sigma_bar(np.asarray(stack))

    # (3) instance choice: one model, one LIME seed, disjoint instance sets
    per_set = []
    for w in range(cfg['n_instance_sets']):
        v, _ = vectors_and_fit(base, explainer, b, instance_indices(ids, cfg, which=w),
                               cfg, 1000)
        per_set.append(v.mean(axis=0))
    sigma_instances = sigma_bar(np.asarray(per_set)[:, None, :])

    return {'target': target, 'model': model_name, 'surrogate_r2': r2,
            'sigma_explainer': sigma_explainer, 'sigma_model': sigma_model,
            'sigma_instances': sigma_instances,
            'instances': len(rows), 'explainer_seeds': cfg['n_explainer_seeds'],
            'seconds': time.time() - start}


def task(args):
    target, model_name, cfg = args
    with threadpool_limits(limits=1):
        return cell(target, model_name, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = config()
    out = OUT
    if a.smoke:
        cfg.update(n_instances=2, n_explainer_seeds=3, n_model_seeds=2, n_instance_sets=2,
                   lime_num_samples=300, projects=cfg['projects'][:2])
        out = OUT.parent / 'lime_confound_smoke'
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(t, m, cfg) for t in cfg['projects'] for m in cfg['models']]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(task, jobs):
            rows.append(r)
            print(f"  {r['target']:<16}{r['model']:<20}R2={r['surrogate_r2']:.3f} "
                  f"sigma={r['sigma_explainer']:.5f} ({r['seconds']:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'cells.csv', index=False)

    per_target = []
    for target, d in df.groupby('target'):
        if d.model.nunique() < 3:
            continue
        rho = spearmanr(d.surrogate_r2, d.sigma_explainer).statistic
        per_target.append({'target': target, 'models': int(d.model.nunique()),
                           'spearman_r2_vs_sigma': float(rho)})
    pt = pd.DataFrame(per_target)
    pt.to_csv(out / 'per_target.csv', index=False)

    neg = int((pt.spearman_r2_vs_sigma < 0).sum())
    verdict = {
        'targets': len(pt), 'negative_correlations': neg,
        'median_spearman': float(pt.spearman_r2_vs_sigma.median()),
        'supported': bool(neg > len(pt) / 2),
        'variance_sources': {
            'explainer': float(df.sigma_explainer.mean()),
            'model_training': float(df.sigma_model.mean()),
            'instance_choice': float(df.sigma_instances.mean())},
        'note': ('Descriptive. Seven targets sharing source projects, one explainer, one '
                 'benchmark. No significance test, no causal claim.'),
        'completed_utc': datetime.now(timezone.utc).isoformat()}
    dump(out / 'verdict.json', verdict)
    print(pt.to_string(index=False))
    print(json.dumps(verdict, indent=2))


if __name__ == '__main__':
    main()
