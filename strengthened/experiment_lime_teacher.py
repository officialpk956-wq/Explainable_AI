"""RQ-M: does the small weighting/selection effect generalize beyond SHAP as
the transferred attribution mechanism? Every prior run (v1/v2/v3, the
decorrelation and label-noise sensitivities, and experiment_linear_teacher's
new teacher) has used SHAP -- TreeExplainer or LinearExplainer -- to build the
per-feature importance vector that drives shap_sum/select_shap. This module
holds the teacher fixed at the study's own RandomForest and swaps ONLY the
attribution mechanism to LIME (LimeTabularExplainer), so it isolates the
attribution-method effect cleanly, orthogonal to experiment_linear_teacher's
model-family ablation.

python -m strengthened.experiment_lime_teacher --workers 10

BreakDown has no maintained Python package (checked: neither `breakdown` nor
`pyBreakDown` is installable in this environment) -- this is disclosed here
rather than silently dropped. Only LIME is implemented.

Exploratory extension, like v3 and experiment_linear_teacher: a first-time
measurement, no protocol_lime_teacher.json, no predeclared threshold.

Cost: unlike shap.TreeExplainer, which returns every row's attribution in one
vectorized call, LIME explains one instance at a time (its own local
perturbation sampling and ridge-regression fit per call). Benchmarked at
~0.03-0.06s/instance against this study's RandomForest teacher (32 trees,
depth 5) -- see the module docstring history for the benchmark script. That
keeps the full v3 grid (7 targets x 10 seeds, identical to the main study)
tractable, so no seed or target reduction was needed; unlike
experiment_decorrelation/experiment_labelnoise this required no scope cut.

Reduced to the same weighting/selection arm family as experiment_linear_teacher
and items 7/8: augmentation arms (p_aug/ps_aug/pn_aug/pr_aug) require a
per-row attribution matrix at prediction time for arms that don't otherwise
need one, and are orthogonal to the attribution-mechanism question here, so
they are dropped. Arms are named with a `lime_` prefix (lime_sum,
select_lime, shuffled_lime) instead of reusing the shap_* names, so a scores.csv
row is unambiguous about which attribution mechanism produced it even outside
this module's own output directory.

Structurally this module copies experiment_v3.representations()/fit_predict()/
task() (same reason experiment_linear_teacher.py gives: those functions call
their own module-level teacher()/shap_values() by name, hardcoded to a tree
model and TreeExplainer, and editing experiment_v3.py in place would
invalidate the fingerprint of the already-completed main run and its
sensitivities). learner(), permute_within_groups(), streams() and teacher()
are imported and reused UNCHANGED from experiment_v3 -- the teacher model
itself is untouched; only how it is explained changes.
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from threadpoolctl import threadpool_limits

from .experiment import ROOT, DATA, digest, dump, load, metric
from .experiment_v3 import (config as v3_config, learner, permute_within_groups,
                            streams, teacher)

OUT = ROOT / 'outputs/revised/strengthening_v3/lime_teacher'
REDUCED_ARMS = ['original', 'lime_sum', 'uniform_sum', 'shuffled_lime', 'select_lime', 'select_random']
COMPARISONS = [('lime_sum', 'uniform_sum'), ('lime_sum', 'shuffled_lime'),
              ('select_lime', 'select_random')]
LIME_NUM_SAMPLES = 1000  # benchmarked: ~0.03s/instance at this budget against the RF teacher


def config():
    return v3_config(arms=REDUCED_ARMS, multi_draw_arms=[], control_draws=1,
                     teachers=['RandomForest'])


def lime_attribution(explainer, model, x):
    """One |LIME weight| row per instance in x, aligned to x's column order."""
    out = np.zeros_like(x)
    for i in range(len(x)):
        e = explainer.explain_instance(x[i], model.predict_proba, num_features=x.shape[1],
                                       num_samples=LIME_NUM_SAMPLES, labels=(1,))
        for idx, w in e.local_exp[1]:
            out[i, idx] = w
    return out


def representations(x, y, z, cfg, seed, tname, groups=None):
    """Mirrors experiment_v3.representations(), restricted to REDUCED_ARMS, with
    the teacher's own TreeExplainer SHAP swapped for LIME."""
    rng, entropy, ints = streams(seed, cfg)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    a = imputer.transform(x); b = imputer.transform(z)
    keep = np.flatnonzero(np.std(a, axis=0) > 0)
    if not len(keep):
        raise ValueError('no varying source metrics')
    a = a[:, keep]; b = b[:, keep]

    splitter = StratifiedKFold(cfg['oof_folds'], shuffle=True,
                               random_state=ints['teacher']).split(a, y)
    pp = np.empty(len(y)); ss = np.empty_like(a); folds = []
    for tr, va in splitter:
        fold_imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(x[tr])
        at = fold_imp.transform(x[tr])[:, keep]; av = fold_imp.transform(x[va])[:, keep]
        t = teacher(tname, cfg, ints['teacher']); t.fit(at, y[tr])
        pp[va] = t.predict_proba(av)[:, 1]
        explainer = LimeTabularExplainer(at, mode='classification', discretize_continuous=False,
                                         feature_names=[str(i) for i in range(at.shape[1])],
                                         random_state=np.random.RandomState(ints['teacher']))
        ss[va] = lime_attribution(explainer, t, av)
        folds.append({'train': tr.tolist(), 'validation': va.tolist()})
    t = teacher(tname, cfg, ints['teacher']); t.fit(a, y)
    pt = t.predict_proba(b)[:, 1]
    explainer = LimeTabularExplainer(a, mode='classification', discretize_continuous=False,
                                     feature_names=[str(i) for i in range(a.shape[1])],
                                     random_state=np.random.RandomState(ints['teacher']))
    st = lime_attribution(explainer, t, b)

    w = np.maximum(np.abs(ss).mean(axis=0), 1e-12); w = w / w.sum()
    sel = rng['selection']
    shuffle = w[sel.permutation(len(w))]
    k = int(np.ceil(len(w) * cfg['selection_fraction']))
    top = np.argsort(-w, kind='stable')[:k]
    random = sel.choice(len(w), k, replace=False)

    aa = {'original': a, 'lime_sum': a * w, 'uniform_sum': a / len(w), 'shuffled_lime': a * shuffle,
          'select_lime': a[:, top], 'select_random': a[:, random]}
    bb = {'original': b, 'lime_sum': b * w, 'uniform_sum': b / len(w), 'shuffled_lime': b * shuffle,
          'select_lime': b[:, top], 'select_random': b[:, random]}

    corr = pd.DataFrame(a).corr(method='spearman').abs().fillna(0).to_numpy()
    audit = {'keep': keep.tolist(), 'weights': w.tolist(), 'shuffled_weights': shuffle.tolist(),
             'selected': top.tolist(), 'random_selected': random.tolist(), 'oof': folds,
             'teacher': tname, 'attribution': 'LimeTabularExplainer',
             'lime_num_samples': LIME_NUM_SAMPLES,
             'source_redundancy': float(corr[np.triu_indices(len(w), 1)].mean()),
             'stream_entropy': entropy, 'stream_seeds': ints}
    return aa, bb, audit


def fit_predict(x, y, groups, target_x, cfg, seed, tname):
    rng, _, ints = streams(seed, cfg)
    fitted = cfg['models']
    learned = [a for a in cfg['arms'] if a != 'prevalence']
    permuted = permute_within_groups(y, groups, rng['labelperm'])
    labels_for = lambda arm: permuted if arm == 'permuted_original' else y

    cv = list(GroupKFold(cfg['outer_tuning_folds']).split(x, y, groups))
    candidate = {(m, a, c): [] for m in fitted for a in learned for c in range(2)}
    folds = []
    for tr, va in cv:
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError('project leakage in tuning')
        aa, bb, rep = representations(x[tr], y[tr], x[va], cfg, seed, tname, groups[tr])
        for m, a, c in candidate:
            ytr, yva = labels_for(a)[tr], labels_for(a)[va]
            model = learner(m, c, cfg, ints['downstream']); model.fit(aa[a], ytr)
            candidate[m, a, c].append(float(roc_auc_score(yva, model.predict_proba(bb[a])[:, 1])))
        folds.append({'train': tr.tolist(), 'validation': va.tolist(), 'representation': rep})

    choice = {(m, a): int(np.argmax([np.mean(candidate[m, a, c]) for c in range(2)]))
              for m in fitted for a in learned}
    aa, bb, rep = representations(x, y, target_x, cfg, seed, tname, groups)

    predictions, selected = {}, {}
    for m in fitted:
        for a in learned:
            for c in set([choice[m, a], choice[m, 'original']]):
                model = learner(m, c, cfg, ints['downstream'])
                model.fit(aa[a], labels_for(a)); prob = model.predict_proba(bb[a])[:, 1]
                for regime in cfg['tuning_regimes']:
                    want = choice[m, a] if regime == 'independent' else choice[m, 'original']
                    if want == c:
                        predictions['__'.join([regime, m, a, '0'])] = prob
                        selected['__'.join([regime, m, a])] = c

    audit = {'tuning': folds, 'chosen': selected, 'final_representation': rep,
             'candidate_auc': {'__'.join([m, a, str(c)]): v for (m, a, c), v in candidate.items()}}
    return predictions, audit


def fingerprint(cfg):
    here = Path(__file__).parent
    paths = [Path(__file__), here / 'experiment_v3.py', here / 'experiment.py',
             here / 'experiment_v2.py', here / 'protocol_v3.json', here / 'protocol.json',
             here / 'SELECTION.json'] + [DATA / (n + '.csv') for n in cfg['projects']]
    from importlib.metadata import version
    return {'files': {str(p.relative_to(ROOT)): digest(p) for p in paths},
            'packages': {n: version(n) for n in
                         ['numpy', 'pandas', 'scikit-learn', 'scipy', 'lime', 'xgboost']},
            'lime_num_samples': LIME_NUM_SAMPLES, 'config': cfg}


def task(target, seed, tname, out, cfg, fp):
    warnings.filterwarnings('error', category=ConvergenceWarning)
    start = time.time(); stem = out / f'{target}__{seed}__{tname}'
    path = lambda ext: Path(str(stem) + ext)
    if path('.done.json').exists():
        d = json.loads(path('.done.json').read_text())
        if d['fingerprint'] != fp:
            raise RuntimeError('checkpoint fingerprint mismatch')
        if any(digest(out / n) != v for n, v in d['hashes'].items()):
            raise RuntimeError('checkpoint corrupted')
        return f'cached {target} {seed} {tname}'
    rng, _, ints = streams(seed, cfg)
    xx, yy, gg, indices = [], [], [], {}
    for name in cfg['projects']:
        if name == target:
            continue
        x, y, _ = load(name, cfg); ix = np.arange(len(y))
        if len(ix) > cfg['source_cap_per_project']:
            ix, _ = train_test_split(ix, train_size=cfg['source_cap_per_project'],
                                     stratify=y, random_state=ints['sampling'])
        xx.append(x[ix]); yy.append(y[ix]); gg.extend([name] * len(ix)); indices[name] = ix.tolist()
    x = np.concatenate(xx); y = np.concatenate(yy); g = np.asarray(gg)
    xt, yt, it = load(target, cfg)
    with threadpool_limits(limits=1):
        pred, audit = fit_predict(x, y, g, xt, cfg, seed, tname)
    rows = []
    for key, prob in pred.items():
        regime, m, a, d = key.split('__')
        rows.append({'target': target, 'seed': seed, 'teacher': tname, 'tuning': regime,
                     'model': m, 'arm': a, 'draw': int(d), **metric(yt, prob)})
    pred['labels'] = yt; pred['target_ids'] = it
    np.savez_compressed(path('.npz'), **pred)
    pd.DataFrame(rows).to_csv(path('.csv'), index=False)
    audit.update({'target': target, 'seed': seed, 'teacher': tname, 'source_indices': indices,
                  'source_groups': g.tolist(), 'source_labels': y.tolist(),
                  'target_rows': len(yt), 'seconds': time.time() - start})
    dump(path('.audit.json'), audit)
    dump(path('.done.json'), {'fingerprint': fp,
                              'hashes': {path(e).name: digest(path(e))
                                         for e in ['.csv', '.npz', '.audit.json']}})
    return f'completed {target} seed={seed} teacher={tname} ({time.time() - start:.1f}s)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    cfg = config()
    out = OUT
    if args.smoke:
        cfg.update(downstream_trees=4, teacher_trees=4, seeds=cfg['seeds'][:1])
        out = OUT.parent / 'lime_teacher_smoke'
    out.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(cfg); mp = out / 'manifest.json'
    if mp.exists():
        prior = json.loads(mp.read_text())
        if prior.get('status') == 'complete' and prior.get('fingerprint') == fp:
            print('cached: manifest already complete with a matching fingerprint')
            return
        if prior.get('fingerprint') and prior.get('fingerprint') != fp:
            raise RuntimeError('use a new output directory after execution changes')
    dump(mp, {'status': 'running', 'fingerprint': fp,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'smoke': args.smoke})
    targets = cfg['projects'][:1] if args.smoke else cfg['projects']
    jobs = [(t, s, n, out, cfg, fp) for t in targets for s in cfg['seeds'] for n in cfg['teachers']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in concurrent.futures.as_completed([ex.submit(task, *j) for j in jobs]):
            print(f.result(), flush=True)
    scores = pd.concat([pd.read_csv(Path(str(out / f'{t}__{s}__{n}') + '.csv'))
                        for t, s, n, *_ in jobs], ignore_index=True)
    scores.to_csv(out / 'scores.csv', index=False)
    dump(mp, {'status': 'complete', 'fingerprint': fp,
              'completed_utc': datetime.now(timezone.utc).isoformat(),
              'smoke': args.smoke, 'rows': len(scores)})
    print('COMPLETE', len(scores), flush=True)


if __name__ == '__main__':
    main()
