"""RQ-L: does the small shap_sum/select_shap effect survive a non-tree
attribution mechanism? v1/v2/v3 have only ever used tree teachers
(RandomForest, ExtraTrees) explained with shap.TreeExplainer. This is the
sharpest reviewer threat available: tree-SHAP is known to be distorted by
correlated features, so a null weighting/selection result could just be an
artifact of that one mechanism. This module adds a LogisticRegression teacher
explained with shap.LinearExplainer(feature_perturbation='interventional') --
an entirely different model family and a different SHAP algorithm -- and reuses
the same weighting/selection arms unchanged.

python -m strengthened.experiment_linear_teacher --workers 10

Exploratory extension, like v3 itself: this is a first-time measurement with
no prior same-mechanism result to preregister a ceiling against, so there is
no protocol_linear.json and no predeclared threshold. It is reported
descriptively against the existing tree-teacher effect sizes, the same way
v3's own protocol document describes v3 as "exploratory... not a
preregistration."

experiment_v3.representations()/fit_predict() call their own module-level
teacher()/shap_values() by name, both hardcoded to a tree model and
TreeExplainer. Extending them in place to dispatch on teacher name would edit
experiment_v3.py itself and invalidate the fingerprint of the
already-completed v3 main run and every sensitivity built on it (decorrelation,
labelnoise). Instead this module defines its own representations()/
fit_predict()/task(), copied from experiment_v3.py with only the
teacher-and-attribution step factored into teacher_and_attribution(), which
dispatches on teacher name; learner(), permute_within_groups() and streams()
are imported and reused UNCHANGED. Reduced to the same six arms items 7/8
used (original + the two weighting/selection comparisons and their controls),
since the augmentation/coral arms are orthogonal to the attribution-mechanism
question this module tests.

The linear teacher is not scale-sensitive-by-construction like the downstream
LogisticRegression/SVCRBF models: it is fit on RobustScaler-transformed source
features purely so its own coefficients (and therefore its SHAP attribution)
are not dominated by raw metric scale. That scaling is internal to attribution
weight computation only -- the returned SHAP matrix is used exactly as the
tree teacher's is, to build a per-feature importance vector w, and w is
applied to the ORIGINAL (unscaled) feature matrix, exactly as v3 does.
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
import shap
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits

from .experiment import ROOT, DATA, digest, dump, load, metric, signed_log
from .experiment_v3 import config as v3_config, learner, permute_within_groups, streams

OUT = ROOT / 'outputs/revised/strengthening_v3/linear_teacher'
REDUCED_ARMS = ['original', 'shap_sum', 'uniform_sum', 'shuffled_sum', 'select_shap', 'select_random']
COMPARISONS = [('shap_sum', 'uniform_sum'), ('shap_sum', 'shuffled_sum'),
              ('select_shap', 'select_random')]
LINEAR_C = 1.0  # fixed, not tuned -- matches teacher_trees/teacher_depth being fixed constants


def config():
    return v3_config(arms=REDUCED_ARMS, multi_draw_arms=[], control_draws=1,
                     teachers=['LinearLogit'])


def teacher_and_attribution(cfg, seed, x_train, y_train, x_eval):
    """Fit the linear teacher on x_train and return (prob_on_x_eval, shap_on_x_eval).
    Preprocessing (signed_log then RobustScaler) matches the codebase's existing
    convention for every other linear learner (see experiment_v3.learner)."""
    scaler = RobustScaler().fit(signed_log(x_train))
    at, ae = scaler.transform(signed_log(x_train)), scaler.transform(signed_log(x_eval))
    clf = LogisticRegression(C=LINEAR_C, class_weight='balanced', solver='liblinear',
                             max_iter=20000, random_state=seed)
    clf.fit(at, y_train)
    # Explicit masker with max_samples=len(at): shap.LinearExplainer's legacy
    # feature_perturbation='interventional' path silently subsamples the
    # background to 100 rows by default, which would make the attribution
    # depend on an RNG this module does not control or seed. Using the whole
    # training fold as background is still cheap -- interventional/independent
    # masking for a linear model is a closed-form function of the background
    # mean, not a Monte Carlo sampler.
    masker = shap.maskers.Independent(at, max_samples=len(at))
    explainer = shap.LinearExplainer(clf, masker)
    prob = clf.predict_proba(ae)[:, 1]
    sv = np.asarray(explainer.shap_values(ae))
    if sv.shape != ae.shape or not np.isfinite(sv).all():
        raise ValueError('linear SHAP output invalid')
    return prob, sv


def representations(x, y, z, cfg, seed, tname, groups=None):
    """Fitted on x only. z contributes covariates, never labels. Mirrors
    experiment_v3.representations() but restricted to REDUCED_ARMS and routed
    through teacher_and_attribution() instead of the tree-only teacher()/
    shap_values() pair."""
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
        pp[va], ss[va] = teacher_and_attribution(cfg, ints['teacher'], at, y[tr], av)
        folds.append({'train': tr.tolist(), 'validation': va.tolist()})
    pt, st = teacher_and_attribution(cfg, ints['teacher'], a, y, b)

    w = np.maximum(np.abs(ss).mean(axis=0), 1e-12); w = w / w.sum()
    sel = rng['selection']
    shuffle = w[sel.permutation(len(w))]
    k = int(np.ceil(len(w) * cfg['selection_fraction']))
    top = np.argsort(-w, kind='stable')[:k]
    random = sel.choice(len(w), k, replace=False)

    aa = {'original': a, 'shap_sum': a * w, 'uniform_sum': a / len(w), 'shuffled_sum': a * shuffle,
          'select_shap': a[:, top], 'select_random': a[:, random]}
    bb = {'original': b, 'shap_sum': b * w, 'uniform_sum': b / len(w), 'shuffled_sum': b * shuffle,
          'select_shap': b[:, top], 'select_random': b[:, random]}

    corr = pd.DataFrame(a).corr(method='spearman').abs().fillna(0).to_numpy()
    audit = {'keep': keep.tolist(), 'weights': w.tolist(), 'shuffled_weights': shuffle.tolist(),
             'selected': top.tolist(), 'random_selected': random.tolist(), 'oof': folds,
             'teacher': 'LinearLogit', 'attribution': 'shap.LinearExplainer(interventional)',
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
                         ['numpy', 'pandas', 'scikit-learn', 'scipy', 'shap', 'xgboost']},
            'config': cfg}


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
        cfg.update(downstream_trees=4, seeds=cfg['seeds'][:1])
        out = OUT.parent / 'linear_teacher_smoke'
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
