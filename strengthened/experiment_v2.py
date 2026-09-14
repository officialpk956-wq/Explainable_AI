"""External benchmark, v2: harness controls, four learners, ten source seeds.

python -m strengthened.experiment_v2 --workers 10

v1 remains untouched under outputs/revised/strengthening/full. This module writes
to a separate directory with its own fingerprint, so no completed checkpoint is
invalidated or overwritten.

Four arms are new. prevalence and permuted_original establish whether the harness
registers an uninformative predictor and total source-signal loss; coral is a
documented adaptation comparator; pn_aug matches ps_aug in column count using
uninformative draws, separating an attribution-specific augmentation effect from a
generic added-dimension penalty.

fit_predict still takes no target response. coral is the only arm that reads target
covariates, and it reads them without labels.
"""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import argparse
import concurrent.futures
import hashlib
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .experiment import (ROOT, DATA, digest, dump, load, metric, teacher,
                         shap_values, signed_log)

OUT = ROOT / 'outputs/revised/strengthening_v2/full'
NOISE_STREAM = 90000          # keeps the pn_aug generator distinct from the weight shuffle


def config(cap=None, seeds=None):
    here = Path(__file__).parent
    cfg = json.loads((here / 'protocol_v2.json').read_text())
    cfg['features'] = json.loads((here / 'protocol.json').read_text())['features']
    if cap is not None:
        cfg['source_cap_per_project'] = cap
    if seeds is not None:
        cfg['seeds'] = seeds
    return cfg


def coral_map(source, target, ridge):
    """Whiten source covariance, recolour with target covariance. Target untouched."""
    def power(c, p):
        v, q = np.linalg.eigh(np.atleast_2d(c) + ridge * np.eye(len(c)))
        return (q * np.maximum(v, 1e-12) ** p) @ q.T
    mapped = ((source - source.mean(axis=0))
              @ power(np.cov(source, rowvar=False), -.5)
              @ power(np.cov(target, rowvar=False), .5)) + target.mean(axis=0)
    if not np.isfinite(mapped).all():
        raise ValueError('CORAL produced non-finite features')
    return mapped


def representations(x, y, z, cfg, seed, tname):
    """Every transform is fitted on x; z contributes covariates only, never labels."""
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    a = imputer.transform(x); b = imputer.transform(z)
    keep = np.flatnonzero(np.std(a, axis=0) > 0)
    if not len(keep):
        raise ValueError('No varying source metrics')
    a = a[:, keep]; b = b[:, keep]

    pp = np.empty(len(y)); ss = np.empty_like(a); folds = []
    for tr, va in StratifiedKFold(cfg['oof_folds'], shuffle=True, random_state=seed).split(a, y):
        fold_imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(x[tr])
        at = fold_imp.transform(x[tr])[:, keep]; av = fold_imp.transform(x[va])[:, keep]
        t = teacher(tname, cfg, seed); t.fit(at, y[tr])
        pp[va] = t.predict_proba(av)[:, 1]; ss[va] = shap_values(t, av)
        folds.append({'train': tr.tolist(), 'validation': va.tolist()})
    t = teacher(tname, cfg, seed); t.fit(a, y)
    pt = t.predict_proba(b)[:, 1]; st = shap_values(t, b)

    w = np.maximum(np.abs(ss).mean(axis=0), 1e-12); w = w / w.sum()
    rng = np.random.default_rng(seed)
    shuffle = w[rng.permutation(len(w))]
    k = int(np.ceil(len(w) * cfg['selection_fraction']))
    top = np.argsort(-w, kind='stable')[:k]
    random = rng.choice(len(w), k, replace=False)

    # pn_aug: one uninformative column per active feature, moments taken from the
    # source SHAP columns so the control matches ps_aug in width and in scale.
    noise_rng = np.random.default_rng(seed + NOISE_STREAM)
    loc = ss.mean(axis=0); scale = np.maximum(ss.std(axis=0), 1e-12)
    ns = noise_rng.normal(loc, scale, size=ss.shape)
    nt = noise_rng.normal(loc, scale, size=st.shape)

    ca = coral_map(a, b, cfg['coral_ridge'])

    aa = {'original': a, 'shap_sum': a * w, 'uniform_sum': a / len(w),
          'shuffled_sum': a * shuffle, 'select_shap': a[:, top], 'select_random': a[:, random],
          'p_aug': np.column_stack([a, pp]), 'ps_aug': np.column_stack([a, pp, ss]),
          'pn_aug': np.column_stack([a, pp, ns]), 'coral': ca, 'permuted_original': a}
    bb = {'original': b, 'shap_sum': b * w, 'uniform_sum': b / len(w),
          'shuffled_sum': b * shuffle, 'select_shap': b[:, top], 'select_random': b[:, random],
          'p_aug': np.column_stack([b, pt]), 'ps_aug': np.column_stack([b, pt, st]),
          'pn_aug': np.column_stack([b, pt, nt]), 'coral': b, 'permuted_original': b}
    if aa['pn_aug'].shape[1] != aa['ps_aug'].shape[1]:
        raise ValueError('Augmentation control is not dimension matched')

    corr = pd.DataFrame(a).corr(method='spearman').abs().fillna(0).to_numpy()
    scaler = RobustScaler().fit(a)
    audit = {'keep': keep.tolist(), 'weights': w.tolist(), 'shuffled_weights': shuffle.tolist(),
             'selected': top.tolist(), 'random_selected': random.tolist(), 'oof': folds,
             'source_redundancy': float(corr[np.triu_indices(len(w), 1)].mean()),
             'target_median_shift': float(np.mean(np.abs(np.median(scaler.transform(b), axis=0)))),
             'augmentation_columns': {'ps_aug': int(aa['ps_aug'].shape[1]),
                                      'pn_aug': int(aa['pn_aug'].shape[1])}}
    return aa, bb, audit


class BalancedXGB(XGBClassifier):
    """XGBoost has no class_weight, so the balanced ratio is set from the labels it
    is actually handed. This keeps one fit(X, y) interface across all four learners."""

    def fit(self, X, y, **kw):
        pos = int(np.sum(y))
        self.set_params(scale_pos_weight=(len(y) - pos) / pos if pos else 1.0)
        return super().fit(X, y, **kw)


def learner(name, choice, cfg, seed):
    c = cfg['candidates'][name][choice]
    if name == 'RandomForest':
        return RandomForestClassifier(n_estimators=cfg['downstream_trees'], max_depth=c,
                                      min_samples_leaf=2, class_weight='balanced',
                                      random_state=seed, n_jobs=1)
    if name == 'ExtraTrees':
        return ExtraTreesClassifier(n_estimators=cfg['downstream_trees'], max_depth=c,
                                    min_samples_leaf=2, class_weight='balanced',
                                    random_state=seed, n_jobs=1)
    if name == 'XGBoost':
        return BalancedXGB(n_estimators=cfg['downstream_trees'], max_depth=c, learning_rate=.1,
                           eval_metric='logloss', n_jobs=1, random_state=seed)
    return Pipeline([('log', FunctionTransformer(signed_log)), ('scale', RobustScaler()),
                     ('clf', LogisticRegression(C=c, class_weight='balanced', solver='liblinear',
                                                max_iter=20000, random_state=seed))])


def permute_within_groups(y, groups, seed):
    """Permute labels inside each project so per-project class counts are preserved."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = y[idx][rng.permutation(len(idx))]
        if out[idx].sum() != y[idx].sum():
            raise AssertionError('Permutation changed a project class count')
    return out


def fit_predict(x, y, groups, target_x, cfg, seed, tname):
    """No target_y parameter: tuning, representations and fitting are source-only."""
    fitted = cfg['models']
    learned = [a for a in cfg['arms'] if a != 'prevalence']
    permuted = permute_within_groups(y, groups, seed)
    labels_for = lambda arm: permuted if arm == 'permuted_original' else y

    cv = list(GroupKFold(cfg['outer_tuning_folds']).split(x, y, groups))
    candidate = {(m, a, c): [] for m in fitted for a in learned for c in range(2)}
    folds = []
    for tr, va in cv:
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError('Project leakage in tuning')
        aa, bb, rep = representations(x[tr], y[tr], x[va], cfg, seed, tname)
        for m, a, c in candidate:
            ytr, yva = labels_for(a)[tr], labels_for(a)[va]
            model = learner(m, c, cfg, seed); model.fit(aa[a], ytr)
            candidate[m, a, c].append(float(roc_auc_score(yva, model.predict_proba(bb[a])[:, 1])))
        folds.append({'train': tr.tolist(), 'validation': va.tolist(), 'representation': rep})

    choice = {(m, a): int(np.argmax([np.mean(candidate[m, a, c]) for c in range(2)]))
              for m in fitted for a in learned}
    aa, bb, rep = representations(x, y, target_x, cfg, seed, tname)

    predictions = {}; selected = {}
    for m in fitted:
        for a in learned:
            for c in set([choice[m, a], choice[m, 'original']]):
                model = learner(m, c, cfg, seed); model.fit(aa[a], labels_for(a))
                prob = model.predict_proba(bb[a])[:, 1]
                for regime in cfg['tuning_regimes']:
                    if (choice[m, a] if regime == 'independent' else choice[m, 'original']) == c:
                        key = '__'.join([regime, m, a])
                        predictions[key] = prob; selected[key] = c
    # Featureless reference: the source training positive rate, identical for every
    # learner and regime by construction.
    rate = float(y.mean())
    for m in fitted:
        for regime in cfg['tuning_regimes']:
            predictions['__'.join([regime, m, 'prevalence'])] = np.full(len(target_x), rate)

    audit = {'tuning': folds, 'chosen': selected, 'final_representation': rep,
             'source_positive_rate': rate,
             'permuted_source_labels': permuted.tolist(),
             'candidate_auc': {'__'.join([m, a, str(c)]): v for (m, a, c), v in candidate.items()}}
    return predictions, audit


def fingerprint(cfg):
    here = Path(__file__).parent
    paths = [Path(__file__), here / 'experiment.py', here / 'protocol_v2.json',
             here / 'protocol.json', here / 'SELECTION.json'] + \
            [DATA / (n + '.csv') for n in cfg['projects']]
    return {'files': {str(p.relative_to(ROOT)): digest(p) for p in paths},
            'packages': {n: version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'shap']},
            'config': cfg}


def task(target, seed, tname, out, cfg, fp):
    warnings.filterwarnings('error', category=ConvergenceWarning)
    start = time.time(); stem = out / f'{target}__{seed}__{tname}'
    path = lambda ext: Path(str(stem) + ext)
    if path('.done.json').exists():
        d = json.loads(path('.done.json').read_text())
        if d['fingerprint'] != fp:
            raise RuntimeError('Checkpoint fingerprint mismatch')
        if any(digest(out / n) != v for n, v in d['hashes'].items()):
            raise RuntimeError('Checkpoint corrupted')
        return f'cached {target} {seed} {tname}'
    xx = []; yy = []; gg = []; indices = {}
    for name in cfg['projects']:
        if name == target:
            continue
        x, y, _ = load(name, cfg); ix = np.arange(len(y))
        if len(ix) > cfg['source_cap_per_project']:
            ix, _ = train_test_split(ix, train_size=cfg['source_cap_per_project'],
                                     stratify=y, random_state=seed)
        xx.append(x[ix]); yy.append(y[ix]); gg.extend([name] * len(ix)); indices[name] = ix.tolist()
    x = np.concatenate(xx); y = np.concatenate(yy); g = np.asarray(gg)
    xt, yt, it = load(target, cfg)
    with threadpool_limits(limits=1):
        pred, audit = fit_predict(x, y, g, xt, cfg, seed, tname)
    rows = []
    for key, prob in pred.items():
        regime, m, a = key.split('__')
        rows.append({'target': target, 'seed': seed, 'teacher': tname, 'tuning': regime,
                     'model': m, 'arm': a, **metric(yt, prob)})
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
    ap.add_argument('--cap', type=int, default=None, help='source rows per project')
    ap.add_argument('--seeds', type=int, nargs='*', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg = config(cap=args.cap, seeds=args.seeds)
    out = Path(args.out) if args.out else OUT
    if args.smoke:
        cfg.update(teacher_trees=4, downstream_trees=4, seeds=[101])
        out = OUT.parent / 'smoke'
    out.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(cfg); mp = out / 'manifest.json'
    if mp.exists() and json.loads(mp.read_text())['fingerprint'] != fp:
        raise RuntimeError('Use a new output directory after execution changes')
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
