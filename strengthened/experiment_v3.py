"""External benchmark, v3. Addresses six technical findings against v2.

python -m strengthened.experiment_v3 --workers 10

What changed and why:

  * A scale-sensitive fifth learner (RBF SVC). A positive per-feature rescaling
    cannot move an axis-aligned split, so RandomForest, ExtraTrees and XGBoost
    cannot respond to the weighting arms at all. v2's weighting aggregate mixed
    one responsive learner with three structural zeros.
  * pr_aug: the SHAP matrix with whole rows permuted. Preserves every column
    marginal and every inter-column correlation, breaking only the row
    correspondence. pn_aug matches width and marginal scale but discards joint
    structure, so on its own it cannot isolate row-level attribution content.
  * Multiple control draws for the permutation/noise arms, evaluated on a fixed
    training sample.
  * teacher_cv='grouped' holds whole projects out of the teacher's own
    cross-fitting, as a predeclared sensitivity against the default.
  * Independent SeedSequence streams per role, with control roles offset
    separately so control randomness can be varied with the training sample held
    fixed.

v1 and v2 outputs and fingerprints are untouched; v3 writes its own directory.
"""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import argparse
import concurrent.futures
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from .experiment import ROOT, DATA, digest, dump, load, metric, shap_values, signed_log
from .experiment_v2 import coral_map, BalancedXGB

OUT = ROOT / 'outputs/revised/strengthening_v3/full'
ROLES = ['sampling', 'teacher', 'downstream', 'selection', 'noise', 'rowperm', 'labelperm']
CONTROL_ROLES = {'selection', 'noise', 'rowperm', 'labelperm'}


def config(**over):
    here = Path(__file__).parent
    cfg = json.loads((here / 'protocol_v3.json').read_text())
    cfg['features'] = json.loads((here / 'protocol.json').read_text())['features']
    cfg.update({k: v for k, v in over.items() if v is not None})
    return cfg


def streams(seed, cfg):
    """One independent stream per role, so repeat variation can be attributed.

    Control roles are spawned from a separately offset seed, which lets a
    sensitivity run change only the control draws while the training sample,
    teacher and downstream fits stay bit-identical.
    """
    offset = cfg.get('control_seed_offset', 0)
    main = dict(zip(ROLES, np.random.SeedSequence(seed).spawn(len(ROLES))))
    ctrl = dict(zip(ROLES, np.random.SeedSequence(seed + offset).spawn(len(ROLES))))
    chosen = {r: (ctrl[r] if r in CONTROL_ROLES else main[r]) for r in ROLES}
    return ({r: np.random.default_rng(s) for r, s in chosen.items()},
            {r: int(s.entropy) if hasattr(s, 'entropy') else None for r, s in chosen.items()},
            {r: int(np.random.default_rng(s).integers(0, 2 ** 31 - 1)) for r, s in chosen.items()})


def teacher(name, cfg, seed):
    cls = {'RandomForest': RandomForestClassifier, 'ExtraTrees': ExtraTreesClassifier}[name]
    return cls(n_estimators=cfg['teacher_trees'], max_depth=cfg['teacher_depth'],
               min_samples_leaf=2, class_weight='balanced', random_state=seed, n_jobs=1)


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
    if name == 'SVCRBF':
        # SVC(probability=True) is deprecated from sklearn 1.9; calibrate explicitly.
        svc = SVC(C=c, kernel='rbf', gamma='scale', class_weight='balanced',
                  cache_size=500, random_state=seed)
        return Pipeline([('log', FunctionTransformer(signed_log)), ('scale', RobustScaler()),
                         ('clf', CalibratedClassifierCV(svc, cv=cfg['svc_calibration_folds'],
                                                        ensemble=False))])
    return Pipeline([('log', FunctionTransformer(signed_log)), ('scale', RobustScaler()),
                     ('clf', LogisticRegression(C=c, class_weight='balanced', solver='liblinear',
                                                max_iter=20000, random_state=seed))])


def permute_within_groups(y, groups, rng):
    out = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = y[idx][rng.permutation(len(idx))]
        if out[idx].sum() != y[idx].sum():
            raise AssertionError('permutation changed a project class count')
    return out


def representations(x, y, z, cfg, seed, tname, groups=None):
    """Fitted on x only. z contributes covariates, never labels."""
    rng, entropy, ints = streams(seed, cfg)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    a = imputer.transform(x); b = imputer.transform(z)
    keep = np.flatnonzero(np.std(a, axis=0) > 0)
    if not len(keep):
        raise ValueError('no varying source metrics')
    a = a[:, keep]; b = b[:, keep]

    if cfg['teacher_cv'] == 'grouped':
        if groups is None:
            raise ValueError('grouped teacher cross-fitting needs source project groups')
        n_folds = min(cfg['oof_folds'], len(np.unique(groups)))
        splitter = GroupKFold(n_folds).split(a, y, groups)
    else:
        splitter = StratifiedKFold(cfg['oof_folds'], shuffle=True,
                                   random_state=ints['teacher']).split(a, y)

    pp = np.empty(len(y)); ss = np.empty_like(a); folds = []
    for tr, va in splitter:
        if cfg['teacher_cv'] == 'grouped' and set(groups[tr]) & set(groups[va]):
            raise AssertionError('project appears on both sides of a teacher fold')
        fold_imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(x[tr])
        at = fold_imp.transform(x[tr])[:, keep]; av = fold_imp.transform(x[va])[:, keep]
        t = teacher(tname, cfg, ints['teacher']); t.fit(at, y[tr])
        pp[va] = t.predict_proba(av)[:, 1]; ss[va] = shap_values(t, av)
        folds.append({'train': tr.tolist(), 'validation': va.tolist()})
    t = teacher(tname, cfg, ints['teacher']); t.fit(a, y)
    pt = t.predict_proba(b)[:, 1]; st = shap_values(t, b)

    w = np.maximum(np.abs(ss).mean(axis=0), 1e-12); w = w / w.sum()
    sel = rng['selection']
    shuffle = w[sel.permutation(len(w))]
    k = int(np.ceil(len(w) * cfg['selection_fraction']))
    top = np.argsort(-w, kind='stable')[:k]
    random = sel.choice(len(w), k, replace=False)

    aa = {'original': a, 'shap_sum': a * w, 'uniform_sum': a / len(w), 'shuffled_sum': a * shuffle,
          'select_shap': a[:, top], 'select_random': a[:, random],
          'p_aug': np.column_stack([a, pp]), 'ps_aug': np.column_stack([a, pp, ss]),
          'coral': coral_map(a, b, cfg['coral_ridge']), 'permuted_original': a}
    bb = {'original': b, 'shap_sum': b * w, 'uniform_sum': b / len(w), 'shuffled_sum': b * shuffle,
          'select_shap': b[:, top], 'select_random': b[:, random],
          'p_aug': np.column_stack([b, pt]), 'ps_aug': np.column_stack([b, pt, st]),
          'coral': b, 'permuted_original': b}

    # multi-draw controls: independent noise, and whole-row permutation of the
    # attribution matrix which keeps every column marginal and correlation intact
    loc, scale = ss.mean(axis=0), np.maximum(ss.std(axis=0), 1e-12)
    draws = {}
    for d in range(cfg['control_draws']):
        ns = rng['noise'].normal(loc, scale, size=ss.shape)
        nt = rng['noise'].normal(loc, scale, size=st.shape)
        ps = rng['rowperm'].permutation(len(ss)); ptg = rng['rowperm'].permutation(len(st))
        draws[('pn_aug', d)] = (np.column_stack([a, pp, ns]), np.column_stack([b, pt, nt]))
        draws[('pr_aug', d)] = (np.column_stack([a, pp, ss[ps]]), np.column_stack([b, pt, st[ptg]]))
    for (arm, d), (sa, sb) in draws.items():
        if d == 0:
            aa[arm], bb[arm] = sa, sb
    for arm in ['pn_aug', 'pr_aug']:
        if aa[arm].shape[1] != aa['ps_aug'].shape[1]:
            raise ValueError(f'{arm} is not dimension matched to ps_aug')

    corr = pd.DataFrame(a).corr(method='spearman').abs().fillna(0).to_numpy()
    audit = {'keep': keep.tolist(), 'weights': w.tolist(), 'shuffled_weights': shuffle.tolist(),
             'selected': top.tolist(), 'random_selected': random.tolist(), 'oof': folds,
             'teacher_cv': cfg['teacher_cv'],
             'source_redundancy': float(corr[np.triu_indices(len(w), 1)].mean()),
             'target_median_shift': float(np.mean(np.abs(
                 np.median(RobustScaler().fit(a).transform(b), axis=0)))),
             'augmentation_columns': {k2: int(aa[k2].shape[1])
                                      for k2 in ['ps_aug', 'pn_aug', 'pr_aug']},
             'stream_entropy': entropy, 'stream_seeds': ints}
    return aa, bb, audit, draws


def fit_predict(x, y, groups, target_x, cfg, seed, tname):
    """No target response parameter. Verified behaviourally in the test suite."""
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
        aa, bb, rep, _ = representations(x[tr], y[tr], x[va], cfg, seed, tname, groups[tr])
        for m, a, c in candidate:
            ytr, yva = labels_for(a)[tr], labels_for(a)[va]
            model = learner(m, c, cfg, ints['downstream']); model.fit(aa[a], ytr)
            candidate[m, a, c].append(float(roc_auc_score(yva, model.predict_proba(bb[a])[:, 1])))
        folds.append({'train': tr.tolist(), 'validation': va.tolist(), 'representation': rep})

    choice = {(m, a): int(np.argmax([np.mean(candidate[m, a, c]) for c in range(2)]))
              for m in fitted for a in learned}
    aa, bb, rep, draws = representations(x, y, target_x, cfg, seed, tname, groups)

    predictions, selected = {}, {}
    for m in fitted:
        for a in learned:
            variants = {0: (aa[a], bb[a])}
            if a in cfg['multi_draw_arms']:
                variants = {d: draws[(a, d)] for d in range(cfg['control_draws'])}
            for c in set([choice[m, a], choice[m, 'original']]):
                for d, (sa, sb) in variants.items():
                    model = learner(m, c, cfg, ints['downstream'])
                    model.fit(sa, labels_for(a)); prob = model.predict_proba(sb)[:, 1]
                    for regime in cfg['tuning_regimes']:
                        want = choice[m, a] if regime == 'independent' else choice[m, 'original']
                        if want == c:
                            predictions['__'.join([regime, m, a, str(d)])] = prob
                            selected['__'.join([regime, m, a])] = c
    rate = float(y.mean())
    for m in fitted:
        for regime in cfg['tuning_regimes']:
            predictions['__'.join([regime, m, 'prevalence', '0'])] = np.full(len(target_x), rate)

    audit = {'tuning': folds, 'chosen': selected, 'final_representation': rep,
             'source_positive_rate': rate, 'permuted_source_labels': permuted.tolist(),
             'candidate_auc': {'__'.join([m, a, str(c)]): v for (m, a, c), v in candidate.items()}}
    return predictions, audit


def fingerprint(cfg):
    here = Path(__file__).parent
    paths = [Path(__file__), here / 'experiment.py', here / 'experiment_v2.py',
             here / 'protocol_v3.json', here / 'protocol.json', here / 'SELECTION.json'] + \
            [DATA / (n + '.csv') for n in cfg['projects']]
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
    ap.add_argument('--seeds', type=int, nargs='*', default=None)
    ap.add_argument('--teacher-cv', choices=['stratified', 'grouped'], default=None)
    ap.add_argument('--trees', type=int, default=None)
    ap.add_argument('--control-offset', type=int, default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    over = {'seeds': args.seeds, 'teacher_cv': args.teacher_cv,
            'control_seed_offset': args.control_offset}
    if args.trees:
        over.update(downstream_trees=args.trees, teacher_trees=args.trees)
    cfg = config(**over)
    out = Path(args.out) if args.out else OUT
    if args.smoke:
        cfg.update(teacher_trees=4, downstream_trees=4, seeds=[101], control_draws=2)
        out = OUT.parent / 'smoke'
    out.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(cfg); mp = out / 'manifest.json'
    if mp.exists() and json.loads(mp.read_text())['fingerprint'] != fp:
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
