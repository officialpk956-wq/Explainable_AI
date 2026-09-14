"""RQ-N: do the small shap_sum/select_shap effects survive graded source-label
noise? Implements protocol_labelnoise.json exactly -- frozen before this file
was written.

python -m strengthened.experiment_labelnoise --workers 10

Reuses experiment_v3.fit_predict, representations() and streams() UNCHANGED --
label corruption happens before fit_predict is called, and fit_predict treats
whatever y it receives as ground truth, so the teacher and downstream both see
the corrupted labels without any change to that function. sampling/teacher/
downstream/selection randomness is drawn from experiment_v3.streams(seed, cfg)
exactly as the main v3 run does, so the only deliberate difference from that
run is the label corruption itself.

The frozen spec calls for "a new labelnoise random stream role, distinct from
the existing labelperm role." experiment_v3.ROLES is not extended to add it,
because doing so would change experiment_v3.py's own file hash and invalidate
the fingerprint of the already-completed v3 main run and its four sensitivity
runs. Instead this module derives its own independent RNG, seeded from
(seed, noise level) alone, entirely outside experiment_v3's stream bookkeeping
-- a distinct stream in substance, implemented outside that file rather than
inside its ROLES enum. This is a disclosed deviation from a literal reading of
"role", not a silent one.
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .experiment import ROOT, DATA, digest, dump, load, metric
from .experiment_v3 import config as v3_config, fingerprint as v3_fingerprint, \
    fit_predict, streams

SPEC = json.loads((Path(__file__).parent / 'protocol_labelnoise.json').read_text())
OUT = ROOT / 'outputs/revised/strengthening_v3/labelnoise'


def config():
    return v3_config(arms=SPEC['arms_to_rerun_per_noise_level'], multi_draw_arms=[],
                     control_draws=1, seeds=SPEC['grid']['seeds'],
                     teachers=SPEC['grid']['teachers'], models=SPEC['grid']['models'],
                     tuning_regimes=SPEC['grid']['tuning_regimes'],
                     source_cap_per_project=SPEC['grid']['source_cap_per_project'],
                     teacher_trees=SPEC['grid']['teacher_trees'],
                     teacher_depth=SPEC['grid']['teacher_depth'],
                     downstream_trees=SPEC['grid']['downstream_trees'],
                     outer_tuning_folds=SPEC['grid']['outer_tuning_folds'],
                     oof_folds=SPEC['grid']['oof_folds'])


def noise_rng(seed, p):
    """The 'labelnoise' stream: independent of experiment_v3's ROLES, seeded
    from (seed, noise level) alone."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), round(p * 100000)]))


def inject_noise(y, groups, p, rng):
    """Within each source project independently, flip floor(p * n_rows) labels
    chosen uniformly at random without replacement. Per protocol_labelnoise.json
    this does NOT preserve per-project class counts -- unlike
    permute_within_groups, it is a direct corruption, and the resulting
    prevalence drift is returned for audit, not corrected."""
    out = y.copy()
    flipped = np.zeros(len(y), dtype=bool)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        k = int(np.floor(p * len(idx)))
        if k == 0:
            continue
        chosen = rng.choice(idx, size=k, replace=False)
        out[chosen] = 1 - out[chosen]
        flipped[chosen] = True
    return out, flipped


def fingerprint(cfg, noise_p):
    """Same file set as v3's fingerprint plus this module and the frozen
    label-noise spec, so a change to either invalidates cached checkpoints."""
    here = Path(__file__).parent
    paths = [Path(__file__), here / 'experiment_v3.py', here / 'experiment.py',
             here / 'experiment_v2.py', here / 'protocol_labelnoise.json',
             here / 'protocol.json', here / 'SELECTION.json'] + \
            [DATA / (n + '.csv') for n in cfg['projects']]
    from importlib.metadata import version
    return {'files': {str(p.relative_to(ROOT)): digest(p) for p in paths},
            'packages': {n: version(n) for n in
                         ['numpy', 'pandas', 'scikit-learn', 'scipy', 'shap', 'xgboost']},
            'config': cfg, 'noise_level': noise_p}


def task(target, seed, tname, noise_p, out, cfg, fp):
    warnings.filterwarnings('error', category=ConvergenceWarning)
    start = time.time()
    stem = out / f'{target}__{seed}__{tname}__p{noise_p}'
    path = lambda ext: Path(str(stem) + ext)
    if path('.done.json').exists():
        d = json.loads(path('.done.json').read_text())
        if d['fingerprint'] != fp:
            raise RuntimeError('checkpoint fingerprint mismatch')
        if any(digest(out / n) != v for n, v in d['hashes'].items()):
            raise RuntimeError('checkpoint corrupted')
        return f'cached {target} {seed} {tname} p={noise_p}'

    rng, _, ints = streams(seed, cfg)   # reuse v3's sampling/teacher/downstream/selection streams
    xx, yy, gg, indices = [], [], [], {}
    for name in cfg['projects']:
        if name == target:
            continue
        x, y, _ = load(name, cfg); ix = np.arange(len(y))
        if len(ix) > cfg['source_cap_per_project']:
            ix, _ = train_test_split(ix, train_size=cfg['source_cap_per_project'],
                                     stratify=y, random_state=ints['sampling'])
        xx.append(x[ix]); yy.append(y[ix]); gg.extend([name] * len(ix)); indices[name] = ix.tolist()
    x = np.concatenate(xx); y_true = np.concatenate(yy); g = np.asarray(gg)
    y_noisy, flipped = inject_noise(y_true, g, noise_p, noise_rng(seed, noise_p))

    xt, yt, it = load(target, cfg)
    with threadpool_limits(limits=1):
        pred, audit = fit_predict(x, y_noisy, g, xt, cfg, seed, tname)

    rows = []
    for key, prob in pred.items():
        regime, m, a, d = key.split('__')
        rows.append({'target': target, 'seed': seed, 'teacher': tname, 'noise_level': noise_p,
                     'tuning': regime, 'model': m, 'arm': a, 'draw': int(d),
                     **metric(yt, prob)})
    pred['labels'] = yt; pred['target_ids'] = it
    np.savez_compressed(path('.npz'), **pred)
    pd.DataFrame(rows).to_csv(path('.csv'), index=False)
    audit.update({'target': target, 'seed': seed, 'teacher': tname, 'noise_level': noise_p,
                  'source_indices': indices, 'source_groups': g.tolist(),
                  'source_labels_true': y_true.tolist(), 'source_labels_noisy': y_noisy.tolist(),
                  'flipped_row_count': int(flipped.sum()),
                  'flipped_fraction_of_source': float(flipped.mean()),
                  'source_prevalence_true': float(y_true.mean()),
                  'source_prevalence_noisy': float(y_noisy.mean()),
                  'target_rows': len(yt), 'seconds': time.time() - start})
    dump(path('.audit.json'), audit)
    dump(path('.done.json'), {'fingerprint': fp,
                              'hashes': {path(e).name: digest(path(e))
                                         for e in ['.csv', '.npz', '.audit.json']}})
    return f'completed {target} seed={seed} teacher={tname} p={noise_p} ({time.time() - start:.1f}s)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    cfg = config()
    out = OUT
    levels = SPEC['noise_levels']
    if args.smoke:
        cfg.update(teacher_trees=4, downstream_trees=4, seeds=cfg['seeds'][:1])
        levels = levels[:1]
        out = OUT.parent / 'labelnoise_smoke'
    out.mkdir(parents=True, exist_ok=True)

    fps = {p: fingerprint(cfg, p) for p in levels}
    mp = out / 'manifest.json'
    if mp.exists():
        prior = json.loads(mp.read_text())
        # JSON keys round-trip as strings; compare both sides normalized, not by
        # raw dict identity, or a float-vs-string key mismatch looks like a change.
        prior_fps = {str(k): v for k, v in prior.get('fingerprints', {}).items()}
        fresh_fps = {str(k): v for k, v in fps.items()}
        if prior.get('status') == 'complete' and prior_fps == fresh_fps:
            print('cached: manifest already complete with matching fingerprints')
            return
        if prior.get('fingerprints') and prior_fps != fresh_fps:
            raise RuntimeError('use a new output directory after execution changes')
    dump(mp, {'status': 'running', 'fingerprints': fps,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'smoke': args.smoke,
              'protocol': 'protocol_labelnoise.json'})

    targets = cfg['projects'][:1] if args.smoke else cfg['projects']
    jobs = [(t, s, n, p, out, cfg, fps[p]) for t in targets for s in cfg['seeds']
            for n in cfg['teachers'] for p in levels]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in concurrent.futures.as_completed([ex.submit(task, *j) for j in jobs]):
            print(f.result(), flush=True)
    scores = pd.concat([pd.read_csv(Path(str(out / f'{t}__{s}__{n}__p{p}') + '.csv'))
                        for t, s, n, p, *_ in jobs], ignore_index=True)
    scores.to_csv(out / 'scores.csv', index=False)
    dump(mp, {'status': 'complete', 'fingerprints': fps,
              'completed_utc': datetime.now(timezone.utc).isoformat(),
              'smoke': args.smoke, 'rows': len(scores), 'protocol': 'protocol_labelnoise.json'})
    print('COMPLETE', len(scores), flush=True)


if __name__ == '__main__':
    main()
