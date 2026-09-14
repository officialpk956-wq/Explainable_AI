"""Capacity dose-response for the augmentation decomposition.

python -m strengthened.sweep_capacity --workers 10

v3 compared two capacity points at three seeds and the row-correspondence contrast
reversed. Two points cannot distinguish a genuine reversal from noise, and cannot
say whether the augmentation penalty is a capacity-limited fitting cost. This
sweeps four capacity levels on a focused arm subset, with more control draws so
the contrast is estimated precisely enough to interpret.

It imports experiment_v3 without modifying it, so the main v3 checkpoints keep
their fingerprints.
"""
import argparse
import concurrent.futures
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .experiment import dump
from .experiment_v3 import OUT as V3, config, fingerprint, task

SWEEP = V3.parent / 'capacity_sweep'
# Three points establish monotonicity. A 256-tree level was specified initially and
# dropped on measured runtime before any sweep result was read; it would have taken
# longer than the other three combined.
LEVELS = [32, 64, 128]
# only the arms that enter the augmentation decomposition, so four capacity
# levels cost about what one full-grid run would
ARMS = ['original', 'p_aug', 'ps_aug', 'pn_aug', 'pr_aug']
SEEDS = [101, 102, 103]
DRAWS = 5


def level_config(trees):
    return config(seeds=SEEDS, arms=ARMS, control_draws=DRAWS,
                  teacher_trees=trees, downstream_trees=trees)


def run(workers):
    SWEEP.mkdir(parents=True, exist_ok=True)
    for trees in LEVELS:
        cfg = level_config(trees)
        out = SWEEP / f'trees{trees}'
        out.mkdir(parents=True, exist_ok=True)
        fp = fingerprint(cfg)
        mp = out / 'manifest.json'
        if mp.exists() and json.loads(mp.read_text()).get('status') == 'complete':
            print(f'trees={trees}: cached', flush=True)
            continue
        dump(mp, {'status': 'running', 'fingerprint': fp,
                  'started_utc': datetime.now(timezone.utc).isoformat(), 'smoke': False})
        jobs = [(t, s, n, out, cfg, fp) for t in cfg['projects']
                for s in cfg['seeds'] for n in cfg['teachers']]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            done = 0
            for f in concurrent.futures.as_completed([ex.submit(task, *j) for j in jobs]):
                f.result(); done += 1
        scores = pd.concat([pd.read_csv(Path(str(out / f'{t}__{s}__{n}') + '.csv'))
                            for t, s, n, *_ in jobs], ignore_index=True)
        scores.to_csv(out / 'scores.csv', index=False)
        dump(mp, {'status': 'complete', 'fingerprint': fp,
                  'completed_utc': datetime.now(timezone.utc).isoformat(),
                  'smoke': False, 'rows': len(scores)})
        print(f'trees={trees}: complete, {len(scores)} rows', flush=True)


def collect():
    PK = ['target', 'seed', 'teacher', 'tuning', 'model']
    METRICS = ['auc', 'average_precision', 'f1_macro']
    pairs = [('ps_aug', 'p_aug'), ('pn_aug', 'p_aug'), ('pr_aug', 'p_aug'),
             ('ps_aug', 'pr_aug'), ('ps_aug', 'pn_aug')]
    rows = []
    for trees in LEVELS:
        p = SWEEP / f'trees{trees}' / 'scores.csv'
        if not p.exists():
            continue
        s = pd.read_csv(p)
        s = s[s.tuning == 'independent'].groupby(PK + ['arm'], as_index=False)[METRICS].mean()
        for arm, control in pairs:
            m = s[s.arm == arm].merge(s[s.arm == control], on=PK, suffixes=('_a', '_b'),
                                      validate='one_to_one')
            per_target = (m.auc_a - m.auc_b).groupby(m.target).mean()
            rows.append({'trees': trees, 'arm': arm, 'control': control,
                         'mean_difference': float(per_target.mean()),
                         'target_min': float(per_target.min()),
                         'target_max': float(per_target.max()),
                         'targets_negative': int((per_target < 0).sum()),
                         'targets': len(per_target)})
    df = pd.DataFrame(rows)
    df.to_csv(SWEEP / 'capacity_sweep.csv', index=False)

    verdict = {}
    for arm, control in pairs:
        d = df[(df.arm == arm) & (df.control == control)].sort_values('trees')
        if len(d) < 2:
            continue
        vals = d.mean_difference.tolist()
        rising = all(b >= a for a, b in zip(vals, vals[1:]))
        falling = all(b <= a for a, b in zip(vals, vals[1:]))
        verdict[f'{arm}_vs_{control}'] = {
            'levels': d.trees.tolist(), 'values': vals,
            'monotone': 'increasing' if rising else ('decreasing' if falling else 'no'),
            'sign_changes': int(sum((a > 0) != (b > 0) for a, b in zip(vals, vals[1:]))),
            'range': max(vals) - min(vals)}
    dump(SWEEP / 'capacity_verdict.json', verdict)
    print(df.to_string(index=False))
    print()
    print(json.dumps(verdict, indent=2))
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--collect-only', action='store_true')
    a = ap.parse_args()
    if not a.collect_only:
        run(a.workers)
    collect()
