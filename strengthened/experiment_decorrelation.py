"""RQ-D: does the near-zero shap_sum effect and the small select_shap effect
survive on a decorrelated 20-feature set? Implements protocol_decorrelation.json
exactly -- frozen before this file was written.

python -m strengthened.experiment_decorrelation --workers 10

Reuses experiment_v3's config/task/fingerprint machinery unchanged via a config
override, rather than reimplementing the harness: fingerprint(cfg) embeds the
whole (overridden) cfg dict, so the reduced feature list and arm set are
captured distinctly from the main v3 run's fingerprint, and task() needs no
changes since load()/representations() already index features by name and
size arrays from len(cfg['features']) rather than a hardcoded 54.
"""
import argparse
import concurrent.futures
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .experiment import ROOT, dump
from .experiment_v3 import config as v3_config, fingerprint, task

SPEC = json.loads((Path(__file__).parent / 'protocol_decorrelation.json').read_text())
OUT = ROOT / 'outputs/revised/strengthening_v3/decorrelation'


def verify_spec_reproducible():
    """The frozen reduced_features list must still be exactly what the frozen
    rule derives from the frozen input files -- if either drifted since
    protocol_decorrelation.json was written, this must fail loudly, not run
    silently against a list nobody actually re-derived."""
    from .experiment import digest
    for key in ['clusters_csv', 'vif_csv']:
        path = ROOT / SPEC['inputs'][key]
        got = digest(path)
        want = SPEC['inputs'][f'{key}_sha256']
        if got != want:
            raise RuntimeError(f'{key} no longer matches the hash frozen in '
                               f'protocol_decorrelation.json ({got} != {want})')
    clusters = pd.read_csv(ROOT / SPEC['inputs']['clusters_csv'])
    vif = pd.read_csv(ROOT / SPEC['inputs']['vif_csv'])
    m = clusters.merge(vif, on='feature')
    reps = []
    for _, g in m.groupby(SPEC['inputs']['cluster_column_used']):
        g = g.sort_values(['vif', 'feature'])
        reps.append(g.iloc[0]['feature'])
    reps = sorted(reps)
    if reps != SPEC['reduced_features']:
        raise RuntimeError('re-deriving the representative-selection rule gives a different '
                           f'feature list than the one frozen in protocol_decorrelation.json: '
                           f'{reps} != {SPEC["reduced_features"]}')


def config():
    cfg = v3_config(features=SPEC['reduced_features'], arms=SPEC['arms_to_rerun'],
                    multi_draw_arms=[], control_draws=1,
                    seeds=SPEC['grid']['seeds'], teachers=SPEC['grid']['teachers'],
                    models=SPEC['grid']['models'],
                    tuning_regimes=SPEC['grid']['tuning_regimes'],
                    source_cap_per_project=SPEC['grid']['source_cap_per_project'],
                    teacher_trees=SPEC['grid']['teacher_trees'],
                    teacher_depth=SPEC['grid']['teacher_depth'],
                    downstream_trees=SPEC['grid']['downstream_trees'],
                    outer_tuning_folds=SPEC['grid']['outer_tuning_folds'],
                    oof_folds=SPEC['grid']['oof_folds'],
                    selection_fraction=SPEC['selection_fraction'])
    if len(cfg['features']) != SPEC['reduced_feature_count']:
        raise RuntimeError('feature count does not match the frozen spec')
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    verify_spec_reproducible()
    cfg = config()
    out = OUT
    if args.smoke:
        cfg.update(teacher_trees=4, downstream_trees=4, seeds=cfg['seeds'][:1])
        out = OUT.parent / 'decorrelation_smoke'
    out.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(cfg)
    mp = out / 'manifest.json'
    if mp.exists() and json.loads(mp.read_text()).get('status') == 'complete' \
            and json.loads(mp.read_text())['fingerprint'] == fp:
        print('cached: manifest already complete with a matching fingerprint')
        return
    if mp.exists() and json.loads(mp.read_text()).get('fingerprint') != fp:
        raise RuntimeError('use a new output directory after execution changes')
    dump(mp, {'status': 'running', 'fingerprint': fp,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'smoke': args.smoke,
              'protocol': 'protocol_decorrelation.json'})
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
              'smoke': args.smoke, 'rows': len(scores), 'protocol': 'protocol_decorrelation.json'})
    print('COMPLETE', len(scores), flush=True)


if __name__ == '__main__':
    main()
