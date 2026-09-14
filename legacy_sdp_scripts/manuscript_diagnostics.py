"""Regenerate manuscript data diagnostics without fitting predictive models."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ECLIPSE = ['eclipse', 'mylyn', 'equinox', 'lucene', 'pde']
APACHE = ['ant-1.7', 'camel-1.6', 'xalan-2.6', 'poi-3.0', 'velocity-1.6']
RAW = ['numberOfBugsFoundUntil:', 'numberOfNonTrivialBugsFoundUntil:',
       'numberOfMajorBugsFoundUntil:', 'numberOfCriticalBugsFoundUntil:',
       'numberOfHighPriorityBugsFoundUntil:']
FEATURES = ['NBFU', 'NNTBFU', 'NMBFU', 'NCBFU', 'NHPBFU']

def load(name):
    if name in ECLIPSE:
        d = pd.read_csv(ROOT / 'data' / f'{name}-bug-metrics.csv', sep=';')
        d.columns = d.columns.str.strip()
        x = d[RAW].astype(float).copy()
        x.columns = FEATURES
        return x, (d.bugs > 0).astype(int)
    d = pd.read_csv(ROOT / 'data_promise' / f'{name}.csv')
    return d.drop(columns=['name', 'name.1', 'version', 'bug'], errors='ignore').astype(float), (d.bug > 0).astype(int)

def iqr_keep(x, multiplier=1.5):
    q1, q3 = x.quantile(.25), x.quantile(.75)
    spread = q3 - q1
    return ((x >= q1 - multiplier * spread) & (x <= q3 + multiplier * spread)).all(axis=1)

def robust_scale(x):
    spread = x.quantile(.75) - x.quantile(.25)
    # Matches RobustScaler's handling of numerically constant scales.
    scale = spread.mask(spread < 10 * np.finfo(float).eps, 1.)
    return (x - x.median()) / scale

def generate(out):
    out.mkdir(parents=True, exist_ok=True)
    summary, removal, scaler = [], [], []
    for name in ECLIPSE + APACHE + ['jedit-4.3', 'log4j-1.2']:
        x, y = load(name)
        counts = y.value_counts().reindex([0, 1], fill_value=0)
        summary.append(dict(project=name, ecosystem='Eclipse' if name in ECLIPSE else 'Apache',
            instances=len(y), buggy=int(counts[1]), clean=int(counts[0]),
            majority_minority_ratio=counts.max()/counts.min(), clean_buggy_ratio=counts[0]/counts[1],
            raw_features=x.shape[1], active_features=int((x.std() > 0).sum()),
            included=bool(counts.min() >= 30)))
        if name in APACHE:
            keep = iqr_keep(x)
            removal.append(dict(project=name, rows=len(x), rows_kept=int(keep.sum()),
                rows_kept_pct=100*keep.mean(), buggy=int(y.sum()),
                defective_deleted=int(y[~keep].sum()), defective_deleted_pct=100*y[~keep].sum()/y.sum()))
        if name in ECLIPSE + APACHE:
            a, b = robust_scale(x), robust_scale(.2*x)
            for col in x:
                scaler.append(dict(project=name, feature=col, multiplier=.2,
                    iqr=x[col].quantile(.75)-x[col].quantile(.25),
                    max_abs_difference=float((a[col]-b[col]).abs().max())))
    pd.DataFrame(summary).to_csv(out/'dataset_summary.csv', index=False)
    pd.DataFrame(removal).to_csv(out/'iqr_removal.csv', index=False)
    pd.DataFrame(scaler).to_csv(out/'scaler_diagnostics.csv', index=False)
    print(pd.DataFrame(removal).to_string(index=False))
    print('Diagnostic tables written to', out)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs'/'audit')
    generate(p.parse_args().output_dir)
