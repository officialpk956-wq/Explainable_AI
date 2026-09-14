"""Correlation/redundancy diagnostic on the 54 static metrics, cross-referenced
against the OOF SHAP weights and select_shap membership the v3 run already
computed.

python -m strengthened.diagnostic_correlation

Pure diagnostic: no new model is fitted on any label. VIF regresses one static
metric on the other 53 static metrics -- it characterises the design matrix's
own collinearity and never touches RealBug. Everything else is read from the
already-completed v3 checkpoints under outputs/revised/strengthening_v3/full.

Jiarpakdee & Tantithamthavorn (TSE 2019) and Rajbahadur et al. (TSE 2022) both
report that correlated/interacting static metrics distort attribution rankings:
a top-ranked metric shares credit with its correlated siblings, so which
specific member ends up ranked top is partly arbitrary. This checks whether
that mechanism is present in the 54-metric Rnalytica benchmark, and whether it
is a plausible explanation for why shap_sum sits near zero and select_shap does
not beat select_random.

A literal Friedman H-statistic needs partial-dependence curves from a fitted
model, which would be a new model fit and is deliberately not done here. In its
place, Part B uses the mean-absolute-SHAP weight vectors the v3 run already
computed and stored per checkpoint (140 checkpoints: 7 targets x 10 seeds x 2
teachers) as a fit-free stand-in: if a correlated cluster's *total* SHAP weight
is stable across checkpoints while *which member* carries that weight is not,
that is direct evidence of the dilution mechanism the two papers describe.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

from .experiment import ROOT, load
from .experiment_v3 import OUT, config, fingerprint

DEST = OUT / 'correlation_diagnostic'
CLUSTER_THRESHOLDS = [0.7, 0.8]


def pooled_features(cfg):
    """All 7 projects' raw metrics, imputed exactly as representations() would:
    fit on the pooled matrix, since every project plays source in 6 of 7
    rotations. Returns the imputed matrix and which columns had any variance."""
    xs = [load(p, cfg)[0] for p in cfg['projects']]
    x = np.concatenate(xs)
    x = SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(x)
    keep = np.std(x, axis=0) > 0
    return x, keep


def clusters_at(corr, threshold):
    adjacency = (np.abs(corr) >= threshold).astype(int)
    np.fill_diagonal(adjacency, 0)
    n, labels = connected_components(adjacency, directed=False)
    return labels


def vif(x):
    """Variance inflation factor per column: regress column i on all others.
    No statsmodels dependency; a closed-form OLS R^2 is all VIF needs."""
    out = np.empty(x.shape[1])
    for i in range(x.shape[1]):
        others = np.delete(x, i, axis=1)
        r2 = LinearRegression().fit(others, x[:, i]).score(others, x[:, i])
        out[i] = np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
    return out


def load_checkpoint_weights(cfg):
    """Every completed v3 checkpoint's final-representation SHAP weight vector
    and select_shap membership, verified against the run's own fingerprint so a
    stale or foreign checkpoint cannot silently enter the diagnostic."""
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['fingerprint'] != fingerprint(cfg):
        raise RuntimeError('v3 full run is not complete or no longer matches the current code')
    features = cfg['features']
    rows = []
    for target in cfg['projects']:
        for seed in cfg['seeds']:
            for teacher in cfg['teachers']:
                stem = OUT / f'{target}__{seed}__{teacher}'
                audit = json.loads(Path(str(stem) + '.audit.json').read_text())
                rep = audit['final_representation']
                if rep['keep'] != list(range(len(features))):
                    raise RuntimeError(f'{stem.name}: keep is not the identity map; '
                                       'weight-to-feature-name mapping needs remapping, '
                                       'not the direct index this script assumes')
                w = np.asarray(rep['weights'])
                sel = set(rep['selected'])
                checkpoint = f'{target}__{seed}__{teacher}'
                for i, feat in enumerate(features):
                    rows.append({'checkpoint': checkpoint, 'target': target, 'seed': seed,
                                'teacher': teacher, 'feature': feat, 'weight': w[i],
                                'selected': i in sel})
    return pd.DataFrame(rows)


def cluster_dilution(long, cluster_of, features, total_features, draw_size):
    """For each correlated cluster: is total cluster credit far more stable
    across checkpoints than which member carries it? That instability is what
    would make select_shap's specific pick close to arbitrary.

    dominant_member_selection_share alone is a poor summary: with a
    27-of-54 draw and a 35-member cluster, some member is selected in nearly
    every checkpoint almost by construction, even if selection rotates freely
    among all 35. The full per-member share distribution (min/median/max) is
    what actually distinguishes "one member always wins" from "credit rotates
    unpredictably across most of the cluster" -- so both are reported.
    """
    long = long.copy()
    long['cluster'] = long.feature.map(cluster_of)
    pivot = long.pivot_table(index='checkpoint', columns='feature', values='weight')
    n_checkpoints = long.checkpoint.nunique()

    rows = []
    for cid, members in pd.Series(cluster_of).groupby(cluster_of):
        members = list(members.index)
        if len(members) < 2:
            continue
        member_w = pivot[members]
        total = member_w.sum(axis=1)
        total_cv = float(total.std() / max(total.mean(), 1e-12))
        member_cv = (member_w.std() / member_w.mean().clip(lower=1e-12))
        dilution_ratio = float(member_cv.mean() / max(total_cv, 1e-12))

        sel = long[long.cluster == cid]
        per_member_share = (sel.groupby('feature').selected.mean()
                            .reindex(members, fill_value=0.0))
        base_rate = draw_size / total_features
        avg_picks_per_draw = float(sel.selected.sum() / n_checkpoints)
        rows.append({
            'cluster': cid, 'size': len(members), 'members': ';'.join(members),
            'total_weight_cv': total_cv, 'mean_member_weight_cv': float(member_cv.mean()),
            'max_member_weight_cv': float(member_cv.max()),
            'dilution_ratio': dilution_ratio,
            'min_member_selection_share': float(per_member_share.min()),
            'median_member_selection_share': float(per_member_share.median()),
            'max_member_selection_share': float(per_member_share.max()),
            'iqr_member_selection_share': float(per_member_share.quantile(.75)
                                                - per_member_share.quantile(.25)),
            'avg_cluster_members_per_draw': avg_picks_per_draw,
            'expected_members_per_draw_if_random': len(members) * base_rate,
            'over_representation_vs_random': (avg_picks_per_draw / (len(members) * base_rate)
                                              if len(members) * base_rate > 0 else float('nan')),
        })
    return pd.DataFrame(rows).sort_values('size', ascending=False)


def run():
    cfg = config()
    DEST.mkdir(parents=True, exist_ok=True)
    features = cfg['features']

    # ---- Part A: correlation / redundancy structure, no labels touched -----
    x, keep = pooled_features(cfg)
    if not keep.all():
        raise RuntimeError('a static metric is constant across the pooled 7 projects; '
                           'the fixed feature-index assumption below needs revisiting')
    corr = pd.DataFrame(x, columns=features).corr(method='spearman').to_numpy()
    pd.DataFrame(corr, index=features, columns=features).to_csv(DEST / 'correlation_matrix.csv')

    cluster_cols = {}
    for t in CLUSTER_THRESHOLDS:
        labels = clusters_at(corr, t)
        cluster_cols[f'cluster_{t}'] = labels
    clusters_df = pd.DataFrame({'feature': features, **cluster_cols})
    for t in CLUSTER_THRESHOLDS:
        sizes = clusters_df[f'cluster_{t}'].value_counts()
        clusters_df[f'cluster_{t}_size'] = clusters_df[f'cluster_{t}'].map(sizes)
    clusters_df.to_csv(DEST / 'clusters.csv', index=False)

    vif_scores = vif(x)
    vif_df = pd.DataFrame({'feature': features, 'vif': vif_scores}).sort_values(
        'vif', ascending=False)
    vif_df.to_csv(DEST / 'vif.csv', index=False)

    # ---- Part B: does the already-computed SHAP weight track the clusters? -
    long = load_checkpoint_weights(cfg)
    long.to_csv(DEST / 'checkpoint_weights_long.csv', index=False)

    summary = long.groupby('feature').agg(
        mean_weight=('weight', 'mean'), std_weight=('weight', 'std'),
        selection_frequency=('selected', 'mean')).reset_index()
    summary['cv_weight'] = summary.std_weight / summary.mean_weight.clip(lower=1e-12)
    summary = summary.merge(vif_df, on='feature').merge(
        clusters_df[['feature', 'cluster_0.7', 'cluster_0.7_size']], on='feature')
    summary.sort_values('mean_weight', ascending=False).to_csv(
        DEST / 'shap_weight_summary.csv', index=False)

    draw_size = max(1, round(len(features) * cfg['selection_fraction']))
    cluster_of_07 = dict(zip(clusters_df.feature, clusters_df['cluster_0.7']))
    dilution = cluster_dilution(long, cluster_of_07, features, len(features), draw_size)
    dilution.to_csv(DEST / 'cluster_credit_dilution.csv', index=False)

    # ---- verdict ------------------------------------------------------------
    multi = dilution[dilution['size'] >= 2]
    high_vif = vif_df[np.isfinite(vif_df.vif) & (vif_df.vif > 10)]
    verdict = {
        'checkpoints_analysed': int(long.checkpoint.nunique()),
        'features': len(features),
        'clusters_at_0.7': int(clusters_df['cluster_0.7'].nunique()),
        'multi_member_clusters_at_0.7': int(len(multi)),
        'largest_cluster_at_0.7': int(clusters_df['cluster_0.7_size'].max()),
        'features_with_vif_over_10': int(len(high_vif)),
        'features_with_vif_over_5': int((np.isfinite(vif_df.vif) & (vif_df.vif > 5)).sum()),
        'median_dilution_ratio_multi_member_clusters':
            float(multi.dilution_ratio.median()) if len(multi) else None,
        'share_of_multi_member_clusters_with_dilution_ratio_over_1':
            float((multi.dilution_ratio > 1).mean()) if len(multi) else None,
        'largest_cluster_over_representation_vs_random_draw':
            float(multi.loc[multi['size'].idxmax(), 'over_representation_vs_random'])
            if len(multi) else None,
        'largest_cluster_member_selection_share_min_median_max':
            [float(multi.loc[multi['size'].idxmax(), c]) for c in
             ['min_member_selection_share', 'median_member_selection_share',
              'max_member_selection_share']] if len(multi) else None,
        'interpretation': (
            'A dilution ratio above 1 means a correlated cluster\'s total SHAP credit is '
            'more stable across checkpoints than which specific member carries it -- the '
            'mechanism Jiarpakdee & Tantithamthavorn (TSE 2019) and Rajbahadur et al. '
            '(TSE 2022) describe. Within the largest cluster, per-member selection share '
            'spans from min to max across its members (not just the maximum, which is '
            'inflated by construction when a cluster supplies most of a half-sized draw): '
            'a wide spread means select_shap does not consistently pick the same cluster '
            'member across checkpoints, so which specific feature enters the top half is '
            'close to arbitrary for that cluster -- a candidate explanation for why '
            'select_shap does not beat select_random. Over-representation above 1 means the '
            'cluster supplies more of each draw than its share of all 54 features would '
            'predict by chance.'),
        'caveat': (
            'This substitutes the already-computed OOF SHAP weight vectors for a literal '
            'Friedman H-statistic, which would require partial-dependence curves from a new '
            'model fit. No new model was fitted on any label for this diagnostic; VIF '
            'regresses static metrics on each other and never touches RealBug.'),
    }
    (DEST / 'verdict.json').write_text(json.dumps(verdict, indent=2), encoding='utf-8')

    print(f"clusters at |rho|>=0.7: {verdict['clusters_at_0.7']} "
          f"({verdict['multi_member_clusters_at_0.7']} with >=2 members, "
          f"largest {verdict['largest_cluster_at_0.7']})")
    print(f"features with VIF>10: {verdict['features_with_vif_over_10']}/{len(features)}")
    print(f"median dilution ratio (multi-member clusters): "
          f"{verdict['median_dilution_ratio_multi_member_clusters']:.2f}")
    print(f"clusters with dilution ratio > 1: "
          f"{verdict['share_of_multi_member_clusters_with_dilution_ratio_over_1']:.0%}")
    if verdict['largest_cluster_member_selection_share_min_median_max']:
        lo, med, hi = verdict['largest_cluster_member_selection_share_min_median_max']
        print(f"largest cluster per-member selection share: min {lo:.2f} / "
              f"median {med:.2f} / max {hi:.2f}")
        print(f"largest cluster over-representation vs random draw: "
              f"{verdict['largest_cluster_over_representation_vs_random_draw']:.2f}x")
    print(f'\nwrote {DEST}')
    return verdict


if __name__ == '__main__':
    run()
