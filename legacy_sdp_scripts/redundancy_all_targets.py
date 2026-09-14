"""
Task 4: Redundancy Across All Targets
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import pipeline as P
import cross_ecosystem as XE

warnings.filterwarnings("ignore")

def run_task_4():
    print("=== Task 4: Redundancy Across All Targets ===")
    
    apache_w = pd.read_csv(P.OUT_DIR / "xe_shap_weights.csv")
    apache_w_map = dict(apache_w[["Feature", "Weight"]].values)
    
    eclipse_w = pd.read_csv(P.OUT_DIR / "shap_weights.csv")
    eclipse_w_map = dict(eclipse_w[["Feature", "Weight"]].values)

    targets = [
        ("apache", "xalan-2.6", XE.load_promise, apache_w_map),
        ("apache", "poi-3.0", XE.load_promise, apache_w_map),
        ("apache", "velocity-1.6", XE.load_promise, apache_w_map),
        ("eclipse", "equinox", P.load_xy, eclipse_w_map),
        ("eclipse", "lucene", P.load_xy, eclipse_w_map),
        ("eclipse", "pde", P.load_xy, eclipse_w_map)
    ]
    
    rows = []
    
    for eco, dataset, load_fn, w_map in targets:
        print(f"Dataset: {dataset}")
        X, y, cols = load_fn(dataset)
        use = [c for c in cols if c in w_map]
        X = X[use]
        F = len(use)
        k = int(np.ceil(F / 2.0))
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        imp_array = np.array([w_map[c] for c in use])
        top_shap_indices = np.argsort(-imp_array)[:k]
        top_shap_names = [use[i] for i in top_shap_indices]
        
        def calc_mean_corr(feature_subset):
            if len(feature_subset) < 2:
                return 1.0
            corrs = []
            for i in range(len(feature_subset)):
                for j in range(i+1, len(feature_subset)):
                    corrs.append(corr_matrix.loc[feature_subset[i], feature_subset[j]])
            return np.mean(corrs)
            
        shap_mean_corr = calc_mean_corr(top_shap_names)
        
        rnd_corrs = []
        rng = np.random.RandomState(P.SEED)
        for _ in range(500):
            rnd_subset = list(rng.choice(use, k, replace=False))
            rnd_corrs.append(calc_mean_corr(rnd_subset))
            
        random_mean = np.mean(rnd_corrs)
        random_std = np.std(rnd_corrs, ddof=1)
        
        percentile = np.mean(np.array(rnd_corrs) < shap_mean_corr) * 100
        
        rows.append({
            "dataset": dataset,
            "n_features": F,
            "k": k,
            "shap_set_mean_abs_corr": shap_mean_corr,
            "random_mean": random_mean,
            "random_std": random_std,
            "percentile": percentile
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(P.OUT_DIR / "redundancy_all_targets.csv", index=False)
    print("Done Task 4.")

if __name__ == "__main__":
    run_task_4()
