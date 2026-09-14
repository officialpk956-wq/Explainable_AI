"""
Phase 3: Genuine CPDP Harness.
Zero-shot and few-shot cross-project defect prediction transferring models from Eclipse to Apache.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pipeline as P
import powered_phase_h as PH
import cross_ecosystem as XE

warnings.filterwarnings("ignore")
OUT_DIR = P.OUT_DIR

def run_cpdp():
    print("=== Phase 3: Genuine CPDP Harness ===")
    
    apache_sources = XE.XE_SOURCES
    apache_targets = XE.XE_TARGETS
    
    # 1. Leave-One-Source-Out (LOSO) Weight Stability
    print("Computing LOSO weight stability on Apache pool...")
    loso_weights = []
    
    pool_data = {name: XE.load_promise(name) for name in apache_sources}
    
    # We need the common features across all Apache sets
    common_cols = set(pool_data[apache_sources[0]][2])
    for name in apache_sources[1:]:
        common_cols = common_cols.intersection(set(pool_data[name][2]))
    common_cols = list(common_cols)
    
    for left_out in apache_sources:
        # Pool others
        X_pool = []
        y_pool = []
        for name in apache_sources:
            if name != left_out:
                X, y, _ = pool_data[name]
                X_pool.append(X[common_cols])
                y_pool.append(y)
                
        if len(X_pool) == 0:
            continue
            
        X_train = pd.concat(X_pool, axis=0).reset_index(drop=True)
        y_train = pd.concat(y_pool, axis=0).reset_index(drop=True)
        
        tuned = P.tune_hyperparams(X_train, y_train, f"pool_no_{left_out}")
        
        # We need a stable model to extract weights. Using RandomForest.
        model_name = "RandomForest"
        w_abs = P.shap_weights_out_of_fold(X_train, y_train, model_name, tuned)
        w = P.normalise(w_abs)
        loso_weights.append(w)
        
    rhos = []
    for i in range(len(loso_weights)):
        for j in range(i + 1, len(loso_weights)):
            r = spearmanr(loso_weights[i], loso_weights[j])
            rhos.append(r.statistic)
            
    median_rho = np.median(rhos)
    print(f"LOSO Median Spearman rho: {median_rho:.3f}")
    
    if median_rho < 0.5:
        print("INTERPRETATION: The importance vector is itself source-unstable.")
        
    pd.DataFrame([{"median_rho": median_rho, "unstable": median_rho < 0.5}]).to_csv(OUT_DIR / "loso_weight_stability.csv", index=False)
    
    # 2. Transfer to Apache targets
    print("Transferring to Apache targets...")
    
    # Train full Apache pool model
    X_pool = []
    y_pool = []
    for name in apache_sources:
        X, y, _ = pool_data[name]
        X_pool.append(X[common_cols])
        y_pool.append(y)
        
    X_train_full = pd.concat(X_pool, axis=0).reset_index(drop=True)
    y_train_full = pd.concat(y_pool, axis=0).reset_index(drop=True)
    
    tuned_full = P.tune_hyperparams(X_train_full, y_train_full, "full_apache_pool")
    model_name = "RandomForest"
    w_abs_full = P.shap_weights_out_of_fold(X_train_full, y_train_full, model_name, tuned_full)
    w_full = P.normalise(w_abs_full)
    w_map = dict(zip(common_cols, w_full))
    
    clf = P.build_classifier(model_name, tuned_full, y_train_full.values)
    pipe = P.make_pipeline_for(model_name, clf)
    pipe.fit(X_train_full.values, y_train_full.values)
    
    rows = []
    
    for target in apache_targets:
        X_t, y_t, t_cols = XE.load_promise(target)
        
        # intersect features
        use = [c for c in common_cols if c in t_cols]
        if len(use) == 0:
            continue
            
        X_t = X_t[use]
        w_target = np.array([w_map[c] for c in use])
        w_target = w_target / w_target.sum()
        w_uniform = np.full(len(use), 1.0 / len(use))
        
        # Retrain full pool on restricted features
        X_tr_rest = X_train_full[use].values
        clf_r = P.build_classifier(model_name, tuned_full, y_train_full.values)
        pipe_r = P.make_pipeline_for(model_name, clf_r)
        pipe_r.fit(X_tr_rest, y_train_full.values)
        
        # Coral aligned
        Xc_tr, Xc_te = P.coral_align(X_tr_rest, X_tr_rest, X_t.values)
        clf_c = P.build_classifier(model_name, tuned_full, y_train_full.values)
        pipe_c = P.make_pipeline_for(model_name, clf_c)
        pipe_c.fit(Xc_tr, y_train_full.values)
        
        # Test original
        pred = pipe_r.predict(X_t.values)
        proba = pipe_r.predict_proba(X_t.values)[:, 1]
        rows.append({"target": target, "variant": "original", "f1": f1_score(y_t, pred, average="macro"), "auc": roc_auc_score(y_t, proba)})
        
        # Test shap weighted (we multiply both source and target by w_target, then train/test)
        # Actually in zero-shot, we just multiply target by w_target before predicting if we want to simulate the paper.
        # But wait, the paper multiplies both training and testing.
        pipe_w = P.make_pipeline_for(model_name, P.build_classifier(model_name, tuned_full, y_train_full.values))
        pipe_w.fit(X_tr_rest * w_target, y_train_full.values)
        pred_w = pipe_w.predict(X_t.values * w_target)
        proba_w = pipe_w.predict_proba(X_t.values * w_target)[:, 1]
        rows.append({"target": target, "variant": "shap_sum1", "f1": f1_score(y_t, pred_w, average="macro"), "auc": roc_auc_score(y_t, proba_w)})
        
        # Test uniform
        pipe_u = P.make_pipeline_for(model_name, P.build_classifier(model_name, tuned_full, y_train_full.values))
        pipe_u.fit(X_tr_rest * w_uniform, y_train_full.values)
        pred_u = pipe_u.predict(X_t.values * w_uniform)
        proba_u = pipe_u.predict_proba(X_t.values * w_uniform)[:, 1]
        rows.append({"target": target, "variant": "uniform", "f1": f1_score(y_t, pred_u, average="macro"), "auc": roc_auc_score(y_t, proba_u)})
        
        # Test coral
        pred_c = pipe_c.predict(Xc_te)
        proba_c = pipe_c.predict_proba(Xc_te)[:, 1]
        rows.append({"target": target, "variant": "coral_aligned", "f1": f1_score(y_t, pred_c, average="macro"), "auc": roc_auc_score(y_t, proba_c)})
        
    cpdp_df = pd.DataFrame(rows)
    cpdp_df.to_csv(OUT_DIR / "cpdp_scores.csv", index=False)
    print(cpdp_df.to_string(index=False))

if __name__ == "__main__":
    run_cpdp()
