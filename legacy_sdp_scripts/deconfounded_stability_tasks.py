"""
Task 2: Deconfounded Stability Metric
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, linregress
from sklearn.model_selection import train_test_split
import shap

import pipeline as P
import powered_phase_h as PH
import cross_ecosystem as XE

warnings.filterwarnings("ignore")

def get_shap_values(clf, model_name, X_train_trans, X_test_inst_trans):
    def predict_proba(X):
        return clf.predict_proba(X)
        
    if model_name in ["RandomForest", "XGBoost", "LightGBM", "DecisionTree", "ExtraTrees", "GradientBoosting"]:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_test_inst_trans)
        if isinstance(sv, list):
            sv = sv[1]
    else:
        bg = shap.sample(X_train_trans, 100, random_state=42)
        explainer = shap.KernelExplainer(predict_proba, bg)
        sv = explainer.shap_values(X_test_inst_trans, nsamples=100, silent=True)
        if isinstance(sv, list):
            sv = sv[1]
    return sv

def aopc(clf, X_inst, order, feature_means):
    x_curr = X_inst.copy().reshape(1, -1)
    
    initial_proba = clf.predict_proba(x_curr)[0]
    pred_class = np.argmax(initial_proba)
    p0 = initial_proba[pred_class]
    
    drops = []
    x_curr = x_curr[0]
    for idx in order:
        x_curr[idx] = feature_means[idx]
        pk = clf.predict_proba(x_curr.reshape(1, -1))[0, pred_class]
        drops.append(max(0.0, p0 - pk))
        
    return np.mean(drops)

def run_task_2():
    print("=== Task 2a: AOPC full grid ===")
    rows_2a = []
    datasets = P.PHASE1 + P.PHASE2 + XE.XE_SOURCES + XE.XE_TARGETS
    # total 10 datasets
    P.TREE_MODELS = list(set(list(P.TREE_MODELS) + list(PH.NEW_TREE_LIKE)))
    models = list(P.make_classifiers().keys())
    
    for dataset in datasets:
        print(f"Dataset: {dataset}")
        if dataset in XE.XE_SOURCES or dataset in XE.XE_TARGETS:
            X, y, cols = XE.load_promise(dataset)
            X = X[cols]
        else:
            X, y, cols = P.load_xy(dataset)
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=P.SEED)
            
        tuned = P.tune_hyperparams(X_train, y_train, f"task2_{dataset}")
        
        extended_models = PH.extended_classifiers()
        for model_name in models:
            if model_name in P.make_classifiers():
                clf = P.build_classifier(model_name, tuned, y_train.values)
            else:
                clf = extended_models[model_name]
            pipe = P.make_pipeline_for(model_name, clf)
            pipe.fit(X_train.values, y_train.values)
            clf_fitted = pipe.named_steps["clf"]
            
            proba = pipe.predict_proba(X_test.values)[:, 1]
            inst_indices = P.select_xai_instances(proba)
            
            X_tr_trans = P.transform_only(pipe, X_train.values)
            X_te_trans = P.transform_only(pipe, X_test.values)
            feature_means = np.mean(X_tr_trans, axis=0)
            
            for idx in inst_indices:
                x_test_inst = X_te_trans[idx]
                sv = get_shap_values(clf_fitted, model_name, X_tr_trans, x_test_inst.reshape(1, -1))[0]
                
                # SHAP order: descending magnitude of SHAP values
                shap_order = np.argsort(-np.abs(sv))
                aopc_s = aopc(clf_fitted, x_test_inst, shap_order, feature_means)
                
                # Random permutations (average of 5)
                rnd_aopc = []
                for s in range(5):
                    rng = np.random.RandomState(P.SEED + s)
                    rand_order = rng.permutation(len(feature_means))
                    rnd_aopc.append(aopc(clf_fitted, x_test_inst, rand_order, feature_means))
                aopc_r = np.mean(rnd_aopc)
                
                rows_2a.append({
                    "dataset": dataset,
                    "model": model_name,
                    "instance": idx,
                    "aopc_shap": aopc_s,
                    "aopc_random": aopc_r,
                    "faith_gain": aopc_s - aopc_r
                })
                
    df_2a = pd.DataFrame(rows_2a)
    df_2a.to_csv(P.OUT_DIR / "aopc_full_grid.csv", index=False)
    
    print("=== Task 2b: Deconfounded stability ===")
    h_splithalf = pd.read_csv(P.OUT_DIR / "h_splithalf.csv")
    rows_2b = []
    
    for dataset in datasets:
        sub = h_splithalf[h_splithalf.dataset == dataset].copy()
        if len(sub) == 0:
            print(f"Warning: no split half data for {dataset}")
            continue
            
        x_val = np.log(1 - sub["r2_all"])
        y_val = np.log(sub["sigma_all"])
        slope, intercept, r_value, p_value, std_err = linregress(x_val, y_val)
        
        pred_y = intercept + slope * x_val
        sigma_star = np.exp(y_val - pred_y)
        sub["sigma_star"] = sigma_star
        
        # Rank by raw sigma (lower is better, so rank 1 is lowest sigma)
        sub["rank_raw"] = sub["sigma_all"].rank(method="min", ascending=True)
        sub["rank_deconfounded"] = sub["sigma_star"].rank(method="min", ascending=True)
        sub["rank_changed"] = sub["rank_raw"] != sub["rank_deconfounded"]
        
        for _, r in sub.iterrows():
            rows_2b.append({
                "dataset": dataset,
                "model": r["model"],
                "sigma_all": r["sigma_all"],
                "r2_all": r["r2_all"],
                "sigma_star": r["sigma_star"],
                "rank_raw": r["rank_raw"],
                "rank_deconfounded": r["rank_deconfounded"],
                "rank_changed": r["rank_changed"]
            })
            
    df_2b = pd.DataFrame(rows_2b)
    df_2b.to_csv(P.OUT_DIR / "deconfounded_stability.csv", index=False)
    
    print("=== Task 2c: Validation ===")
    val_rows = []
    faith_agg = df_2a.groupby(["dataset", "model"])["faith_gain"].mean().reset_index()
    
    for dataset in datasets:
        sub_2b = df_2b[df_2b.dataset == dataset]
        sub_faith = faith_agg[faith_agg.dataset == dataset]
        
        merged = pd.merge(sub_2b, sub_faith, on="model")
        
        if len(merged) < 2:
            continue
            
        rho_raw, _ = spearmanr(merged["sigma_all"], merged["faith_gain"])
        rho_star, _ = spearmanr(merged["sigma_star"], merged["faith_gain"])
        
        improved = rho_star < rho_raw
        
        val_rows.append({
            "dataset": dataset,
            "rho_sigma_raw_vs_faith": rho_raw,
            "rho_sigma_star_vs_faith": rho_star,
            "improved": improved
        })
        
    df_val = pd.DataFrame(val_rows)
    df_val.to_csv(P.OUT_DIR / "deconfounded_validation.csv", index=False)
    
    n_improved = df_val["improved"].sum()
    try:
        w_p = wilcoxon(df_val["rho_sigma_raw_vs_faith"], df_val["rho_sigma_star_vs_faith"]).pvalue
    except:
        w_p = 1.0
        
    claim_supported = bool(n_improved >= 8 and w_p < 0.05)
    
    pd.DataFrame([{
        "n_improved": n_improved,
        "wilcoxon_p": w_p,
        "claim_supported": claim_supported
    }]).to_csv(P.OUT_DIR / "deconfounded_verdict.csv", index=False)
    print("Done Task 2.")

if __name__ == "__main__":
    run_task_2()
