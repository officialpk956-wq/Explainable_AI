"""
Task 6: Genuine CPDP Harness
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.metrics.pairwise import cosine_similarity

import pipeline as P
import cross_ecosystem as XE

warnings.filterwarnings("ignore")

def get_pooled_source(ecosystem):
    if ecosystem == "eclipse":
        sources = ["eclipse", "mylyn"]
        load_fn = P.load_xy
    else:
        sources = ["ant-1.7", "camel-1.6"]
        load_fn = XE.load_promise
        
    X_pool, y_pool = [], []
    cols = None
    for src in sources:
        X, y, c = load_fn(src)
        X_pool.append(X)
        y_pool.append(y)
        if cols is None:
            cols = c
            
    return pd.concat(X_pool, ignore_index=True), pd.concat(y_pool, ignore_index=True), cols
    
def get_target(ecosystem, target_name):
    if ecosystem == "eclipse":
        return P.load_xy(target_name)
    else:
        return XE.load_promise(target_name)

def run_task_6():
    print("=== Task 6: Genuine CPDP ===")
    
    apache_w = pd.read_csv(P.OUT_DIR / "xe_shap_weights.csv")
    apache_w_map = dict(apache_w[["Feature", "Weight"]].values)
    
    eclipse_w = pd.read_csv(P.OUT_DIR / "shap_weights.csv")
    eclipse_w_map = dict(eclipse_w[["Feature", "Weight"]].values)
    
    w_maps = {"eclipse": eclipse_w_map, "apache": apache_w_map}
    targets = {
        "eclipse": ["equinox", "lucene", "pde"],
        "apache": ["xalan-2.6", "poi-3.0", "velocity-1.6"]
    }
    
    zero_rows = []
    few_rows = []
    
    for eco in ["eclipse", "apache"]:
        print(f"Ecosystem: {eco}")
        X_src, y_src, cols_src = get_pooled_source(eco)
        w_map = w_maps[eco]
        use = [c for c in cols_src if c in w_map]
        
        X_src = X_src[use]
        F = len(use)
        
        imp_array = np.array([w_map[c] for c in use])
        shap_sum1 = imp_array / np.sum(imp_array)
        uniform = np.ones(F) / F
        
        # tune on pooled source
        tuned = P.tune_hyperparams(X_src, y_src, f"cpdp_{eco}")
        
        for target_name in targets[eco]:
            print(f"  Target: {target_name}")
            X_tgt, y_tgt, t_cols = get_target(eco, target_name)
            valid = [c for c in use if c in t_cols]
            if len(valid) == 0:
                print(f"Skipping {target_name} because 0 overlapping features.")
                continue
                
            X_tgt = X_tgt[valid]
            
            imp_array = np.array([w_map[c] for c in valid])
            F = len(valid)
            shap_sum1 = imp_array / np.sum(imp_array)
            uniform = np.ones(F) / F
            
            # Subsets
            for seed in P.repeated_seeds():
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                for train_idx, _ in sss.split(X_src, y_src):
                    X_src_boot = X_src[valid].iloc[train_idx].values
                    y_src_boot = y_src.iloc[train_idx].values
                    
                # For CORAL
                _, X_tgt_coral = P.coral_align(X_src_boot, X_tgt.values, X_tgt.values)
                X_src_coral = X_src_boot.copy()
                
                # Zero-shot
                for model_name in P.make_classifiers():
                    clf = P.build_classifier(model_name, tuned, y_src_boot)
                    
                    def eval_arm(arm_name, multiplier, X_t_eval, y_t_eval, X_s_train=None, y_s_train=None):
                        if X_s_train is None:
                            X_s_train = X_src_boot.copy()
                            y_s_train = y_src_boot.copy()
                            
                        if arm_name == "coral_aligned":
                            # coral already applied
                            pass
                        elif multiplier is not None:
                            X_s_train = X_s_train * multiplier
                            X_t_eval = X_t_eval * multiplier
                            
                        pipe = P.make_pipeline_for(model_name, clf)
                        pipe.fit(X_s_train, y_s_train)
                        pred = pipe.predict(X_t_eval)
                        proba = pipe.predict_proba(X_t_eval)[:, 1]
                        return f1_score(y_t_eval, pred, average="macro"), roc_auc_score(y_t_eval, proba), average_precision_score(y_t_eval, proba)
                        
                    f1, auc, pr = eval_arm("original", None, X_tgt.values, y_tgt.values)
                    zero_rows.append({"ecosystem": eco, "target": target_name, "model": model_name, "arm": "original", "boot_seed": seed, "f1": f1, "auc": auc, "prauc": pr})
                    
                    f1, auc, pr = eval_arm("shap_sum1", shap_sum1, X_tgt.values, y_tgt.values)
                    zero_rows.append({"ecosystem": eco, "target": target_name, "model": model_name, "arm": "shap_sum1", "boot_seed": seed, "f1": f1, "auc": auc, "prauc": pr})
                    
                    f1, auc, pr = eval_arm("uniform", uniform, X_tgt.values, y_tgt.values)
                    zero_rows.append({"ecosystem": eco, "target": target_name, "model": model_name, "arm": "uniform", "boot_seed": seed, "f1": f1, "auc": auc, "prauc": pr})
                    
                    f1, auc, pr = eval_arm("coral_aligned", None, X_tgt_coral, y_tgt.values, X_src_coral, y_src_boot)
                    zero_rows.append({"ecosystem": eco, "target": target_name, "model": model_name, "arm": "coral_aligned", "boot_seed": seed, "f1": f1, "auc": auc, "prauc": pr})
                    
                # Few-shot
                for k in [10, 25, 50]:
                    # draw k instances from target
                    sss_k = StratifiedShuffleSplit(n_splits=1, train_size=k, random_state=seed)
                    for k_idx, rem_idx in sss_k.split(X_tgt, y_tgt):
                        X_k = X_tgt.iloc[k_idx].values
                        y_k = y_tgt.iloc[k_idx].values
                        X_rem = X_tgt.iloc[rem_idx].values
                        y_rem = y_tgt.iloc[rem_idx].values
                        
                    X_mix = np.vstack([X_src_boot, X_k])
                    y_mix = np.concatenate([y_src_boot, y_k])
                    
                    X_mix_coral, X_rem_coral = P.coral_align(X_src_boot, X_mix, X_rem)
                    
                    for model_name in P.make_classifiers():
                        clf = P.build_classifier(model_name, tuned, y_mix)
                        
                        def eval_arm_fs(arm_name, multiplier):
                            if arm_name == "coral_aligned":
                                X_tr, X_te = X_mix_coral.copy(), X_rem_coral.copy()
                            else:
                                X_tr, X_te = X_mix.copy(), X_rem.copy()
                                if multiplier is not None:
                                    X_tr = X_tr * multiplier
                                    X_te = X_te * multiplier
                            pipe = P.make_pipeline_for(model_name, clf)
                            pipe.fit(X_tr, y_mix)
                            pred = pipe.predict(X_te)
                            proba = pipe.predict_proba(X_te)[:, 1]
                            return f1_score(y_rem, pred, average="macro"), roc_auc_score(y_rem, proba), average_precision_score(y_rem, proba)
                            
                        f1, auc, pr = eval_arm_fs("original", None)
                        few_rows.append({"ecosystem": eco, "target": target_name, "model": model_name, "arm": "original", "k": k, "draw_seed": seed, "f1": f1, "auc": auc, "prauc": pr})
                        
                        f1, auc, pr = eval_arm_fs("shap_sum1", shap_sum1)
                        few_rows.append({"ecosystem": eco, "target": target_name, "model": model_name, "arm": "shap_sum1", "k": k, "draw_seed": seed, "f1": f1, "auc": auc, "prauc": pr})
                        
                        f1, auc, pr = eval_arm_fs("uniform", uniform)
                        few_rows.append({"ecosystem": eco, "target": target_name, "model": model_name, "arm": "uniform", "k": k, "draw_seed": seed, "f1": f1, "auc": auc, "prauc": pr})
                        
                        f1, auc, pr = eval_arm_fs("coral_aligned", None)
                        few_rows.append({"ecosystem": eco, "target": target_name, "model": model_name, "arm": "coral_aligned", "k": k, "draw_seed": seed, "f1": f1, "auc": auc, "prauc": pr})
                        
    df_zero = pd.DataFrame(zero_rows)
    df_zero.to_csv(P.OUT_DIR / "cpdp_zeroshot_scores.csv", index=False)
    
    df_few = pd.DataFrame(few_rows)
    df_few.to_csv(P.OUT_DIR / "cpdp_fewshot_scores.csv", index=False)
    
    # 6c Significance
    sig_rows = []
    
    for df_scores, setting in [(df_zero, "zero-shot"), (df_few, "few-shot")]:
        k_vals = df_scores["k"].unique() if "k" in df_scores.columns else [None]
        for target in df_scores.target.unique():
            for model in df_scores.model.unique():
                for k_val in k_vals:
                    sub = df_scores[(df_scores.target == target) & (df_scores.model == model)]
                    if k_val is not None:
                        sub = sub[sub.k == k_val]
                        
                    for metric in ["f1", "auc", "prauc"]:
                        seed_col = "draw_seed" if k_val is not None else "boot_seed"
                        sub = sub.sort_values(seed_col)
                        
                        orig = sub[sub.arm == "original"][metric].values
                        for arm in ["shap_sum1", "uniform", "coral_aligned"]:
                            A = sub[sub.arm == arm][metric].values
                            if np.all(A == orig):
                                p = 1.0
                            else:
                                try:
                                    p = wilcoxon(A, orig).pvalue
                                except ValueError:
                                    p = 1.0
                            
                            sig_rows.append({
                                "setting": f"few-shot_k={k_val}" if k_val is not None else "zero-shot",
                                "target": target,
                                "model": model,
                                "metric": metric,
                                "arm": arm,
                                "mean_diff": (A - orig).mean(),
                                "wilcoxon_p": p
                            })
                            
    sig_df = pd.DataFrame(sig_rows)
    adj, sig = P.holm_bonferroni(sig_df["wilcoxon_p"].values)
    sig_df["holm_corrected_p"] = adj
    sig_df["significant_at_0.05"] = sig
    sig_df.to_csv(P.OUT_DIR / "cpdp_significance.csv", index=False)
    
    # 6d LOSO Weight Stability
    print("=== Task 6d: LOSO Weight Stability ===")
    loso_w_rows = []
    eclipse_projs = ["eclipse", "mylyn", "equinox", "lucene", "pde"]
    apache_projs = ["ant-1.7", "camel-1.6", "xalan-2.6", "poi-3.0", "velocity-1.6"]
    
    vectors = {"eclipse": {}, "apache": {}}
    
    for eco, projs, load_fn in [("eclipse", eclipse_projs, P.load_xy), ("apache", apache_projs, XE.load_promise)]:
        for proj in projs:
            X, y, cols = load_fn(proj)
            w_map = w_maps[eco]
            use = [c for c in cols if c in w_map]
            X = X[use]
            
            X_tr0, _, y_tr0, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=P.SEED)
            tuned = P.tune_hyperparams(X_tr0, y_tr0, f"loso_{proj}")
            
            w_df = pd.DataFrame(P.shap_weights_out_of_fold(X, y, proj, tuned))
            w_vec = dict(zip(w_df["Feature"], w_df["Weight"]))
            
            vec_arr = np.array([w_vec.get(c, 0) for c in use])
            vectors[eco][proj] = vec_arr
            
            for c in use:
                loso_w_rows.append({
                    "ecosystem": eco,
                    "source_project": proj,
                    "feature": c,
                    "weight": w_vec.get(c, 0)
                })
                
    pd.DataFrame(loso_w_rows).to_csv(P.OUT_DIR / "loso_weight_vectors.csv", index=False)
    
    stab_rows = []
    for eco in ["eclipse", "apache"]:
        projs = list(vectors[eco].keys())
        for i in range(len(projs)):
            for j in range(i+1, len(projs)):
                p_a = projs[i]
                p_b = projs[j]
                v_a = vectors[eco][p_a]
                v_b = vectors[eco][p_b]
                
                rho, _ = spearmanr(v_a, v_b)
                cos_sim = cosine_similarity(v_a.reshape(1, -1), v_b.reshape(1, -1))[0,0]
                
                stab_rows.append({
                    "ecosystem": eco,
                    "project_a": p_a,
                    "project_b": p_b,
                    "spearman_rho": rho,
                    "cosine_sim": cos_sim
                })
                
    pd.DataFrame(stab_rows).to_csv(P.OUT_DIR / "loso_weight_stability.csv", index=False)
    print("Done Task 6.")

if __name__ == "__main__":
    run_task_6()
