"""
Task 7: SHAP-As-Prior Fixed
"""
import warnings
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import FunctionTransformer

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

def run_task_7():
    print("=== Task 7: SHAP-As-Prior ===")
    
    apache_w = pd.read_csv(P.OUT_DIR / "xe_shap_weights.csv")
    apache_w_map = dict(apache_w[["Feature", "Weight"]].values)
    
    eclipse_w = pd.read_csv(P.OUT_DIR / "shap_weights.csv")
    eclipse_w_map = dict(eclipse_w[["Feature", "Weight"]].values)
    
    w_maps = {"eclipse": eclipse_w_map, "apache": apache_w_map}
    targets = {
        "eclipse": ["equinox", "lucene", "pde"],
        "apache": ["xalan-2.6", "poi-3.0", "velocity-1.6"]
    }
    
    models = ["LogisticRegression", "SVC", "XGBoost", "LightGBM"]
    
    zero_rows = []
    few_rows = []
    
    group_vectors_nonconstant = True
    
    for eco in ["eclipse", "apache"]:
        print(f"Ecosystem: {eco}")
        X_src, y_src, cols_src = get_pooled_source(eco)
        w_map = w_maps[eco]
        use = [c for c in cols_src if c in w_map]
        
        X_src = X_src[use]
        F = len(use)
        
        corr = X_src.corr().fillna(0).values
        dist = np.clip(1 - np.abs(corr), 0, 1)
        np.fill_diagonal(dist, 0)
        condensed_dist = squareform(dist)
        
        Z = linkage(condensed_dist, method='average')
        # cut so within-cluster |corr| >= 0.8 => distance <= 0.2
        groups = fcluster(Z, t=0.2, criterion='distance')
        
        group_map = {}
        for i, c in enumerate(use):
            g = groups[i]
            if g not in group_map:
                group_map[g] = []
            group_map[g].append(c)
            
        prior = {}
        for g, feats in group_map.items():
            total_w = sum(w_map[f] for f in feats)
            for f in feats:
                prior[f] = total_w / len(feats)
                
        base_prior_vec = np.array([prior[c] for c in use])
        
        # MANDATORY SANITY CHECK
        if np.std(base_prior_vec) <= 1e-9:
            print(f"WARNING: Group importance vector for {eco} is constant! STD={np.std(base_prior_vec)}")
            group_vectors_nonconstant = False
            
        tuned = P.tune_hyperparams(X_src, y_src, f"shap_prior_{eco}")
        
        for target_name in targets[eco]:
            X_tgt, y_tgt, t_cols = get_target(eco, target_name)
            valid = [c for c in use if c in t_cols]
            if len(valid) == 0:
                print(f"Skipping {target_name} because 0 overlapping features.")
                continue
                
            X_tgt = X_tgt[valid]
            F_valid = len(valid)
            
            for seed in P.repeated_seeds():
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                for train_idx, _ in sss.split(X_src, y_src):
                    X_src_boot = X_src[valid].iloc[train_idx].values
                    y_src_boot = y_src.iloc[train_idx].values
                    
                rng = np.random.RandomState(seed)
                
                def eval_setting(X_train, y_train, X_test, y_test, setting_name, is_few_shot, k_val=None):
                    for model_name in models:
                        clf = P.build_classifier(model_name, tuned, y_train) if model_name != "LightGBM" else P.build_classifier("LogisticRegression", None, None) # Placeholder if no LGBM
                        
                        if model_name == "LightGBM":
                            import lightgbm as lgb
                            clf = lgb.LGBMClassifier(random_state=P.SEED)
                            if tuned and "LightGBM" in tuned:
                                clf.set_params(**tuned["LightGBM"])
                                
                        for tau, tau_label in [(1.0, ""), (0.5, "_tau0.5")]:
                            valid_idx = [use.index(c) for c in valid]
                            sub_prior = base_prior_vec[valid_idx]
                            tau_prior_vec = sub_prior ** (1.0 / tau)
                            prior_vec = tau_prior_vec / np.mean(tau_prior_vec)
                            uniform_vec = np.ones(F_valid)
                            
                            shuffled_vecs = []
                            for d in range(3):
                                rng_shuf = np.random.RandomState(1000 * seed + d)
                                shuf_prior = rng_shuf.permutation(tau_prior_vec)
                                shuffled_vecs.append(shuf_prior / np.mean(shuf_prior))
                            shuffled_vec = np.mean(shuffled_vecs, axis=0)
                            
                            def run_arm(arm_name, vec):
                                pipe_steps = []
                                # Make pipeline mimicking P.make_pipeline_for but inserting FunctionTransformer
                                from sklearn.pipeline import Pipeline
                                from sklearn.preprocessing import RobustScaler
                                from imblearn.pipeline import Pipeline as ImbPipeline
                                from pipeline import ConditionalSMOTE
                                
                                def log1p_trans(X):
                                    return np.log1p(np.abs(X))
                                    
                                pipe_steps = [
                                    ("log1p", FunctionTransformer(log1p_trans)),
                                    ("scaler", RobustScaler())
                                ]
                                
                                if arm_name != "original" and model_name in ["LogisticRegression", "SVC"]:
                                    def scale_prior(X, v=vec):
                                        return X * np.sqrt(v)
                                    pipe_steps.append(("prior_scaler", FunctionTransformer(scale_prior)))
                                    
                                pipe_steps.append(("smote", ConditionalSMOTE()))
                                pipe_steps.append(("clf", clf))
                                
                                pipe = ImbPipeline(pipe_steps)
                                
                                if arm_name != "original" and model_name == "XGBoost":
                                    # pass feature_weights to fit
                                    try:
                                        pipe.fit(X_train, y_train, clf__feature_weights=vec)
                                    except Exception as e:
                                        print(f"XGBoost feature_weights failed: {e}")
                                        pipe.fit(X_train, y_train)
                                elif arm_name != "original" and model_name == "LightGBM":
                                    try:
                                        clf.set_params(feature_contri=vec.tolist())
                                        pipe.fit(X_train, y_train)
                                    except Exception as e:
                                        print(f"LightGBM feature_contri failed: {e}")
                                        pipe.fit(X_train, y_train)
                                else:
                                    pipe.fit(X_train, y_train)
                                    
                                # Verify order
                                if arm_name == "shap_prior" and tau == 1.0 and model_name == "LogisticRegression" and seed == 42 and target_name == targets[eco][0]:
                                    step_names = [s[0] for s in pipe.steps]
                                    print(f"Pipeline steps for {model_name} {arm_name}: {step_names}")
                                    
                                pred = pipe.predict(X_test)
                                proba = pipe.predict_proba(X_test)[:, 1]
                                f1 = f1_score(y_test, pred, average="macro")
                                auc = roc_auc_score(y_test, proba)
                                pr = average_precision_score(y_test, proba)
                                
                                row = {
                                    "target": target_name,
                                    "model": model_name,
                                    "arm": arm_name + tau_label,
                                    "f1": f1,
                                    "auc": auc,
                                    "prauc": pr
                                }
                                if is_few_shot:
                                    row["k"] = k_val
                                    row["draw_seed"] = seed
                                    few_rows.append(row)
                                else:
                                    row["boot_seed"] = seed
                                    zero_rows.append(row)
                                    
                            run_arm("shap_prior", prior_vec)
                            run_arm("uniform", uniform_vec)
                            run_arm("shuffled", shuffled_vec)
                            if tau == 1.0:
                                run_arm("original", uniform_vec) # no prior
                                
                eval_setting(X_src_boot, y_src_boot, X_tgt.values, y_tgt.values, "zero-shot", False)
                
                k = 25
                sss_k = StratifiedShuffleSplit(n_splits=1, train_size=k, random_state=seed)
                for k_idx, rem_idx in sss_k.split(X_tgt, y_tgt):
                    X_k = X_tgt.iloc[k_idx].values
                    y_k = y_tgt.iloc[k_idx].values
                    X_rem = X_tgt.iloc[rem_idx].values
                    y_rem = y_tgt.iloc[rem_idx].values
                    
                X_mix = np.vstack([X_src_boot, X_k])
                y_mix = np.concatenate([y_src_boot, y_k])
                eval_setting(X_mix, y_mix, X_rem, y_rem, "few-shot", True, k)
                
    df_zero = pd.DataFrame(zero_rows)
    df_zero.to_csv(P.OUT_DIR / "shap_prior_scores.csv", index=False)
    
    df_few = pd.DataFrame(few_rows)
    df_few.to_csv(P.OUT_DIR / "shap_prior_fewshot_scores.csv", index=False)
    
    # Tests (only for primary zero-shot tau=1.0)
    sig_rows = []
    
    for target in df_zero.target.unique():
        for model in models:
            for metric in ["f1", "auc", "prauc"]:
                sub = df_zero[(df_zero.target == target) & (df_zero.model == model)].sort_values("boot_seed")
                
                p_vals = sub[sub.arm == "shap_prior"][metric].values
                u_vals = sub[sub.arm == "uniform"][metric].values
                s_vals = sub[sub.arm == "shuffled"][metric].values
                
                for comp, A, B in [("prior_vs_uniform", p_vals, u_vals), ("prior_vs_shuffled", p_vals, s_vals)]:
                    if np.all(A == B):
                        p = 1.0
                    else:
                        try:
                            p = wilcoxon(A, B).pvalue
                        except ValueError:
                            p = 1.0
                            
                    sig_rows.append({
                        "target": target,
                        "model": model,
                        "metric": metric,
                        "comparison": comp,
                        "mean_diff": (A - B).mean(),
                        "wilcoxon_p": p
                    })
                    
    sig_df = pd.DataFrame(sig_rows)
    adj, sig = P.holm_bonferroni(sig_df["wilcoxon_p"].values)
    sig_df["holm_corrected_p"] = adj
    sig_df["significant_at_0.05"] = sig
    sig_df.to_csv(P.OUT_DIR / "shap_prior_tests.csv", index=False)
    
    vs_u = sig_df[sig_df.comparison == "prior_vs_uniform"]
    vs_s = sig_df[sig_df.comparison == "prior_vs_shuffled"]
    
    wins_u = len(vs_u[(vs_u["significant_at_0.05"]) & (vs_u.mean_diff > 0)])
    wins_s = len(vs_s[(vs_s["significant_at_0.05"]) & (vs_s.mean_diff > 0)])
    
    loss_u = len(vs_u[(vs_u["significant_at_0.05"]) & (vs_u.mean_diff < 0)])
    loss_s = len(vs_s[(vs_s["significant_at_0.05"]) & (vs_s.mean_diff < 0)])
    
    passed = (wins_u >= 2) and (wins_s >= 2) and (loss_u == 0) and (loss_s == 0)
    
    pd.DataFrame([{
        "wins_uniform": wins_u,
        "wins_shuffled": wins_s,
        "losses_uniform": loss_u,
        "losses_shuffled": loss_s,
        "passed": passed,
        "group_vectors_nonconstant": group_vectors_nonconstant
    }]).to_csv(P.OUT_DIR / "shap_prior_verdict.csv", index=False)
    
    print("Done Task 7.")

if __name__ == "__main__":
    run_task_7()
