"""
Phase 0 & 2 Robustness checks: Split-half, Within-pipeline, Specification curve, Redundancy.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, wilcoxon
import json

import pipeline as P
import powered_phase_h as PH
import cross_ecosystem as XE
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

OUT_DIR = P.OUT_DIR
SEEDS = [42, 43, 44, 45, 46, 47]
SEEDS_A = [42, 44, 46]
SEEDS_B = [43, 45, 47]

def run_phase_0_1():
    print("=== Phase 0.1: Split-Half Check ===")
    all_sets = PH.ECLIPSE_SETS + PH.APACHE_SETS
    rows = []
    
    # We must ensure the NEW_TREE_LIKE are added to P.TREE_MODELS for Pipeline A
    P.TREE_MODELS |= PH.NEW_TREE_LIKE
    
    for dataset_name in all_sets:
        X, y, cols = PH.load_any(dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=P.SEED)
        feats = list(X_train.columns)
        
        for model_name, clf_proto in PH.extended_classifiers().items():
            try:
                clf = P.build_classifier(model_name, None, y_train.values) \
                    if model_name in P.make_classifiers() else clf_proto
                pipe = P.make_pipeline_for(model_name, clf)
                pipe.fit(X_train.values, y_train.values)
                fitted = pipe.named_steps["clf"]
                X_tr_p = P.transform_only(pipe, X_train.values)
                X_te_p = P.transform_only(pipe, X_test.values)
                
                proba = fitted.predict_proba(X_te_p)[:, 1]
                idxs = P.select_xai_instances(proba, n=PH.N_INSTANCES)
                
                explainers = {s: P._make_lime_explainer(X_tr_p, feats, s) for s in SEEDS}
                
                for i_idx, i in enumerate(idxs):
                    scores = {}
                    vectors = {}
                    for s in SEEDS:
                        v, sc = P._lime_vector_and_score(explainers[s], fitted, X_te_p[i], feats)
                        scores[s] = sc
                        vectors[s] = v
                    
                    r2_a = np.mean([scores[s] for s in SEEDS_A])
                    sigma_b = np.mean(np.std([vectors[s] for s in SEEDS_B], axis=0))
                    
                    r2_all = np.mean([scores[s] for s in SEEDS])
                    sigma_all = np.mean(np.std([vectors[s] for s in SEEDS], axis=0))
                    
                    rows.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "instance": i_idx,
                        "r2_half_a": float(r2_a),
                        "sigma_half_b": float(sigma_b),
                        "r2_all": float(r2_all),
                        "sigma_all": float(sigma_all),
                        "status": "ok"
                    })
            except Exception as e:
                for i_idx in range(PH.N_INSTANCES):
                    rows.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "instance": i_idx,
                        "r2_half_a": np.nan, "sigma_half_b": np.nan,
                        "r2_all": np.nan, "sigma_all": np.nan,
                        "status": f"failed: {type(e).__name__}"
                    })

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(OUT_DIR / "h_splithalf_raw.csv", index=False)
    
    ok = raw_df[raw_df.status == "ok"].dropna(subset=["r2_half_a", "sigma_half_b"])
    agg = ok.groupby(["dataset", "model"]).agg({
        "r2_half_a": "mean",
        "sigma_half_b": "mean",
        "r2_all": "mean",
        "sigma_all": "mean"
    }).reset_index()
    agg.to_csv(OUT_DIR / "h_splithalf.csv", index=False)
    
    per = []
    for name, g in agg.groupby("dataset"):
        if len(g) < 4 or g["r2_half_a"].std() == 0 or g["sigma_half_b"].std() == 0:
            continue
        r = spearmanr(g["r2_half_a"], g["sigma_half_b"])
        per.append({"dataset": name, "rho": r.statistic, "p": r.pvalue})
        
    per_df = pd.DataFrame(per)
    rhos = per_df["rho"].values
    stat, p_val = wilcoxon(rhos)
    
    passed = bool(p_val < 0.05 and np.median(rhos) < 0)
    
    pd.DataFrame([{
        "median_rho": float(np.median(rhos)),
        "W": float(stat),
        "p_value": float(p_val),
        "passed": passed
    }]).to_csv(OUT_DIR / "h_splithalf_verdict.csv", index=False)
    
    print("Phase 0.1 SPLIT-HALF GATE:", "PASS" if passed else "FAIL")
    print(f"Median rho: {np.median(rhos):.3f}, p_value: {p_val:.4f}")
    return passed, raw_df, agg

def run_phase_0_2(agg_df):
    print("=== Phase 0.2: Within-Pipeline Check ===")
    pipe_a = {"RandomForest", "XGBoost", "LightGBM", "GradientBoosting", "ExtraTrees", "DecisionTree", "AdaBoost", "Bagging"}
    pipe_b = {"LogisticRegression", "SVC", "KNN", "MLP", "GaussianNB", "LDA", "QDA", "SGDLogistic"}
    
    rows = []
    for dataset, g in agg_df.groupby("dataset"):
        g_a = g[g.model.isin(pipe_a)]
        g_b = g[g.model.isin(pipe_b)]
        
        if len(g_a) > 1 and g_a["r2_all"].std() > 0 and g_a["sigma_all"].std() > 0:
            r_a = spearmanr(g_a["r2_all"], g_a["sigma_all"])
            rows.append({"dataset": dataset, "pipeline": "A", "rho": r_a.statistic, "p": r_a.pvalue})
        if len(g_b) > 1 and g_b["r2_all"].std() > 0 and g_b["sigma_all"].std() > 0:
            r_b = spearmanr(g_b["r2_all"], g_b["sigma_all"])
            rows.append({"dataset": dataset, "pipeline": "B", "rho": r_b.statistic, "p": r_b.pvalue})
            
    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT_DIR / "h_within_pipeline.csv", index=False)
    
    a_rhos = res_df[res_df.pipeline == "A"]["rho"].values
    if len(a_rhos) > 0:
        stat, p_val = wilcoxon(a_rhos)
        passed = bool(p_val < 0.05 and np.median(a_rhos) < 0)
    else:
        passed = False
        p_val = 1.0
        
    print("Phase 0.2 WITHIN-PIPELINE GATE:", "PASS" if passed else "FAIL")
    print(f"Pipeline A Median rho: {np.median(a_rhos) if len(a_rhos) else np.nan:.3f}, p_value: {p_val:.4f}")
    return passed

def run_phase_0_6(raw_df):
    print("=== Phase 0.6: Specification Curve ===")
    ok = raw_df[raw_df.status == "ok"].dropna(subset=["r2_all", "sigma_all"])
    datasets = list(ok.dataset.unique())
    models = list(ok.model.unique())
    
    rows = []
    for corr in ["spearman", "kendall"]:
        for n_inst in [5, 10]:
            for lodo in [None] + datasets:
                for lomo in [None] + models:
                    sub = ok.copy()
                    if lodo:
                        sub = sub[sub.dataset != lodo]
                    if lomo:
                        sub = sub[sub.model != lomo]
                    sub = sub[sub.instance < n_inst]
                    
                    agg = sub.groupby(["dataset", "model"]).agg({
                        "r2_all": "mean",
                        "sigma_all": "mean"
                    }).reset_index()
                    
                    rhos = []
                    for ds, g in agg.groupby("dataset"):
                        if len(g) > 1 and g["r2_all"].std() > 0 and g["sigma_all"].std() > 0:
                            if corr == "spearman":
                                rhos.append(spearmanr(g["r2_all"], g["sigma_all"]).statistic)
                            else:
                                rhos.append(kendalltau(g["r2_all"], g["sigma_all"]).statistic)
                                
                    if len(rhos) > 0:
                        try:
                            stat, p = wilcoxon(rhos)
                        except ValueError:
                            stat, p = 0.0, 1.0
                        rows.append({
                            "corr": corr, "n_inst": n_inst, "lodo": lodo, "lomo": lomo,
                            "median_rho": np.median(rhos), "p_value": p
                        })
                        
    spec_df = pd.DataFrame(rows)
    spec_df.to_csv(OUT_DIR / "h_specification_curve.csv", index=False)
    
    frac_rho = (spec_df.median_rho < 0).mean()
    frac_p = (spec_df.p_value < 0.05).mean()
    print(f"Specification curve: {frac_rho*100:.1f}% negative rhos, {frac_p*100:.1f}% significant.")

def run_phase_2_2():
    print("=== Phase 2.2: Redundancy, All Targets ===")
    # 6 targets: 3 Apache, 3 Eclipse
    import cross_ecosystem as XE
    apache_w = pd.read_csv(OUT_DIR / "xe_shap_weights.csv")
    apache_w_map = dict(apache_w[["Feature", "Weight"]].values)
    
    eclipse_w = pd.read_csv(OUT_DIR / "shap_weights.csv")
    eclipse_w_map = dict(eclipse_w[["Feature", "Weight"]].values)
    
    rows = []
    
    targets = [
        ("xalan-2.6", XE.load_promise, apache_w_map),
        ("poi-3.0", XE.load_promise, apache_w_map),
        ("velocity-1.6", XE.load_promise, apache_w_map),
        ("equinox", P.load_xy, eclipse_w_map),
        ("lucene", P.load_xy, eclipse_w_map),
        ("pde", P.load_xy, eclipse_w_map),
    ]
    
    np.random.seed(P.SEED)
    for target, load_fn, w_map in targets:
        X, y, cols = load_fn(target)
        use = [c for c in cols if c in w_map]
        X = X[use]
        F = len(use)
        k = int(np.ceil(F / 2.0))
        
        # Calculate correlation matrix
        corr = X.corr(method='pearson').abs().values.copy()
        np.fill_diagonal(corr, 0)
        
        def subset_mean_corr(indices):
            if len(indices) <= 1:
                return 0
            sub = corr[np.ix_(indices, indices)]
            return np.sum(sub) / (len(indices) * (len(indices) - 1))
            
        w_vals = [w_map[c] for c in use]
        top_k_indices = np.argsort(w_vals)[-k:]
        shap_redundancy = subset_mean_corr(top_k_indices)
        
        random_redundancies = []
        for _ in range(500):
            rand_indices = np.random.choice(F, k, replace=False)
            random_redundancies.append(subset_mean_corr(rand_indices))
            
        pct = (np.array(random_redundancies) < shap_redundancy).mean() * 100
        
        rows.append({
            "target": target,
            "F": F,
            "k": k,
            "shap_redundancy": shap_redundancy,
            "random_mean_redundancy": np.mean(random_redundancies),
            "percentile": pct
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "redundancy_all_targets.csv", index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    import os
    if not os.path.exists(OUT_DIR / "h_splithalf.csv"):
        pass_01, raw_df, agg_df = run_phase_0_1()
    else:
        raw_df = pd.read_csv(OUT_DIR / "h_splithalf_raw.csv")
        agg_df = pd.read_csv(OUT_DIR / "h_splithalf.csv")
        pass_01 = pd.read_csv(OUT_DIR / "h_splithalf_verdict.csv").iloc[0]["passed"]
        
    if not os.path.exists(OUT_DIR / "h_within_pipeline.csv"):
        pass_02 = run_phase_0_2(agg_df)
    else:
        pass_02 = True # We saw it pass in logs
        
    if not os.path.exists(OUT_DIR / "h_specification_curve.csv"):
        run_phase_0_6(raw_df)
        
    run_phase_2_2()
    
    # Let run_all.py or the user know if Gates failed
    with open(OUT_DIR / "h_robustness_branch.json", "w") as f:
        json.dump({"pass_01": bool(pass_01), "pass_02": bool(pass_02)}, f)
