"""
Phase 4: SHAP-As-Prior.
Hierarchical clustering of features to form group-level inductive priors.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
import pipeline as P
import cross_ecosystem as XE

warnings.filterwarnings("ignore")
OUT_DIR = P.OUT_DIR

def run_shap_prior():
    print("=== Phase 4: SHAP-As-Prior ===")
    
    try:
        apache_w = pd.read_csv(OUT_DIR / "xe_shap_weights.csv")
        w_map = dict(apache_w[["Feature", "Weight"]].values)
    except FileNotFoundError:
        print("xe_shap_weights.csv not found.")
        return
        
    X_src, _, cols = XE.load_promise("ant-1.7") # Just use one source to get feature list
    use = [c for c in cols if c in w_map]
    X_src = X_src[use]
    F = len(use)
    
    # 1. Feature clustering
    corr = X_src.corr().fillna(0).values
    dist = np.clip(1 - np.abs(corr), 0, 1)
    np.fill_diagonal(dist, 0)
    condensed_dist = squareform(dist)
    
    Z = linkage(condensed_dist, method='average')
    groups = fcluster(Z, t=5, criterion='maxclust')
    
    group_map = {}
    for i, c in enumerate(use):
        g = groups[i]
        if g not in group_map:
            group_map[g] = []
        group_map[g].append(c)
        
    # Group importance
    prior = {}
    for g, feats in group_map.items():
        total_w = sum(w_map[f] for f in feats)
        for f in feats:
            prior[f] = total_w / len(feats)
            
    prior_vec = np.array([prior[c] for c in use])
    prior_vec = prior_vec / prior_vec.sum()
    
    uniform_vec = np.full(F, 1.0 / F)
    rng = np.random.RandomState(42)
    shuffled_vec = rng.permutation(prior_vec)
    
    rows = []
    
    targets = XE.XE_TARGETS
    models = ["LogisticRegression", "SVC", "KNN"]  # Linear models respond to L2 scaling
    
    for target in targets:
        X_t, y_t, t_cols = XE.load_promise(target)
        valid = [c for c in use if c in t_cols]
        if len(valid) == 0:
            continue
            
        X_t = X_t[valid]
        v_idx = [use.index(c) for c in valid]
        p_vec = prior_vec[v_idx]
        p_vec = p_vec / p_vec.sum()
        
        u_vec = uniform_vec[v_idx]
        u_vec = u_vec / u_vec.sum()
        
        s_vec = shuffled_vec[v_idx]
        s_vec = s_vec / s_vec.sum()
        
        X_tr = X_src[valid]
        y_tr = XE.load_promise("ant-1.7")[1]
        
        tuned = P.tune_hyperparams(X_tr, y_tr, f"prior_{target}")
        
        for seed in P.repeated_seeds():
            rng = np.random.RandomState(seed)
            for model_name in models:
                def run(w_vec):
                    clf = P.build_classifier(model_name, tuned, y_tr)
                    pipe = P.make_pipeline_for(model_name, clf)
                    pipe.fit(X_tr * w_vec, y_tr)
                    pred = pipe.predict(X_t.values * w_vec)
                    return f1_score(y_t, pred, average="macro")
                    
                rows.append({"target": target, "model": model_name, "seed": seed, "variant": "shap_prior", "f1": run(p_vec)})
                rows.append({"target": target, "model": model_name, "seed": seed, "variant": "uniform", "f1": run(u_vec)})
                rows.append({"target": target, "model": model_name, "seed": seed, "variant": "shuffled", "f1": run(s_vec)})
                
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "shap_prior_scores.csv", index=False)
    
    if df.empty:
        print("Dataframe is empty!")
        return
    
    test_rows = []
    for target in df["target"].unique():
        for model in df.model.unique():
            sub = df[(df.target == target) & (df.model == model)].sort_values("seed")
            p = sub[sub.variant == "shap_prior"]["f1"].values
            u = sub[sub.variant == "uniform"]["f1"].values
            s = sub[sub.variant == "shuffled"]["f1"].values
            
            for comp, A, B in [("prior_vs_uniform", p, u), ("prior_vs_shuffled", p, s)]:
                if np.all(A == B):
                    pval = 1.0
                else:
                    try:
                        pval = wilcoxon(A, B).pvalue
                    except ValueError:
                        pval = 1.0
                test_rows.append({"target": target, "model": model, "comparison": comp,
                                  "mean_diff": (A - B).mean(), "wilcoxon_p": pval})
                                  
    res = pd.DataFrame(test_rows)
    adj, sig = P.holm_bonferroni(res["wilcoxon_p"].values)
    res["holm_corrected_p"] = adj
    res["significant_at_0.05"] = sig
    res.to_csv(OUT_DIR / "shap_prior_tests.csv", index=False)
    
    vs_u = res[res.comparison == "prior_vs_uniform"]
    vs_s = res[res.comparison == "prior_vs_shuffled"]
    
    wins_u = len(vs_u[(vs_u["significant_at_0.05"]) & (vs_u.mean_diff > 0)])
    wins_s = len(vs_s[(vs_s["significant_at_0.05"]) & (vs_s.mean_diff > 0)])
    
    loss_u = len(vs_u[(vs_u["significant_at_0.05"]) & (vs_u.mean_diff < 0)])
    loss_s = len(vs_s[(vs_s["significant_at_0.05"]) & (vs_s.mean_diff < 0)])
    
    passed = (wins_u >= 2) and (wins_s >= 2) and (loss_u == 0) and (loss_s == 0)
    
    pd.DataFrame([{
        "wins_uniform": wins_u, "wins_shuffled": wins_s,
        "loss_uniform": loss_u, "loss_shuffled": loss_s,
        "passed": passed
    }]).to_csv(OUT_DIR / "shap_prior_verdict.csv", index=False)
    
    print(f"Wins vs uniform: {wins_u}, vs shuffled: {wins_s}")
    print(f"Losses vs uniform: {loss_u}, vs shuffled: {loss_s}")
    print("Gate 4:", "PASS" if passed else "FAIL")

if __name__ == "__main__":
    run_shap_prior()
