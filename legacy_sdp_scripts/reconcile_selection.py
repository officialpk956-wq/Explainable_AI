"""
Task 1: Reconcile Contradiction
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import cross_ecosystem as XE
import pipeline as P

warnings.filterwarnings("ignore")

def select_mrmr(X_train, w_map, k):
    available = list(w_map.keys())
    selected = []
    
    # Calculate correlation matrix on the training data
    corr_matrix = X_train.corr().abs()
    
    for _ in range(k):
        best_score = -np.inf
        best_f = None
        
        for f in available:
            w = w_map[f]
            if len(selected) == 0:
                redundancy = 0
            else:
                redundancy = np.mean([corr_matrix.loc[f, s] for s in selected])
            
            score = w - redundancy # lambda = 1
            if score > best_score:
                best_score = score
                best_f = f
                
        selected.append(best_f)
        available.remove(best_f)
        
    return selected

def run_reconcile():
    print("=== Task 1: Reconcile Selection ===")
    
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
        X, y, cols = load_fn(dataset)
        use = [c for c in cols if c in w_map]
        X = X[use]
        F = len(use)
        k = int(np.ceil(F / 2.0))
        imp_array = np.array([w_map[c] for c in use])
        top_shap_indices = np.argsort(-imp_array)[:k]
        top_shap_names = [use[i] for i in top_shap_indices]
        
        X_train0, _, y_train0, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=P.SEED)
        tuned = P.tune_hyperparams(X_train0, y_train0, f"rec_{dataset}")
        
        for seed in P.repeated_seeds():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed)
                
            rng = np.random.RandomState(seed)
            
            # Select features
            f_mrmr = select_mrmr(X_train, w_map, k)
            
            f_random_list = [list(rng.choice(use, k, replace=False)) for _ in range(3)]
            
            f_shuffled_list = []
            for _ in range(3):
                w_shuffled = dict(zip(use, rng.permutation(list(w_map.values()))))
                f_shuffled_list.append(select_mrmr(X_train, w_shuffled, k))
            
            for model_name in P.make_classifiers():
                def run(feats):
                    Xtr, Xte = X_train[feats].values, X_test[feats].values
                    clf = P.build_classifier(model_name, tuned, y_train.values)
                    pipe = P.make_pipeline_for(model_name, clf)
                    pipe.fit(Xtr, y_train.values)
                    pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
                    return f1_score(y_test, pred, average="macro"), roc_auc_score(y_test, proba), average_precision_score(y_test, proba)
                    
                v_all = run(use)
                v_plain = run(top_shap_names)
                v_mrmr = run(f_mrmr)
                
                rnd_res = [run(feats) for feats in f_random_list]
                v_random = tuple(np.mean(rnd_res, axis=0))
                
                shuf_res = [run(feats) for feats in f_shuffled_list]
                v_shuffled = tuple(np.mean(shuf_res, axis=0))
                
                variants = {
                    "all_features": v_all,
                    "plain_shap_select": v_plain,
                    "mrmr_shap": v_mrmr,
                    "random_select": v_random,
                    "shuffled_mrmr": v_shuffled
                }
                
                for variant, (f1, auc, pr) in variants.items():
                    rows.append({
                        "dataset": dataset,
                        "model": model_name,
                        "seed": seed,
                        "variant": variant,
                        "f1": f1,
                        "auc": auc,
                        "prauc": pr
                    })
                
    df = pd.DataFrame(rows)
    df.to_csv(P.OUT_DIR / "reconcile_selection_scores.csv", index=False)
    
    # Tests
    test_rows = []
    for dataset in df.dataset.unique():
        for model in df.model.unique():
            for metric in ["f1", "auc", "prauc"]:
                sub = df[(df.dataset == dataset) & (df.model == model)].sort_values("seed")
                
                v = {var: sub[sub.variant == var][metric].values for var in ["all_features", "plain_shap_select", "mrmr_shap", "random_select", "shuffled_mrmr"]}
                
                comps = [
                    ("plain_vs_random", v["plain_shap_select"], v["random_select"]),
                    ("mrmr_vs_random", v["mrmr_shap"], v["random_select"]),
                    ("mrmr_vs_shuffled", v["mrmr_shap"], v["shuffled_mrmr"]),
                    ("plain_vs_all", v["plain_shap_select"], v["all_features"]),
                    ("mrmr_vs_all", v["mrmr_shap"], v["all_features"])
                ]
                
                for comp, A, B in comps:
                    if np.all(A == B):
                        p = 1.0
                    else:
                        try:
                            p = wilcoxon(A, B).pvalue
                        except ValueError:
                            p = 1.0
                            
                    test_rows.append({
                        "dataset": dataset, "model": model, "metric": metric, "comparison": comp,
                        "mean_diff": (A - B).mean(),
                        "wilcoxon_p": p
                    })
                                      
    res = pd.DataFrame(test_rows)
    adj, sig = P.holm_bonferroni(res["wilcoxon_p"].values)
    res["holm_corrected_p"] = adj
    res["significant_at_0.05"] = sig
    res.to_csv(P.OUT_DIR / "reconcile_selection_tests.csv", index=False)
    
    # Verdict
    mrmr_vs_random = res[res.comparison == "mrmr_vs_random"]
    plain_vs_random = res[res.comparison == "plain_vs_random"]
    mrmr_vs_shuffled = res[res.comparison == "mrmr_vs_shuffled"]
    
    mrmr_losses = len(mrmr_vs_random[(mrmr_vs_random["significant_at_0.05"]) & (mrmr_vs_random.mean_diff < 0)])
    mrmr_wins_random = len(mrmr_vs_random[(mrmr_vs_random["significant_at_0.05"]) & (mrmr_vs_random.mean_diff > 0)])
    
    plain_losses = len(plain_vs_random[(plain_vs_random["significant_at_0.05"]) & (plain_vs_random.mean_diff < 0)])
    mrmr_wins_shuffled = len(mrmr_vs_shuffled[(mrmr_vs_shuffled["significant_at_0.05"]) & (mrmr_vs_shuffled.mean_diff > 0)])
    
    R1 = (mrmr_losses == 0) and (plain_losses >= 1)
    R2 = R1 and (mrmr_wins_random >= 1) and (mrmr_wins_shuffled >= 1)
    
    run_a_corroborated = plain_losses > 0  # Run A found plain_shap loses to random
    root_cause = "Run B (mrmr_shap.py) mistakenly compared random_k against all_features instead of plain_shap_select, falsely calling it plain_vs_random. It also used 1 draw instead of 3 and only evaluated f1."
    
    pd.DataFrame([{
        "run_a_corroborated": run_a_corroborated,
        "root_cause": root_cause,
        "R1": R1,
        "R2": R2,
        "plain_losses": plain_losses,
        "mrmr_losses": mrmr_losses,
        "mrmr_wins_random": mrmr_wins_random,
        "mrmr_wins_shuffled": mrmr_wins_shuffled
    }]).to_csv(P.OUT_DIR / "reconcile_verdict.csv", index=False)
    
    print("Done Task 1.")

if __name__ == "__main__":
    run_reconcile()
