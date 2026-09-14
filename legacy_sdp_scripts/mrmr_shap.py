"""
Phase 2.3: mRMR-SHAP Feature Selection.
Tests if SHAP weights carry transferable signal when redundancy is penalized.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from scipy.stats import wilcoxon
import pipeline as P
import cross_ecosystem as XE

warnings.filterwarnings("ignore")
OUT_DIR = P.OUT_DIR

def select_mrmr(X_train, w_map, k):
    available = list(w_map.keys())
    selected = []
    
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
            
            score = w - redundancy
            if score > best_score:
                best_score = score
                best_f = f
                
        selected.append(best_f)
        available.remove(best_f)
        
    return selected

def run_mrmr():
    print("=== Phase 2.3: mRMR-SHAP ===")
    
    try:
        apache_w = pd.read_csv(OUT_DIR / "xe_shap_weights.csv")
        apache_w_map = dict(apache_w[["Feature", "Weight"]].values)
        
        eclipse_w = pd.read_csv(OUT_DIR / "shap_weights.csv")
        eclipse_w_map = dict(eclipse_w[["Feature", "Weight"]].values)
    except FileNotFoundError:
        print("SHAP weights not found. Run earlier phases.")
        return

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
        
        X_train0, _, y_train0, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=P.SEED)
        tuned = P.tune_hyperparams(X_train0, y_train0, f"mrmr_{dataset}")
        
        for seed in P.repeated_seeds():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed)
                
            rng = np.random.RandomState(seed)
            
            # Select features
            f_mrmr = select_mrmr(X_train, w_map, k)
            f_random = list(rng.choice(use, k, replace=False))
            
            w_shuffled = dict(zip(use, rng.permutation(list(w_map.values()))))
            f_mrmr_shuffled = select_mrmr(X_train, w_shuffled, k)
            
            for model_name in P.make_classifiers():
                def run(feats):
                    Xtr, Xte = X_train[feats].values, X_test[feats].values
                    clf = P.build_classifier(model_name, tuned, y_train.values)
                    pipe = P.make_pipeline_for(model_name, clf)
                    pipe.fit(Xtr, y_train.values)
                    pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
                    return f1_score(y_test, pred, average="macro")
                    
                rows.append({"ecosystem": eco, "dataset": dataset, "model": model_name, "seed": seed, "variant": "original", "f1": run(use)})
                rows.append({"ecosystem": eco, "dataset": dataset, "model": model_name, "seed": seed, "variant": "random_k", "f1": run(f_random)})
                rows.append({"ecosystem": eco, "dataset": dataset, "model": model_name, "seed": seed, "variant": "mrmr_shap", "f1": run(f_mrmr)})
                rows.append({"ecosystem": eco, "dataset": dataset, "model": model_name, "seed": seed, "variant": "mrmr_shuffled", "f1": run(f_mrmr_shuffled)})
                
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "mrmr_scores.csv", index=False)
    
    # Tests
    test_rows = []
    for eco in df.ecosystem.unique():
        e = df[df.ecosystem == eco]
        for dataset in e.dataset.unique():
            for model in e.model.unique():
                sub = e[(e.dataset == dataset) & (e.model == model)].sort_values("seed")
                
                v = {var: sub[sub.variant == var]["f1"].values for var in ["original", "random_k", "mrmr_shap", "mrmr_shuffled"]}
                
                for comp, A, B in [("mrmr_vs_random", v["mrmr_shap"], v["random_k"]),
                                   ("plain_vs_random", v["original"], v["random_k"]),
                                   ("mrmr_vs_shuffled_mrmr", v["mrmr_shap"], v["mrmr_shuffled"])]:
                    if np.all(A == B):
                        p = 1.0
                    else:
                        try:
                            p = wilcoxon(A, B).pvalue
                        except ValueError:
                            p = 1.0
                            
                    test_rows.append({"ecosystem": eco, "dataset": dataset, "model": model, "comparison": comp,
                                      "mean_A": A.mean(), "mean_B": B.mean(), "mean_diff": (A - B).mean(),
                                      "wilcoxon_p": p})
                                      
    res = pd.DataFrame(test_rows)
    adj, sig = P.holm_bonferroni(res["wilcoxon_p"].values)
    res["holm_corrected_p"] = adj
    res["significant_at_0.05"] = sig
    res.to_csv(OUT_DIR / "mrmr_tests.csv", index=False)
    
    mrmr_vs_random = res[res.comparison == "mrmr_vs_random"]
    plain_vs_random = res[res.comparison == "plain_vs_random"]
    mrmr_vs_shuffled = res[res.comparison == "mrmr_vs_shuffled_mrmr"]
    
    mrmr_losses = len(mrmr_vs_random[(mrmr_vs_random["significant_at_0.05"]) & (mrmr_vs_random.mean_diff < 0)])
    mrmr_wins = len(mrmr_vs_random[(mrmr_vs_random["significant_at_0.05"]) & (mrmr_vs_random.mean_diff > 0)])
    
    plain_losses = len(plain_vs_random[(plain_vs_random["significant_at_0.05"]) & (plain_vs_random.mean_diff < 0)])
    
    shuf_wins = len(mrmr_vs_shuffled[(mrmr_vs_shuffled["significant_at_0.05"]) & (mrmr_vs_shuffled.mean_diff > 0)])
    
    R1 = (mrmr_losses == 0) and (plain_losses >= 1)
    R2 = R1 and (mrmr_wins >= 1) and (shuf_wins >= 1)
    
    pd.DataFrame([{
        "mrmr_losses": mrmr_losses, "mrmr_wins": mrmr_wins,
        "plain_losses": plain_losses, "shuf_wins": shuf_wins,
        "R1": R1, "R2": R2
    }]).to_csv(OUT_DIR / "mrmr_verdict.csv", index=False)
    
    print(f"mrmr_losses={mrmr_losses}, mrmr_wins={mrmr_wins}, plain_losses={plain_losses}, shuf_wins={shuf_wins}")
    print(f"R1={R1}, R2={R2}")

if __name__ == "__main__":
    run_mrmr()
