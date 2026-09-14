"""
Phase 1.4: Non-SDP Replication.
Checks if the LIME surrogate fit confound holds outside of Software Defect Prediction.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
import pipeline as P
import powered_phase_h as PH

warnings.filterwarnings("ignore")
OUT_DIR = P.OUT_DIR

def run_nonsdp():
    print("=== Phase 1.4: Non-SDP Replication ===")
    
    # Pre-registered prediction
    print("FROZEN PREDICTION: all 3 rhos < 0; median in [-0.8, -0.4].")
    
    datasets = {}
    
    bc = load_breast_cancer()
    datasets["breast_cancer"] = (pd.DataFrame(bc.data, columns=bc.feature_names), pd.Series(bc.target))
    
    dig = load_digits()
    datasets["digits"] = (pd.DataFrame(dig.data, columns=[f"f{i}" for i in range(dig.data.shape[1])]), pd.Series((dig.target >= 5).astype(int)))
    
    wine = load_wine()
    datasets["wine"] = (pd.DataFrame(wine.data, columns=wine.feature_names), pd.Series((wine.target == 0).astype(int)))
    
    rows = []
    
    P.TREE_MODELS |= PH.NEW_TREE_LIKE
    
    for dataset_name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
            
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
                idxs = P.select_xai_instances(proba, n=10)
                
                explainers = [P._make_lime_explainer(X_tr_p, feats, seed) for seed in range(42, 47)]
                
                sigmas, r2s = [], []
                for i in idxs:
                    W, S = [], []
                    for ex in explainers:
                        v, sc = P._lime_vector_and_score(ex, fitted, X_te_p[i], feats)
                        W.append(v)
                        S.append(sc)
                    sigmas.append(np.mean(np.std(W, axis=0)))
                    r2s.append(np.mean(S))
                
                rows.append({"dataset": dataset_name, "model": model_name,
                             "lime_surrogate_r2": float(np.mean(r2s)),
                             "sigma_bar": float(np.mean(sigmas)),
                             "status": "ok"})
            except Exception as e:
                rows.append({"dataset": dataset_name, "model": model_name,
                             "lime_surrogate_r2": np.nan,
                             "sigma_bar": np.nan,
                             "status": f"failed: {type(e).__name__}"})
                
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(OUT_DIR / "h_nonsdp.csv", index=False)
    
    ok = raw_df[raw_df.status == "ok"].dropna(subset=["lime_surrogate_r2", "sigma_bar"])
    
    rhos = []
    for name, g in ok.groupby("dataset"):
        r = spearmanr(g["lime_surrogate_r2"], g["sigma_bar"])
        rhos.append(r.statistic)
        
    all_neg = all(r < 0 for r in rhos)
    med = np.median(rhos)
    med_in_range = -0.8 <= med <= -0.4
    passed = all_neg and med_in_range
    
    pd.DataFrame([{
        "rhos_list": str(rhos),
        "all_negative": all_neg,
        "median_rho": med,
        "median_in_range": med_in_range,
        "passed": passed
    }]).to_csv(OUT_DIR / "h_nonsdp_verdict.csv", index=False)
    
    print("Rhos:", rhos)
    print("Median rho:", med)
    print("Gate:", "PASS" if passed else "FAIL")

if __name__ == "__main__":
    run_nonsdp()
