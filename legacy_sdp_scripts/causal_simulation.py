"""
Phase 0.5: Causal Simulation.
Intervene on model smoothness (tree depth, SVM gamma) with data held fixed.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pipeline as P

warnings.filterwarnings("ignore")
OUT_DIR = P.OUT_DIR

def run_simulation():
    print("=== Phase 0.5: Causal Simulation ===")
    
    replicates = [42, 43, 44]
    depths = [1, 2, 3, 5, 8, 12, 20]
    gammas = np.logspace(-3, 2, 8)
    
    rows = []
    
    for s in replicates:
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=s)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        y = pd.Series(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
            
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        # Family 1: DecisionTree
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, random_state=42)
            clf.fit(X_train.values, y_train.values)
            
            proba = clf.predict_proba(X_test.values)[:, 1]
            idxs = P.select_xai_instances(proba, n=10)
            
            feats = list(X_train.columns)
            explainers = [P._make_lime_explainer(X_train.values, feats, seed) for seed in range(42, 47)]
            
            for i_idx, i in enumerate(idxs):
                W, S = [], []
                for ex in explainers:
                    v, sc = P._lime_vector_and_score(ex, clf, X_test.values[i], feats)
                    W.append(v)
                    S.append(sc)
                rows.append({
                    "replicate": s, "family": "DecisionTree", "config_val": float(d),
                    "instance": i_idx, "r2": float(np.mean(S)),
                    "sigma_bar": float(np.mean(np.std(W, axis=0)))
                })
                
        # Family 2: SVC
        for g in gammas:
            clf = SVC(kernel='rbf', gamma=g, probability=True, random_state=42)
            clf.fit(X_train_scaled.values, y_train.values)
            
            proba = clf.predict_proba(X_test_scaled.values)[:, 1]
            idxs = P.select_xai_instances(proba, n=10)
            
            feats = list(X_train_scaled.columns)
            explainers = [P._make_lime_explainer(X_train_scaled.values, feats, seed) for seed in range(42, 47)]
            
            for i_idx, i in enumerate(idxs):
                W, S = [], []
                for ex in explainers:
                    v, sc = P._lime_vector_and_score(ex, clf, X_test_scaled.values[i], feats)
                    W.append(v)
                    S.append(sc)
                rows.append({
                    "replicate": s, "family": "SVC", "config_val": float(g),
                    "instance": i_idx, "r2": float(np.mean(S)),
                    "sigma_bar": float(np.mean(np.std(W, axis=0)))
                })
                
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(OUT_DIR / "causal_simulation_raw.csv", index=False)
    
    agg = raw_df.groupby(["replicate", "family", "config_val"]).agg({
        "r2": "mean", "sigma_bar": "mean"
    }).reset_index()
    agg.to_csv(OUT_DIR / "causal_simulation.csv", index=False)
    
    # Gate Evaluation
    dt_agg = agg[agg.family == "DecisionTree"]
    svc_agg = agg[agg.family == "SVC"]
    
    p1_dt = spearmanr(dt_agg.config_val.rank(), dt_agg.r2)
    p1_svc = spearmanr(svc_agg.config_val.rank(), svc_agg.r2)
    
    p2_dt = spearmanr(dt_agg.r2, dt_agg.sigma_bar)
    p2_svc = spearmanr(svc_agg.r2, svc_agg.sigma_bar)
    
    passed_p1 = (p1_dt.statistic < -0.7) and (p1_svc.statistic < -0.7)
    passed_p2 = (p2_dt.statistic < 0 and p2_dt.pvalue < 0.05) and (p2_svc.statistic < 0 and p2_svc.pvalue < 0.05)
    
    passed = passed_p1 and passed_p2
    
    pd.DataFrame([{
        "p1_dt_rho": p1_dt.statistic, "p1_dt_p": p1_dt.pvalue,
        "p1_svc_rho": p1_svc.statistic, "p1_svc_p": p1_svc.pvalue,
        "p2_dt_rho": p2_dt.statistic, "p2_dt_p": p2_dt.pvalue,
        "p2_svc_rho": p2_svc.statistic, "p2_svc_p": p2_svc.pvalue,
        "passed": passed
    }]).to_csv(OUT_DIR / "causal_simulation_verdict.csv", index=False)
    
    print("P1 DecisionTree:", p1_dt)
    print("P1 SVC:", p1_svc)
    print("P2 DecisionTree:", p2_dt)
    print("P2 SVC:", p2_svc)
    print("Gate:", "PASS" if passed else "FAIL")
    
if __name__ == "__main__":
    run_simulation()
