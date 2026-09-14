"""
Task 3: Stability Convergence
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import pipeline as P
import cross_ecosystem as XE

warnings.filterwarnings("ignore")

def run_task_3():
    print("=== Task 3: Stability Convergence ===")
    datasets = ["eclipse", "ant-1.7"]
    models = ["LogisticRegression", "SVC", "RandomForest", "XGBoost"]
    budgets = [500, 1000, 2500, 5000, 10000]
    lime_seeds = list(range(42, 47))
    
    rows = []
    
    for dataset in datasets:
        print(f"Dataset: {dataset}")
        if dataset in ["ant-1.7"]:
            X, y, cols = XE.load_promise(dataset)
            X = X[cols]
        else:
            X, y, cols = P.load_xy(dataset)
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=P.SEED)
            
        tuned = P.tune_hyperparams(X_train, y_train, f"task3_{dataset}")
        
        for model_name in models:
            clf = P.build_classifier(model_name, tuned, y_train.values)
            pipe = P.make_pipeline_for(model_name, clf)
            pipe.fit(X_train.values, y_train.values)
            clf_fitted = pipe.named_steps["clf"]
            
            proba = pipe.predict_proba(X_test.values)[:, 1]
            inst_indices = P.select_xai_instances(proba)
            
            X_tr_trans = P.transform_only(pipe, X_train.values)
            X_te_trans = P.transform_only(pipe, X_test.values)
            feats = list(X.columns)
            
            for budget in budgets:
                inst_sigma_bars = []
                for idx in inst_indices:
                    x_inst = X_te_trans[idx]
                    
                    weights = []
                    for seed in lime_seeds:
                        explainer = P._make_lime_explainer(X_tr_trans, feats, seed)
                        clf = clf_fitted
                        def predict_proba(X):
                            return clf.predict_proba(X)
                        exp = explainer.explain_instance(x_inst, predict_proba, num_features=len(feats), num_samples=budget)
                        w = [0] * len(feats)
                        for fname, weight in exp.as_list():
                            if fname in feats:
                                w[feats.index(fname)] = weight
                        weights.append(w)
                        
                    weights = np.array(weights) # (5, F)
                    stds = np.std(weights, axis=0, ddof=1)
                    inst_sigma_bars.append(np.mean(stds))
                    
                sigma_bar = np.mean(inst_sigma_bars)
                rows.append({
                    "dataset": dataset,
                    "model": model_name,
                    "num_samples": budget,
                    "sigma_bar": sigma_bar
                })
                
    df = pd.DataFrame(rows)
    df.to_csv(P.OUT_DIR / "stability_convergence.csv", index=False)
    
    # rank swaps
    swaps = []
    for dataset in datasets:
        for b_a in budgets:
            for b_b in budgets:
                if b_a >= b_b:
                    continue
                
                sub_a = df[(df.dataset == dataset) & (df.num_samples == b_a)]
                sub_b = df[(df.dataset == dataset) & (df.num_samples == b_b)]
                
                # rank models
                rank_a = sub_a.set_index("model")["sigma_bar"].rank()
                rank_b = sub_b.set_index("model")["sigma_bar"].rank()
                
                for i in range(len(models)):
                    for j in range(i+1, len(models)):
                        m_i = models[i]
                        m_j = models[j]
                        if m_i not in rank_a or m_j not in rank_a or m_i not in rank_b or m_j not in rank_b:
                            continue
                            
                        diff_a = rank_a[m_i] - rank_a[m_j]
                        diff_b = rank_b[m_i] - rank_b[m_j]
                        
                        if diff_a * diff_b < 0: # swap occurred
                            swaps.append({
                                "dataset": dataset,
                                "budget_a": b_a,
                                "budget_b": b_b,
                                "model_i": m_i,
                                "model_j": m_j
                            })
                            
    pd.DataFrame(swaps).to_csv(P.OUT_DIR / "convergence_rank_swaps.csv", index=False)
    print("Done Task 3.")

if __name__ == "__main__":
    run_task_3()
