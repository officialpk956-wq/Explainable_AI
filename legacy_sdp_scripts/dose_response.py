"""
Task 5: Dose-Response
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pipeline as P
import cross_ecosystem as XE

warnings.filterwarnings("ignore")

def run_task_5():
    print("=== Task 5: Dose-Response ===")
    
    apache_w = pd.read_csv(P.OUT_DIR / "xe_shap_weights.csv")
    apache_w_map = dict(apache_w[["Feature", "Weight"]].values)
    
    eclipse_w = pd.read_csv(P.OUT_DIR / "shap_weights.csv")
    eclipse_w_map = dict(eclipse_w[["Feature", "Weight"]].values)

    targets = [
        ("eclipse", "pde", P.load_xy, eclipse_w_map, 1.0 / 5.0),
        ("apache", "xalan-2.6", XE.load_promise, apache_w_map, 1.0 / 20.0)
    ]
    
    models = ["LogisticRegression", "SVC", "RandomForest"]
    c_values = np.logspace(-3, 1, 15)
    
    rows = []
    
    for eco, dataset, load_fn, w_map, uniform_val in targets:
        print(f"Dataset: {dataset}")
        X, y, cols = load_fn(dataset)
        use = [c for c in cols if c in w_map]
        X = X[use]
        F = len(use)
        
        # calculate shap sum to 1 weight vector
        imp_array = np.array([w_map[c] for c in use])
        shap_weight_vec = imp_array / np.sum(imp_array)
        
        X_train0, _, y_train0, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=P.SEED)
        tuned = P.tune_hyperparams(X_train0, y_train0, f"dose_{dataset}")
        
        for seed in P.repeated_seeds():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed)
                
            for model_name in models:
                clf = P.build_classifier(model_name, tuned, y_train.values)
                pipe = P.make_pipeline_for(model_name, clf)
                
                def eval_variant(multiplier, variant_name, c_val=None):
                    X_tr_mod = X_train.copy()
                    X_te_mod = X_test.copy()
                    
                    if multiplier is not None:
                        for i, col in enumerate(use):
                            mult = multiplier[i] if isinstance(multiplier, (list, np.ndarray)) else multiplier
                            X_tr_mod[col] = X_tr_mod[col] * mult
                            X_te_mod[col] = X_te_mod[col] * mult
                            
                    pipe.fit(X_tr_mod.values, y_train.values)
                    pred = pipe.predict(X_te_mod.values)
                    proba = pipe.predict_proba(X_te_mod.values)[:, 1]
                    f1 = f1_score(y_test, pred, average="macro")
                    auc = roc_auc_score(y_test, proba)
                    
                    rows.append({
                        "dataset": dataset,
                        "model": model_name,
                        "seed": seed,
                        "variant": variant_name,
                        "c": c_val if c_val is not None else np.nan,
                        "f1": f1,
                        "auc": auc
                    })
                    
                eval_variant(None, "original")
                eval_variant(shap_weight_vec, "shap_weighted")
                for c in c_values:
                    eval_variant(c, "c_multiplier", c)
                    
    df = pd.DataFrame(rows)
    df.to_csv(P.OUT_DIR / "dose_response_raw.csv", index=False)
    
    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, (dataset, F_uniform) in enumerate([("pde", 1.0/5.0), ("xalan-2.6", 1.0/20.0)]):
        for j, model in enumerate(models):
            ax = axes[i, j]
            sub = df[(df.dataset == dataset) & (df.model == model)]
            
            c_data = sub[sub.variant == "c_multiplier"].groupby("c")["f1"].mean()
            ax.plot(np.log10(c_data.index), c_data.values, marker='o', label="multiplier c")
            
            shap_mean = sub[sub.variant == "shap_weighted"]["f1"].mean()
            orig_mean = sub[sub.variant == "original"]["f1"].mean()
            
            ax.axhline(shap_mean, color='r', linestyle='-', label="shap_weighted")
            ax.axhline(orig_mean, color='g', linestyle='-.', label="original")
            ax.axvline(np.log10(F_uniform), color='k', linestyle='--', label=f"c = 1/F ({F_uniform})")
            
            ax.set_title(f"{dataset} - {model}")
            ax.set_xlabel("log10(c)")
            ax.set_ylabel("F1 Macro")
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(P.OUT_DIR / "dose_response.png")
    
    print("Done Task 5.")

if __name__ == "__main__":
    run_task_5()
