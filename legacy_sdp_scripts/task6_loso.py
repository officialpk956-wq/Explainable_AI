import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pipeline as P
import cross_ecosystem as XE

warnings.filterwarnings("ignore")

def run_loso():
    print("=== Task 6d: LOSO Weight Stability ===")
    
    apache_w = pd.read_csv(P.OUT_DIR / "xe_shap_weights.csv")
    apache_w_map = dict(apache_w[["Feature", "Weight"]].values)
    
    eclipse_w = pd.read_csv(P.OUT_DIR / "shap_weights.csv")
    eclipse_w_map = dict(eclipse_w[["Feature", "Weight"]].values)
    
    w_maps = {"eclipse": eclipse_w_map, "apache": apache_w_map}

    loso_w_rows = []
    eclipse_projs = ["eclipse", "mylyn", "equinox", "lucene", "pde"]
    apache_projs = ["ant-1.7", "camel-1.6", "xalan-2.6", "poi-3.0", "velocity-1.6"]
    
    vectors = {"eclipse": {}, "apache": {}}
    
    for eco, projs, load_fn in [("eclipse", eclipse_projs, P.load_xy), ("apache", apache_projs, XE.load_promise)]:
        for proj in projs:
            X, y, cols = load_fn(proj)
            w_map = w_maps[eco]
            use_exist = [c for c in cols if c in w_map]
            X = X[use_exist]
            
            X_tr0, _, y_tr0, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=P.SEED)
            tuned = P.tune_hyperparams(X_tr0, y_tr0, f"loso_{proj}")
            
            w_df = pd.DataFrame({"Feature": use_exist, "Weight": P.shap_weights_out_of_fold(X, y, 'RandomForest', tuned)})
            w_vec = dict(zip(w_df["Feature"], w_df["Weight"]))
            
            all_feats = list(w_map.keys())
            vec_arr = np.array([w_vec.get(c, 0) for c in all_feats])
            vectors[eco][proj] = vec_arr
            
            for c in all_feats:
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
    print("Done LOSO.")

if __name__ == "__main__":
    run_loso()
