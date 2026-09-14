"""
Phase 0.3: TOST Equivalence Tests.
Tests if real SHAP weighting is statistically equivalent to magnitude-matched controls.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import pipeline as P

warnings.filterwarnings("ignore")
OUT_DIR = P.OUT_DIR

def run_tost():
    print("=== Phase 0.3: TOST Equivalence ===")
    
    # Needs ablation_scores.csv from ablation_control.py
    try:
        df = pd.read_csv(OUT_DIR / "ablation_scores.csv")
    except FileNotFoundError:
        print("ablation_scores.csv not found. Please run ablation_control.py first.")
        return
        
    rows = []
    comparisons = [
        ("shap_sum1", "uniform_sum1"),
        ("shap_sum1", "shuffled_shap")
    ]
    
    for eco in df.ecosystem.unique():
        e = df[df.ecosystem == eco]
        for dataset in e.dataset.unique():
            for model in e.model.unique():
                for metric in ["f1", "auc", "prauc"]:
                    sub = e[(e.dataset == dataset) & (e.model == model)].sort_values("seed")
                    
                    for A_name, B_name in comparisons:
                        comp_name = f"{A_name}_vs_{B_name}"
                        A = sub[sub.variant == A_name][metric].values
                        B = sub[sub.variant == B_name][metric].values
                        diff = A - B
                        
                        for delta in [0.01, 0.005]:
                            if np.all(diff == 0):
                                p_lower, p_upper = 0.0, 0.0
                                trivial = True
                            else:
                                trivial = False
                                try:
                                    _, p_lower = wilcoxon(diff + delta, alternative='greater')
                                    _, p_upper = wilcoxon(diff - delta, alternative='less')
                                except ValueError:
                                    p_lower, p_upper = 1.0, 1.0
                                    
                            rows.append({
                                "ecosystem": eco,
                                "dataset": dataset,
                                "model": model,
                                "metric": metric,
                                "comparison": comp_name,
                                "delta": delta,
                                "p_lower": p_lower,
                                "p_upper": p_upper,
                                "equivalent_raw": (p_lower < 0.05 and p_upper < 0.05) or trivial,
                                "trivial_all_zeros": trivial
                            })
                            
    res = pd.DataFrame(rows)
    
    # Apply Holm-Bonferroni correction. We correct over the 288 hypotheses per delta (2 comparisons x 6 datasets x 8 models x 3 metrics)
    final_rows = []
    for delta, g in res.groupby("delta"):
        p_max = np.maximum(g["p_lower"], g["p_upper"])
        adj_p, sig = P.holm_bonferroni(p_max.values, alpha=0.05)
        g = g.copy()
        g["equivalent_holm"] = sig | g["trivial_all_zeros"]
        g["adj_p_max"] = adj_p
        final_rows.append(g)
        
    final_df = pd.concat(final_rows)
    final_df.to_csv(OUT_DIR / "ablation_equivalence.csv", index=False)
    
    print("\n--- Equivalence Results ---")
    for delta in [0.01, 0.005]:
        sub = final_df[final_df.delta == delta]
        raw_eq = sub.equivalent_raw.sum()
        holm_eq = sub.equivalent_holm.sum()
        total = len(sub)
        print(f"Delta = {delta}: {raw_eq}/{total} equivalent (raw), {holm_eq}/{total} equivalent (Holm-corrected)")
        
if __name__ == "__main__":
    run_tost()
