"""
Powered Phase H: is LIME-measured explanation stability confounded by how well
LIME's linear surrogate happens to fit the model being explained?

The first attempt failed for lack of power: 8 models x 2 datasets. Eclipse gave
rho = -0.667 (p = 0.071) -- a moderate effect that n=8 simply cannot resolve.
This version raises n by two routes:
  * 16 classifiers instead of 8, deliberately spanning the range of
    linear-surrogate friendliness (LDA/SGD at one end, single trees and boosted
    stumps at the other). A correlation test needs spread on the x-axis.
  * 10 datasets instead of 2 (5 Eclipse-family + 5 Apache/PROMISE).
That is 160 (dataset, model) pairs instead of 16.

PRE-REGISTERED ANALYSIS (fixed before the run; do not change it afterwards):
  H1  Better LIME surrogate fit (higher R^2) is associated with lower measured
      instability (sigma_bar) -- i.e. a NEGATIVE correlation.
  Primary test: compute Spearman rho between mean R^2 and sigma_bar separately
      WITHIN each of the 10 datasets, giving 10 independent correlations (models
      within a dataset are not independent, so pooling all 160 rows would
      overstate significance). Then a one-sample Wilcoxon signed-rank test on
      those 10 rho values against 0. Two-sided, alpha = 0.05.
  Secondary (descriptive only, not a significance claim): pooled Spearman over
      all 160 pairs, reported for effect size.
  Decision rule: H PASSES only if the primary test is significant AND the median
      per-dataset rho is negative.

Run: python powered_phase_h.py
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import cross_ecosystem as XE
import pipeline as P

warnings.filterwarnings("ignore")

OUT_DIR = P.OUT_DIR
N_INSTANCES = 10     # per dataset/model
N_SEEDS = 5          # LIME seeds; sigma_bar and R^2 both come from these runs

ECLIPSE_SETS = ["eclipse", "mylyn", "equinox", "lucene", "pde"]
APACHE_SETS = XE.XE_SOURCES + XE.XE_TARGETS

# New classifiers, picked to spread the x-axis (expected linear-surrogate fit):
#   high  -> LDA, SGD-log       (linear decision function)
#   mid   -> GaussianNB, QDA, MLP
#   low   -> DecisionTree, AdaBoost, Bagging (non-smooth, axis-aligned)
NEW_TREE_LIKE = {"DecisionTree", "AdaBoost", "Bagging"}


def extended_classifiers():
    base = P.make_classifiers()
    base.update({
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=P.SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=P.SEED),
        "Bagging": BaggingClassifier(n_estimators=100, random_state=P.SEED),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=P.SEED),
        "GaussianNB": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "SGDLogistic": SGDClassifier(loss="log_loss", class_weight="balanced",
                                     max_iter=2000, random_state=P.SEED),
    })
    return base


def load_any(name):
    return P.load_xy(name) if name in ECLIPSE_SETS else XE.load_promise(name)


def measure(dataset_name):
    """For every classifier: mean LIME surrogate R^2 and sigma_bar, from the
    same set of LIME runs (each run returns both a weight vector and a score)."""
    X, y, cols = load_any(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=P.SEED)
    seeds = [P.SEED + i for i in range(N_SEEDS)]
    feats = list(X_train.columns)
    rows = []

    for model_name, clf_proto in extended_classifiers().items():
        try:
            clf = P.build_classifier(model_name, None, y_train.values) \
                if model_name in P.make_classifiers() else clf_proto
            pipe = P.make_pipeline_for(model_name, clf)
            pipe.fit(X_train.values, y_train.values)
            fitted = pipe.named_steps["clf"]
            X_tr_p = P.transform_only(pipe, X_train.values)
            X_te_p = P.transform_only(pipe, X_test.values)

            proba = fitted.predict_proba(X_te_p)[:, 1]
            idxs = P.select_xai_instances(proba, n=N_INSTANCES)

            explainers = [P._make_lime_explainer(X_tr_p, feats, s) for s in seeds]
            sigmas, r2s = [], []
            for i in idxs:
                W, S = [], []
                for ex in explainers:
                    v, sc = P._lime_vector_and_score(ex, fitted, X_te_p[i], feats)
                    W.append(v)
                    S.append(sc)
                sigmas.append(np.mean(np.std(np.array(W), axis=0)))
                r2s.append(np.mean(S))

            rows.append({"dataset": dataset_name, "model": model_name,
                         "n_features": len(feats),
                         "lime_surrogate_r2": float(np.mean(r2s)),
                         "sigma_bar": float(np.mean(sigmas)), "status": "ok"})
            print(f"    {model_name:<19} R^2={np.mean(r2s):.3f}  sigma_bar={np.mean(sigmas):.5f}")
        except Exception as e:
            rows.append({"dataset": dataset_name, "model": model_name,
                         "n_features": len(feats), "lime_surrogate_r2": np.nan,
                         "sigma_bar": np.nan, "status": f"failed: {type(e).__name__}"})
            print(f"    {model_name:<19} FAILED ({type(e).__name__})")
    return rows


def main():
    # New non-smooth models must use Pipeline A (no scaling), like other trees.
    P.TREE_MODELS |= NEW_TREE_LIKE

    all_sets = ECLIPSE_SETS + APACHE_SETS
    print(f"=== Powered Phase H: {len(extended_classifiers())} classifiers x "
          f"{len(all_sets)} datasets ===")
    rows = []
    for name in all_sets:
        print(f"\n  [{name}]")
        rows += measure(name)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "powered_h_measurements.csv", index=False)
    ok = df[df.status == "ok"].dropna(subset=["lime_surrogate_r2", "sigma_bar"])
    print(f"\n{len(ok)} usable (dataset, model) pairs out of {len(df)}")

    # ---- PRIMARY (pre-registered): per-dataset rho, then Wilcoxon on those ----
    per = []
    for name, g in ok.groupby("dataset"):
        if len(g) < 4 or g["lime_surrogate_r2"].std() == 0 or g["sigma_bar"].std() == 0:
            continue
        r = spearmanr(g["lime_surrogate_r2"], g["sigma_bar"])
        per.append({"dataset": name, "n_models": len(g),
                    "rho": r.statistic, "p_within": r.pvalue})
    per_df = pd.DataFrame(per)
    per_df.to_csv(OUT_DIR / "powered_h_per_dataset.csv", index=False)

    print("\n--- Per-dataset Spearman rho(surrogate R^2, sigma_bar) ---")
    print(per_df.to_string(index=False))

    rhos = per_df["rho"].values
    stat, p_primary = wilcoxon(rhos)
    pooled = spearmanr(ok["lime_surrogate_r2"], ok["sigma_bar"])

    print("\n--- PRIMARY TEST (pre-registered) ---")
    print(f"  One-sample Wilcoxon on {len(rhos)} per-dataset correlations vs 0")
    print(f"  median rho = {np.median(rhos):+.3f}   "
          f"negative in {int((rhos < 0).sum())}/{len(rhos)} datasets")
    print(f"  W = {stat:.1f},  p = {p_primary:.4f}")
    print("\n--- SECONDARY (descriptive) ---")
    print(f"  Pooled Spearman over {len(ok)} pairs: rho = {pooled.statistic:+.3f} "
          f"(p = {pooled.pvalue:.2g}; inflated by clustering, effect size only)")

    passed = bool(p_primary < 0.05 and np.median(rhos) < 0)
    print("\n=== GATE VERDICT ===")
    if passed:
        print("  H PASSES. Explanation stability as measured by LIME is systematically")
        print("  related to surrogate fit across 10 datasets -- model rankings built on")
        print("  sigma_bar are confounded by how linear the model happens to be.")
    else:
        print("  H STILL FAILS under the pre-registered test. Report as a null result;")
        print("  do not re-cut the analysis to chase significance.")

    pd.DataFrame([{"test": "wilcoxon_on_per_dataset_rho", "n_datasets": len(rhos),
                   "median_rho": float(np.median(rhos)), "W": float(stat),
                   "p_value": float(p_primary), "pooled_rho": float(pooled.statistic),
                   "n_pairs": len(ok), "passed": passed}]
                 ).to_csv(OUT_DIR / "powered_h_verdict.csv", index=False)
    return df, per_df, passed


if __name__ == "__main__":
    main()
    print("\nDone. Written to", OUT_DIR)
