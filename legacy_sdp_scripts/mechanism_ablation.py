"""
Direct causal test of the proposed mechanism: remove log1p and see if the effect dies.

THE CLAIM WE ARE TESTING (our own). The apparent benefit of SHAP feature weighting is
not explanation transfer but an interaction between two pipeline choices:
  (a) Eq-3 style normalisation makes every weight approximately 1/F, and
  (b) Pipeline B applies log1p AFTER the weighting.
Because log1p is concave, multiplying by a small constant moves values into its
near-linear region and changes the geometry of the feature space.

WHY REMOVING log1p IS DECISIVE. RobustScaler computes (x - median)/IQR. For a strictly
positive per-column constant c:
      (cx - median(cx)) / IQR(cx) = (cx - c*median(x)) / (c*IQR(x)) = (x - median(x))/IQR(x)
so a pure per-column rescaling is EXACTLY undone by RobustScaler. It is only the
intervening concave transform that lets the multiplier survive into the model.

The mechanism therefore makes a sharp, falsifiable prediction:

  P_A  WITHOUT log1p, SHAP weighting must have essentially NO effect -- the weighted and
       unweighted variants should agree to numerical tolerance for every Pipeline-B model.
  P_B  WITH log1p, the effect must reappear, reproducing the earlier ablation.

If P_A fails -- if weighting still changes results with log1p removed -- our stated
mechanism is wrong or incomplete, and we must say so.

SCOPE. Only Pipeline-B models are relevant: tree models already use Pipeline A (identity),
so they have no log1p to remove and are provably scale-invariant regardless.

PRE-REGISTERED THRESHOLDS (frozen before running)
  P_A holds iff, with log1p removed, the median |mean_diff| across cells is < 1e-6 AND
      zero cells reach Holm-corrected significance.
  P_B holds iff, with log1p present, at least one cell reaches Holm-corrected
      significance (i.e. the earlier finding reproduces in this harness).

Run: python mechanism_ablation.py
"""
import warnings

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, RobustScaler

import cross_ecosystem as XE
import pipeline as P

warnings.filterwarnings("ignore")

OUT_DIR = P.OUT_DIR
PIPE_MODELS = ["LogisticRegression", "SVC", "KNN"]      # the Pipeline-B models
ARMS = ["original", "shap_weighted", "uniform"]
ECLIPSE_T = ["equinox", "lucene", "pde"]
APACHE_T = XE.XE_TARGETS


def load_any(name):
    return P.load_xy(name) if name in ECLIPSE_T else XE.load_promise(name)


def build_pipeline(model_name, clf, use_log1p):
    """Pipeline B, with the log1p step optionally removed. Everything else identical."""
    steps = []
    if use_log1p:
        steps.append(("log1p", FunctionTransformer(lambda X: np.log1p(np.abs(X)))))
    steps += [("scaler", RobustScaler()),
              ("smote", P.ConditionalSMOTE()),
              ("clf", clf)]
    return ImbPipeline(steps)


def run():
    print("=== Mechanism ablation: does removing log1p kill the effect? ===")
    w_eclipse = dict(pd.read_csv(OUT_DIR / "shap_weights.csv")[["Feature", "Weight"]].values)
    w_apache = dict(pd.read_csv(OUT_DIR / "xe_shap_weights.csv")[["Feature", "Weight"]].values)
    rows = []

    for target in ECLIPSE_T + APACHE_T:
        wmap = w_eclipse if target in ECLIPSE_T else w_apache
        X, y, cols = load_any(target)
        use = [c for c in cols if c in wmap]
        X = X[use]
        F = len(use)
        w = np.array([wmap[c] for c in use]); w = w / w.sum()
        u = np.full(F, 1.0 / F)

        X_tr0, _, y_tr0, _ = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=P.SEED)
        tuned = P.tune_hyperparams(X_tr0, y_tr0, f"mech_{target}")
        print(f"  [{target}] F={F}")

        for seed in P.repeated_seeds():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed)
            for model_name in PIPE_MODELS:
                for use_log1p in [True, False]:
                    for arm in ARMS:
                        mult = {"original": None, "shap_weighted": w, "uniform": u}[arm]
                        Xtr = X_train.values if mult is None else (X_train * mult).values
                        Xte = X_test.values if mult is None else (X_test * mult).values
                        clf = P.build_classifier(model_name, tuned, y_train.values)
                        pipe = build_pipeline(model_name, clf, use_log1p)
                        pipe.fit(Xtr, y_train.values)
                        pred = pipe.predict(Xte)
                        proba = pipe.predict_proba(Xte)[:, 1]
                        rows.append({
                            "target": target, "model": model_name,
                            "pipeline": "with_log1p" if use_log1p else "no_log1p",
                            "arm": arm, "seed": seed,
                            "f1": f1_score(y_test, pred, average="macro"),
                            "auc": roc_auc_score(y_test, proba),
                            "prauc": average_precision_score(y_test, proba)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "mechanism_ablation_scores.csv", index=False)
    print(f"\nrows={len(df)} (expected {6*3*2*3*20})")

    # ---- tests: shap_weighted vs original, within each pipeline variant ----
    test_rows = []
    for (target, model, pl), g in df.groupby(["target", "model", "pipeline"]):
        g = g.sort_values("seed")
        for metric in ["f1", "auc", "prauc"]:
            a = g[g.arm == "shap_weighted"][metric].values
            b = g[g.arm == "original"][metric].values
            if np.all(a == b):
                p = 1.0
            else:
                try:
                    p = wilcoxon(a, b).pvalue
                except ValueError:
                    p = 1.0
            test_rows.append({"target": target, "model": model, "pipeline": pl,
                              "metric": metric, "mean_diff": (a - b).mean(),
                              "max_abs_diff": np.abs(a - b).max(), "wilcoxon_p": p})
    t = pd.DataFrame(test_rows)
    adj, sig = P.holm_bonferroni(t["wilcoxon_p"].values)
    t["holm_corrected_p"] = adj
    t["significant_at_0.05"] = sig
    t.to_csv(OUT_DIR / "mechanism_ablation_tests.csv", index=False)

    print("\n--- Effect of SHAP weighting, by pipeline variant ---")
    summ = t.groupby("pipeline").agg(
        median_abs_mean_diff=("mean_diff", lambda x: np.median(np.abs(x))),
        max_abs_diff=("max_abs_diff", "max"),
        n_significant=("significant_at_0.05", "sum"), n_cells=("mean_diff", "size"))
    print(summ.to_string())

    no = t[t.pipeline == "no_log1p"]
    wi = t[t.pipeline == "with_log1p"]
    pa = bool(np.median(np.abs(no.mean_diff)) < 1e-6 and no["significant_at_0.05"].sum() == 0)
    pb = bool(wi["significant_at_0.05"].sum() >= 1)

    print("\n--- PRE-REGISTERED ---")
    print(f"  P_A effect vanishes without log1p: median |diff|={np.median(np.abs(no.mean_diff)):.2e}, "
          f"{int(no['significant_at_0.05'].sum())} significant -> {'SUPPORTED' if pa else 'NOT supported'}")
    print(f"  P_B effect present with log1p:     median |diff|={np.median(np.abs(wi.mean_diff)):.2e}, "
          f"{int(wi['significant_at_0.05'].sum())} significant -> {'SUPPORTED' if pb else 'NOT supported'}")
    print("\n=== MECHANISM VERDICT ===")
    if pa and pb:
        print("  CAUSALLY DEMONSTRATED. The effect exists only when log1p follows the")
        print("  weighting. Remove the concave transform and RobustScaler absorbs the")
        print("  multiplier exactly, as the algebra predicts.")
    elif not pa:
        print("  MECHANISM WRONG OR INCOMPLETE: weighting still changes results with")
        print("  log1p removed. The stated explanation does not account for the effect.")
    else:
        print("  INCONCLUSIVE: the effect did not reproduce in the with-log1p arm here.")

    if wi["significant_at_0.05"].any():
        print("\n  Significant cells (with_log1p):")
        print(wi[wi["significant_at_0.05"]][
            ["target", "model", "metric", "mean_diff", "holm_corrected_p"]].to_string(index=False))

    pd.DataFrame([{"P_A_supported": pa, "P_B_supported": pb,
                   "no_log1p_median_abs_diff": float(np.median(np.abs(no.mean_diff))),
                   "no_log1p_max_abs_diff": float(no.max_abs_diff.max()),
                   "no_log1p_n_significant": int(no["significant_at_0.05"].sum()),
                   "with_log1p_median_abs_diff": float(np.median(np.abs(wi.mean_diff))),
                   "with_log1p_n_significant": int(wi["significant_at_0.05"].sum()),
                   "mechanism_demonstrated": bool(pa and pb)}]).to_csv(
        OUT_DIR / "mechanism_ablation_verdict.csv", index=False)
    return df, t


if __name__ == "__main__":
    run()
    print("\nDone. Written to", OUT_DIR)
