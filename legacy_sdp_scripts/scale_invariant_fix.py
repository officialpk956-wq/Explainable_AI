"""
Diagnosis and fix for a design flaw in the SHAP-weighting transform.

FLAW. Eq. (3) normalises the SHAP weight vector to sum to 1. The mean weight is
therefore 1/F, so the transform's MAGNITUDE depends on how many features the
schema has. Pipeline B applies log1p AFTER weighting, and log1p is concave, so
shrinking values toward zero compresses the feature space. On Eclipse (F=5,
mean weight 0.20) the compression is mild; on Apache/PROMISE (F=20, mean weight
0.05) it is severe -- which is why the Eclipse result reverses there.

FIX. Normalise to MEAN = 1 instead of SUM = 1:
    w_scaled = w / mean(w)        (equivalently: w * F, given sum(w)=1)
This preserves every relative importance ratio exactly -- the ordering and the
proportions Eq. (2)-(3) produce are untouched -- but removes the dependence on
feature count, so the transform means the same thing in any schema.

Run: python scale_invariant_fix.py
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

OUT_DIR = P.OUT_DIR


def evaluate_variants(load_fn, targets, weight_map, label):
    """original vs sum=1 weighting (published) vs mean=1 weighting (fixed),
    across 20 repeated splits and all 8 classifiers."""
    rows = []
    for name in targets:
        X, y, cols = load_fn(name)
        use = [c for c in cols if c in weight_map]
        X = X[use]
        w_sum1 = np.array([weight_map[c] for c in use])
        w_sum1 = w_sum1 / w_sum1.sum()          # published Eq. (3)
        w_mean1 = w_sum1 / w_sum1.mean()        # proposed fix

        X_tr0, _, y_tr0, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=P.SEED)
        tuned = P.tune_hyperparams(X_tr0, y_tr0, f"fix_{label}_{name}")

        for seed in P.repeated_seeds():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed)
            for model_name in P.make_classifiers():
                def run(Xtr, Xte):
                    clf = P.build_classifier(model_name, tuned, y_train.values)
                    pipe = P.make_pipeline_for(model_name, clf)
                    pipe.fit(Xtr, y_train.values)
                    pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
                    return (f1_score(y_test, pred, average="macro"),
                            roc_auc_score(y_test, proba),
                            average_precision_score(y_test, proba))

                for variant, mult in [("original", None), ("shap_sum1", w_sum1),
                                      ("shap_mean1_fixed", w_mean1)]:
                    Xtr = X_train.values if mult is None else (X_train * mult).values
                    Xte = X_test.values if mult is None else (X_test * mult).values
                    f1, auc, pr = run(Xtr, Xte)
                    rows.append({"ecosystem": label, "dataset": name, "model": model_name,
                                 "variant": variant, "seed": seed,
                                 "f1": f1, "auc": auc, "prauc": pr})
    return rows


def test_fix(df):
    """Paired Wilcoxon + Holm across both fix comparisons on both ecosystems."""
    rows = []
    for eco in df.ecosystem.unique():
        sub_e = df[df.ecosystem == eco]
        for dataset in sub_e.dataset.unique():
            for model in sub_e.model.unique():
                for metric in ["f1", "auc", "prauc"]:
                    s = sub_e[(sub_e.dataset == dataset) & (sub_e.model == model)].sort_values("seed")
                    o = s[s.variant == "original"][metric].values
                    a = s[s.variant == "shap_sum1"][metric].values
                    b = s[s.variant == "shap_mean1_fixed"][metric].values
                    for comp, A, B in [("published_sum1_vs_original", a, o),
                                       ("fixed_mean1_vs_original", b, o),
                                       ("fixed_vs_published", b, a)]:
                        if np.all(A == B):
                            p = 1.0
                        else:
                            try:
                                p = wilcoxon(A, B).pvalue
                            except ValueError:
                                p = 1.0
                        rows.append({"ecosystem": eco, "comparison": comp, "dataset": dataset,
                                     "model": model, "metric": metric, "mean_A": A.mean(),
                                     "mean_B": B.mean(), "mean_diff": (A - B).mean(),
                                     "wilcoxon_p": p})
    res = pd.DataFrame(rows)
    adj, sig = P.holm_bonferroni(res["wilcoxon_p"].values)
    res["holm_corrected_p"] = adj
    res["significant_at_0.05"] = sig
    return res


if __name__ == "__main__":
    print("=== Testing the scale-invariant normalisation fix on BOTH ecosystems ===")

    apache_w = dict(pd.read_csv(OUT_DIR / "xe_shap_weights.csv")[["Feature", "Weight"]].values)
    eclipse_w = dict(pd.read_csv(OUT_DIR / "shap_weights.csv")[["Feature", "Weight"]].values)

    rows = []
    print("\n[Apache/PROMISE, 20 features -- where the published form fails]")
    rows += evaluate_variants(XE.load_promise, XE.XE_TARGETS, apache_w, "apache")
    print("\n[Eclipse, 5 features -- must not break what already worked]")
    rows += evaluate_variants(P.load_xy, P.PHASE2, eclipse_w, "eclipse")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "scale_invariant_scores.csv", index=False)
    res = test_fix(df)
    res.to_csv(OUT_DIR / "scale_invariant_tests.csv", index=False)

    print("\n--- Mean F1 by variant ---")
    print(df.groupby(["ecosystem", "variant"])["f1"].mean().reset_index().to_string(index=False))

    print("\n--- Significant results after Holm correction ---")
    sig = res[res["significant_at_0.05"]]
    print(f"{len(sig)} of {len(res)} comparisons significant.")
    for comp in ["published_sum1_vs_original", "fixed_mean1_vs_original", "fixed_vs_published"]:
        s = sig[sig.comparison == comp]
        up, down = int((s.mean_diff > 0).sum()), int((s.mean_diff < 0).sum())
        print(f"\n  {comp}: {len(s)} significant ({up} improvements, {down} degradations)")
        if len(s):
            print(s[["ecosystem", "dataset", "model", "metric", "mean_diff",
                     "holm_corrected_p"]].to_string(index=False))

    print("\n--- VERDICT ---")
    for eco in df.ecosystem.unique():
        pub = sig[(sig.ecosystem == eco) & (sig.comparison == "published_sum1_vs_original")]
        fix = sig[(sig.ecosystem == eco) & (sig.comparison == "fixed_mean1_vs_original")]
        print(f"  {eco}: published form -> {int((pub.mean_diff < 0).sum())} significant degradations; "
              f"fixed form -> {int((fix.mean_diff < 0).sum())} significant degradations, "
              f"{int((fix.mean_diff > 0).sum())} improvements")
    print("\nDone. Written to", OUT_DIR)
