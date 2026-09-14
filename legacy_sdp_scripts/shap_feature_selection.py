"""
The honest attempt at a positive Phase K: SHAP-guided feature SELECTION.

WHY THIS MIGHT DIFFER FROM THE FALSIFIED METHOD. Multiplicative weighting is a
monotone per-column rescaling: tree splits are invariant to it, and for the
scaled models RobustScaler largely undoes it -- which is why the ablation found
real SHAP weights indistinguishable from uniform and shuffled ones. Dropping
features is NOT monotone rescaling. It changes the hypothesis space itself, so
trees cannot ignore it and no scaler can undo it. That makes it the one
operationalisation the earlier ablation does not already rule out.

WHAT WOULD COUNT AS SUCCESS. Not "selection beats using everything" -- feature
selection often helps for reasons that have nothing to do with explanations
(less noise, less collinearity, lower variance). The claim only survives if
SHAP-guided selection beats selection that uses NO importance information:

    shap_select      keep the top-k features by source-derived SHAP importance
    random_select    keep k features chosen at random          (3 draws, averaged)
    shuffled_select  keep top-k under a random permutation of the same
                     importance values                          (3 draws, averaged)

Importances come only from the SOURCE projects, so no target labels are used --
the cross-project constraint is preserved.

DECISION RULE, fixed before running: K passes only if `shap_vs_random` AND
`shap_vs_shuffled` both show significant wins after Holm correction. If SHAP
only beats the all-features baseline, that is ordinary feature selection and
carries no claim about explainability.

Run: python shap_feature_selection.py
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
KEEP_FRAC = 0.5      # keep the top half of features by importance
N_DRAWS = 3          # random / shuffled selections averaged over this many draws


def evaluate(load_fn, targets, weight_map, label):
    rows = []
    for name in targets:
        X, y, cols = load_fn(name)
        use = [c for c in cols if c in weight_map]
        X = X[use]
        F = len(use)
        k = max(1, int(np.ceil(F * KEEP_FRAC)))
        imp = np.array([weight_map[c] for c in use])
        top_shap = np.argsort(-imp)[:k]                 # SHAP-guided selection

        X_tr0, _, y_tr0, _ = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=P.SEED)
        tuned = P.tune_hyperparams(X_tr0, y_tr0, f"fs_{label}_{name}")
        print(f"  {label}/{name}: {F} features -> keeping top {k}")

        for seed in P.repeated_seeds():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed)
            Xtr_all, Xte_all = X_train.values, X_test.values

            for model_name in P.make_classifiers():
                def run(sel):
                    Xtr = Xtr_all if sel is None else Xtr_all[:, sel]
                    Xte = Xte_all if sel is None else Xte_all[:, sel]
                    clf = P.build_classifier(model_name, tuned, y_train.values)
                    pipe = P.make_pipeline_for(model_name, clf)
                    pipe.fit(Xtr, y_train.values)
                    pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
                    return (f1_score(y_test, pred, average="macro"),
                            roc_auc_score(y_test, proba),
                            average_precision_score(y_test, proba))

                variants = {"all_features": run(None), "shap_select": run(top_shap)}

                rnd, shf = [], []
                for d in range(N_DRAWS):
                    rng = np.random.RandomState(1000 * seed + d)
                    rnd.append(run(rng.choice(F, size=k, replace=False)))
                    shf.append(run(np.argsort(-rng.permutation(imp))[:k]))
                variants["random_select"] = tuple(np.mean(rnd, axis=0))
                variants["shuffled_select"] = tuple(np.mean(shf, axis=0))

                for variant, (f1, auc, pr) in variants.items():
                    rows.append({"ecosystem": label, "dataset": name, "model": model_name,
                                 "variant": variant, "seed": seed,
                                 "f1": f1, "auc": auc, "prauc": pr})
    return rows


def test_selection(df):
    rows = []
    for eco in df.ecosystem.unique():
        e = df[df.ecosystem == eco]
        for dataset in e.dataset.unique():
            for model in e.model.unique():
                for metric in ["f1", "auc", "prauc"]:
                    s = e[(e.dataset == dataset) & (e.model == model)].sort_values("seed")
                    v = {k: s[s.variant == k][metric].values
                         for k in ["all_features", "shap_select", "random_select",
                                   "shuffled_select"]}
                    comps = [
                        ("shap_vs_all", v["shap_select"], v["all_features"]),
                        ("random_vs_all", v["random_select"], v["all_features"]),
                        # the two that decide it:
                        ("shap_vs_random", v["shap_select"], v["random_select"]),
                        ("shap_vs_shuffled", v["shap_select"], v["shuffled_select"]),
                    ]
                    for comp, A, B in comps:
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
    print("=== Phase K retry: SHAP-guided feature SELECTION (with controls) ===")
    apache_w = dict(pd.read_csv(OUT_DIR / "xe_shap_weights.csv")[["Feature", "Weight"]].values)
    eclipse_w = dict(pd.read_csv(OUT_DIR / "shap_weights.csv")[["Feature", "Weight"]].values)

    rows = []
    rows += evaluate(XE.load_promise, XE.XE_TARGETS, apache_w, "apache")
    rows += evaluate(P.load_xy, P.PHASE2, eclipse_w, "eclipse")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "feature_selection_scores.csv", index=False)
    res = test_selection(df)
    res.to_csv(OUT_DIR / "feature_selection_tests.csv", index=False)

    print("\n--- Mean F1 by variant ---")
    print(df.groupby(["ecosystem", "variant"])["f1"].mean().reset_index().to_string(index=False))

    sig = res[res["significant_at_0.05"]]
    print(f"\n--- {len(sig)} of {len(res)} comparisons significant after Holm ---")
    for comp in ["shap_vs_all", "random_vs_all", "shap_vs_random", "shap_vs_shuffled"]:
        s = sig[sig.comparison == comp]
        up, down = int((s.mean_diff > 0).sum()), int((s.mean_diff < 0).sum())
        print(f"  {comp:<20} {len(s):>2} significant  ({up} wins, {down} losses)")
        if len(s) and comp.startswith("shap_vs_r") or (len(s) and comp == "shap_vs_shuffled"):
            print(s[["ecosystem", "dataset", "model", "metric", "mean_diff",
                     "holm_corrected_p"]].to_string(index=False))

    n_rand = sig[(sig.comparison == "shap_vs_random") & (sig.mean_diff > 0)]
    n_shuf = sig[(sig.comparison == "shap_vs_shuffled") & (sig.mean_diff > 0)]
    print("\n=== DECISION (rule fixed before the run) ===")
    if len(n_rand) > 0 and len(n_shuf) > 0:
        print(f"  K PASSES. SHAP-guided selection beats random selection in {len(n_rand)} and")
        print(f"  shuffled-importance selection in {len(n_shuf)} comparisons. The importance")
        print("  ordering carries information that a magnitude-matched control does not.")
    else:
        print(f"  K STILL FAILS. shap_vs_random wins: {len(n_rand)}; "
              f"shap_vs_shuffled wins: {len(n_shuf)}.")
        print("  Selecting features by SHAP importance is not measurably better than")
        print("  selecting them at random. Report as a null result.")
    print("\nDone. Written to", OUT_DIR)
