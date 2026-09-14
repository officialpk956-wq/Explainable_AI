"""
Cross-ecosystem external-validity replication.

The Eclipse study (pipeline.py) uses 5 cumulative bug-history counts across
5 Eclipse-family projects. This repeats the identical two-phase protocol on a
DIFFERENT ecosystem (Apache, from the PROMISE/Jureczko corpus) with a DIFFERENT
feature schema (20 CK/OO static metrics -- wmc, dit, noc, cbo, rfc, lcom, ...).

If "SHAP-weighting helps linear/margin classifiers and leaves trees untouched"
is a property of the method, it should replicate here. If it only held on
Eclipse, it was a benchmark artifact. That is the whole point of this file.

Dataset inclusion rule, fixed BEFORE looking at any result: a project is used
only if its minority class has >= 30 instances, since 5-fold stratified CV with
SMOTE is unreliable below that. This excludes jedit-4.3 (11 buggy) and
log4j-1.2 (16 clean). Stated up front so the selection is a criterion, not a
post-hoc choice.

Run: python cross_ecosystem.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import pipeline as P

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data_promise"
OUT_DIR = P.OUT_DIR
SEED = P.SEED

# Apache/PROMISE projects. Sources chosen to mirror the Eclipse setup (two
# moderately-imbalanced, reasonably sized projects); targets are the remaining
# projects that pass the >=30-minority-instance rule.
XE_SOURCES = ["ant-1.7", "camel-1.6"]
XE_TARGETS = ["xalan-2.6", "poi-3.0", "velocity-1.6"]
MIN_MINORITY = 30


def load_promise(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    drop = [c for c in ["name", "name.1", "version", "bug"] if c in df.columns]
    X = df.drop(columns=drop).astype(float)
    y = (df["bug"] > 0).astype(int)
    keep = X.columns[X.std(axis=0) > 0]           # same zero-variance rule as pipeline.load_xy
    return X[keep], y, list(keep)


def check_inclusion():
    print("=== Cross-ecosystem corpus (Apache / PROMISE, 20 CK-OO metrics) ===")
    rows = []
    for name in XE_SOURCES + XE_TARGETS:
        X, y, cols = load_promise(name)
        n1, n0 = int(y.sum()), int((y == 0).sum())
        ok = min(n1, n0) >= MIN_MINORITY
        role = "source" if name in XE_SOURCES else "target"
        rows.append({"project": name, "role": role, "n": len(X), "features": len(cols),
                     "buggy": n1, "clean": n0, "IR": round(n0 / max(1, n1), 2),
                     "passes_inclusion": ok})
        assert ok, f"{name} fails the pre-stated minority>={MIN_MINORITY} rule"
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "xe_datasets.csv", index=False)
    print(df.to_string(index=False))
    return df


def xe_phase1():
    """Same protocol as pipeline.phase1: tune, 5-fold CV, pick best by F1-macro,
    derive out-of-fold SHAP weights, average across the two source projects."""
    print("\n=== Cross-ecosystem Phase 1 (sources: %s) ===" % ", ".join(XE_SOURCES))
    weight_vecs, best_models, feature_sets = {}, {}, {}

    for name in XE_SOURCES:
        X, y, cols = load_promise(name)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED)
        tuned = P.tune_hyperparams(X_train, y_train, f"xe_{name}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        rows = []
        for model_name in P.make_classifiers():
            f1s, aucs, prs = [], [], []
            for tr, va in skf.split(X_train, y_train):
                Xtr, Xva = X_train.iloc[tr].values, X_train.iloc[va].values
                ytr, yva = y_train.iloc[tr].values, y_train.iloc[va].values
                clf = P.build_classifier(model_name, tuned, ytr)
                pipe = P.make_pipeline_for(model_name, clf)
                pipe.fit(Xtr, ytr)
                pred, proba = pipe.predict(Xva), pipe.predict_proba(Xva)[:, 1]
                f1s.append(f1_score(yva, pred, average="macro"))
                aucs.append(roc_auc_score(yva, proba))
                prs.append(average_precision_score(yva, proba))
            rows.append({"Model": model_name, "F1_mean": np.mean(f1s), "F1_std": np.std(f1s),
                         "AUC_mean": np.mean(aucs), "AUC_std": np.std(aucs),
                         "PRAUC_mean": np.mean(prs), "PRAUC_std": np.std(prs)})

        cv = pd.DataFrame(rows).sort_values("F1_mean", ascending=False).reset_index(drop=True)
        cv.to_csv(OUT_DIR / f"xe_phase1_cv_{name}.csv", index=False)
        print(f"\n{name} CV (top 3):")
        print(cv.head(3).to_string(index=False))

        best = cv.iloc[0]["Model"]
        best_models[name] = best
        mean_abs = P.shap_weights_out_of_fold(X_train, y_train, best, tuned)
        weight_vecs[name] = P.normalise(mean_abs)
        feature_sets[name] = cols

    # Average the two source weight vectors over their shared features (Eq. 4).
    shared = [c for c in feature_sets[XE_SOURCES[0]] if c in feature_sets[XE_SOURCES[1]]]
    stacked = []
    for name in XE_SOURCES:
        idx = {c: i for i, c in enumerate(feature_sets[name])}
        stacked.append(np.array([weight_vecs[name][idx[c]] for c in shared]))
    w = np.mean(stacked, axis=0)
    w = w / w.sum()
    wdf = pd.DataFrame({"Feature": shared, "Weight": w}).sort_values("Weight", ascending=False)
    wdf.to_csv(OUT_DIR / "xe_shap_weights.csv", index=False)
    print("\nCross-ecosystem SHAP weights (top 8 of %d):" % len(shared))
    print(wdf.head(8).to_string(index=False))
    print("Best Phase-1 models:", best_models)
    return dict(zip(shared, w)), best_models


def xe_phase2(weight_map):
    """20 repeated stratified splits x 8 classifiers x 3 variants, exactly as
    pipeline.phase2_repeated, on the Apache targets."""
    print("\n=== Cross-ecosystem Phase 2 (targets: %s) ===" % ", ".join(XE_TARGETS))
    X_src = pd.concat([load_promise(n)[0] for n in XE_SOURCES], axis=0).reset_index(drop=True)
    all_rows = []

    for name in XE_TARGETS:
        X, y, cols = load_promise(name)
        use = [c for c in cols if c in weight_map and c in X_src.columns]
        X = X[use]
        w = np.array([weight_map[c] for c in use])
        X_train0, X_test0, y_train0, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED)
        tuned = P.tune_hyperparams(X_train0, y_train0, f"xe_{name}")
        print(f"  {name}: {len(use)} shared features, tuned; running 20 splits...")

        for seed in P.repeated_seeds():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed)
            Xc_tr, Xc_te = P.coral_align(X_src[use].values, X_train.values, X_test.values)

            for model_name in P.make_classifiers():
                def run(Xtr, Xte):
                    clf = P.build_classifier(model_name, tuned, y_train.values)
                    pipe = P.make_pipeline_for(model_name, clf)
                    pipe.fit(Xtr, y_train.values)
                    pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
                    return (f1_score(y_test, pred, average="macro"),
                            roc_auc_score(y_test, proba),
                            average_precision_score(y_test, proba))

                for variant, (Xtr, Xte) in {
                    "original": (X_train.values, X_test.values),
                    "shap_weighted": ((X_train * w).values, (X_test * w).values),
                    "coral_aligned": (Xc_tr, Xc_te),
                }.items():
                    f1, auc, pr = run(Xtr, Xte)
                    all_rows.append({"dataset": name, "model": model_name, "variant": variant,
                                     "seed": seed, "f1": f1, "auc": auc, "prauc": pr})

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "xe_phase2_repeated_scores.csv", index=False)
    print("\nMean F1 by variant:")
    print(df.groupby(["dataset", "variant"])["f1"].mean().reset_index().to_string(index=False))
    return df


def xe_significance(df):
    """Paired Wilcoxon + Holm-Bonferroni, identical to pipeline.significance_tests."""
    print("\n=== Cross-ecosystem significance testing ===")
    rows = []
    for dataset in df.dataset.unique():
        for model in df.model.unique():
            for metric in ["f1", "auc", "prauc"]:
                sub = df[(df.dataset == dataset) & (df.model == model)].sort_values("seed")
                o = sub[sub.variant == "original"][metric].values
                s = sub[sub.variant == "shap_weighted"][metric].values
                c = sub[sub.variant == "coral_aligned"][metric].values
                for comp, A, B in [("shap_vs_original", s, o), ("coral_vs_original", c, o),
                                   ("shap_vs_coral", s, c)]:
                    if np.all(A == B):
                        stat, p = 0.0, 1.0
                    else:
                        try:
                            stat, p = wilcoxon(A, B)
                        except ValueError:
                            stat, p = 0.0, 1.0
                    d = A - B
                    sd = np.std(d, ddof=1)
                    rows.append({"comparison": comp, "dataset": dataset, "model": model,
                                 "metric": metric, "mean_A": A.mean(), "mean_B": B.mean(),
                                 "mean_diff": d.mean(), "wilcoxon_p": p,
                                 "effect_size": 0.0 if sd == 0 else d.mean() / sd})
    res = pd.DataFrame(rows)
    adj, sig = P.holm_bonferroni(res["wilcoxon_p"].values)
    res["holm_corrected_p"] = adj
    res["significant_at_0.05"] = sig
    res.to_csv(OUT_DIR / "xe_significance_tests.csv", index=False)

    sig_only = res[res["significant_at_0.05"]]
    print(f"Tested {len(res)} combinations; {len(sig_only)} significant after Holm correction.")
    if len(sig_only):
        print(sig_only.sort_values("mean_diff", ascending=False).to_string(index=False))
    return res


def replication_verdict(res):
    """The question this whole file exists to answer: does the Eclipse finding
    hold in a different ecosystem with a different feature schema?"""
    print("\n=== REPLICATION VERDICT (vs. the Eclipse study) ===")
    lin = {"SVC", "LogisticRegression", "KNN"}
    shap_vs_orig = res[res.comparison == "shap_vs_original"]

    lin_sig = shap_vs_orig[(shap_vs_orig.model.isin(lin)) & (shap_vs_orig["significant_at_0.05"])]
    tree_rows = shap_vs_orig[~shap_vs_orig.model.isin(lin)]
    tree_moved = tree_rows[tree_rows["mean_diff"].abs() > 0.01]

    print(f"  Claim 1 -- SHAP-weighting helps linear/margin models:")
    print(f"    {len(lin_sig)} significant linear/KNN results "
          f"({(lin_sig['mean_diff'] > 0).sum()} improvements, "
          f"{(lin_sig['mean_diff'] < 0).sum()} degradations)")
    if len(lin_sig):
        print(lin_sig[["dataset", "model", "metric", "mean_diff",
                       "holm_corrected_p"]].to_string(index=False))

    print(f"\n  Claim 2 -- tree models are ~unaffected (scale-invariance):")
    print(f"    {len(tree_moved)}/{len(tree_rows)} tree results moved more than 1 point "
          f"(expected: near zero)")

    coral = res[(res.comparison == "shap_vs_coral") & (res["significant_at_0.05"])]
    print(f"\n  Claim 3 -- SHAP-weighting vs CORAL: {len(coral)} significant differences "
          f"({(coral['mean_diff'] > 0).sum()} favour SHAP, "
          f"{(coral['mean_diff'] < 0).sum()} favour CORAL)")
    return {"linear_significant": len(lin_sig), "tree_moved": len(tree_moved),
            "coral_diffs": len(coral)}


if __name__ == "__main__":
    check_inclusion()
    weight_map, best_models = xe_phase1()
    scores = xe_phase2(weight_map)
    res = xe_significance(scores)
    replication_verdict(res)
    print("\nDone. Cross-ecosystem outputs written to", OUT_DIR)
