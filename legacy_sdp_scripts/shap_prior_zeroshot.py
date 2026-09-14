"""
Task 7 (rerun): SHAP-as-prior evaluated in the GENUINE ZERO-SHOT CPDP setting.

Why this rerun exists. The previous Task 7 run (shap_as_prior_fixed.py) executed
before the zero-shot harness existed, so it used the old within-target design in
which the classifier already sees the target's own labels and transfer has no
headroom. It also covered only 4 of 6 targets and produced incomplete cells
(8/9/20 rows). Two further defects are corrected here:

  BUG 1 (control collapse). The old script averaged the shuffled importance
  VECTORS: shuffled_vec = mean(3 permutations). Averaging permutations of a
  mean-1 vector converges to all-ones, i.e. the shuffled control silently
  became the uniform control. Here the three permutations are run separately
  and their METRICS are averaged, which preserves the control.

  BUG 2 (wrong preprocessing for trees). The old script forced log1p +
  RobustScaler on every model. Tree models take Pipeline A (identity) in this
  codebase and receive the prior through their native APIs, not through
  scaling. Fixed.

Design (contract: 6 x 4 x 4 x 20 = 1920 rows, complete cells)
  targets  equinox, lucene, pde (Eclipse) + xalan-2.6, poi-3.0, velocity-1.6 (Apache)
  models   LogisticRegression, SVC (linear prior) and XGBoost, LightGBM (native
           tree priors). No other models: these four are the only ones with a
           mechanism by which a prior can act.
  arms     original, shap_prior, uniform, shuffled
  seeds    20 stratified 80% bootstrap resamples of the pooled source

Prior construction. Features are clustered by hierarchical average linkage on
1 - |corr| and cut at distance 0.2 (within-cluster |corr| >= 0.8). Group
importance = sum of member importances, then the per-feature vector is rescaled
to MEAN 1 -- not sum 1, since the sum-1 form is the falsified one whose
magnitude depends on feature count.

FROZEN GATE (unchanged): PASS iff shap_prior achieves >= 2 Holm-significant wins
over uniform AND >= 2 over shuffled AND has 0 Holm-significant losses to either.

Run: python shap_prior_zeroshot.py
"""
import warnings

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

import cross_ecosystem as XE
import pipeline as P

warnings.filterwarnings("ignore")

OUT_DIR = P.OUT_DIR
MODELS = ["LogisticRegression", "SVC", "XGBoost", "LightGBM"]
ARMS = ["original", "shap_prior", "uniform", "shuffled"]
N_SHUFFLE_DRAWS = 3
CORR_CUT = 0.2  # distance 1-|corr| <= 0.2  <=>  |corr| >= 0.8

# XGBoost's feature_weights sets per-column SAMPLING probabilities. With
# colsample_bytree at its default of 1.0 every column enters every tree, so the
# weights have no opportunity to act -- in the first run the prior and uniform
# arms were bit-identical (max difference 0.0), i.e. the arm was vacuous.
# Subsampling columns is what gives the weights something to bias. This value is
# applied to XGBoost in EVERY arm, including the controls, so the comparison
# still isolates the prior rather than the sampling rate.
XGB_COLSAMPLE = 0.8
TARGETS = {"eclipse": ["equinox", "lucene", "pde"],
           "apache": ["xalan-2.6", "poi-3.0", "velocity-1.6"]}
SOURCES = {"eclipse": P.PHASE1, "apache": XE.XE_SOURCES}


def load_any(eco, name):
    return P.load_xy(name) if eco == "eclipse" else XE.load_promise(name)


def pooled_source(eco):
    Xs, ys, common = [], [], None
    for n in SOURCES[eco]:
        X, y, cols = load_any(eco, n)
        common = set(cols) if common is None else common & set(cols)
        Xs.append(X); ys.append(y)
    common = sorted(common)
    X = pd.concat([x[common] for x in Xs], ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    return X, y, common


def group_prior(X_src, cols, weight_map):
    """Cluster correlated features, sum importances within a group, rescale to
    mean 1. Returns the per-feature prior vector and the group labels."""
    imp = np.array([weight_map[c] for c in cols], dtype=float)
    F = len(cols)
    if F < 2:
        return np.ones(F), np.arange(F)
    C = np.abs(np.corrcoef(X_src.values.T))
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    dist = 1.0 - C
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0            # enforce exact symmetry for squareform
    Z = linkage(squareform(dist, checks=False), method="average")
    groups = fcluster(Z, t=CORR_CUT, criterion="distance")
    gsum = {g: imp[groups == g].sum() for g in np.unique(groups)}
    vec = np.array([gsum[g] for g in groups], dtype=float)
    vec = vec / vec.mean()                   # MEAN 1, not sum 1
    return vec, groups


def build_and_fit(model_name, clf, arm, vec, X_train, y_train):
    """Pipeline A (identity) for tree models -- they receive the prior through
    their native API. Pipeline B for linear models, with the prior inserted
    AFTER RobustScaler so no downstream step can undo it."""
    is_tree = model_name in ("XGBoost", "LightGBM")
    use_prior = arm != "original"

    # Applied to every arm so the controls share the same base configuration.
    if model_name == "XGBoost":
        clf.set_params(colsample_bytree=XGB_COLSAMPLE)

    if is_tree:
        pipe = SkPipeline([("clf", clf)])
        if use_prior and model_name == "XGBoost":
            try:
                pipe.fit(X_train, y_train, clf__feature_weights=vec)
                return pipe, "xgb_feature_weights"
            except Exception as e:
                raise RuntimeError(f"XGBoost feature_weights unavailable: {e}")
        if use_prior and model_name == "LightGBM":
            try:
                clf.set_params(feature_contri=list(vec))
                pipe = SkPipeline([("clf", clf)])
                pipe.fit(X_train, y_train)
                return pipe, "lgbm_feature_contri"
            except Exception as e:
                raise RuntimeError(f"LightGBM feature_contri unavailable: {e}")
        pipe.fit(X_train, y_train)
        return pipe, "none"

    steps = [("log1p", FunctionTransformer(lambda X: np.log1p(np.abs(X)))),
             ("scaler", RobustScaler())]
    if use_prior:
        steps.append(("prior", FunctionTransformer(lambda X, v=vec: X * np.sqrt(v))))
    steps += [("smote", P.ConditionalSMOTE()), ("clf", clf)]
    pipe = ImbPipeline(steps)
    pipe.fit(X_train, y_train)
    return pipe, "linear_sqrt_after_scaler"


def score(pipe, X_te, y_te):
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]
    return (f1_score(y_te, pred, average="macro"),
            roc_auc_score(y_te, proba),
            average_precision_score(y_te, proba))


def run():
    print("=== Task 7 rerun: SHAP-as-prior, ZERO-SHOT CPDP ===")
    w_maps = {
        "eclipse": dict(pd.read_csv(OUT_DIR / "shap_weights.csv")[["Feature", "Weight"]].values),
        "apache": dict(pd.read_csv(OUT_DIR / "xe_shap_weights.csv")[["Feature", "Weight"]].values),
    }
    rows, prior_audit = [], []

    for eco in ["eclipse", "apache"]:
        X_src_full, y_src, src_cols = pooled_source(eco)
        wmap = w_maps[eco]
        use_src = [c for c in src_cols if c in wmap]
        tuned = P.tune_hyperparams(X_src_full[use_src], y_src, f"prior_zs_{eco}")

        for target in TARGETS[eco]:
            X_tgt, y_tgt, t_cols = load_any(eco, target)
            use = [c for c in use_src if c in t_cols]
            X_src = X_src_full[use]
            X_t, y_t = X_tgt[use].values, y_tgt.values

            vec, groups = group_prior(X_src, use, wmap)
            nonconst = bool(np.std(vec) > 1e-9)
            n_groups = len(np.unique(groups))
            print(f"  [{eco}/{target}] F={len(use)} groups={n_groups} "
                  f"prior std={np.std(vec):.4f} nonconstant={nonconst}")
            prior_audit.append({"ecosystem": eco, "target": target, "n_features": len(use),
                                "n_groups": n_groups, "prior_std": float(np.std(vec)),
                                "nonconstant": nonconst,
                                "prior_min": float(vec.min()), "prior_max": float(vec.max())})
            if not nonconst:
                print("    WARNING: prior vector is constant -> prior is a no-op here.")

            uniform_vec = np.ones(len(use))

            for seed in P.repeated_seeds():
                sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
                tr_idx, _ = next(sss.split(X_src, y_src))
                Xb, yb = X_src.iloc[tr_idx].values, y_src.iloc[tr_idx].values
                rng = np.random.RandomState(1000 + seed)

                for model_name in MODELS:
                    def fresh():
                        return P.build_classifier(model_name, tuned, yb)

                    for arm in ARMS:
                        if arm == "shuffled":
                            # average METRICS over separate permutations; averaging
                            # the vectors would collapse this into the uniform arm
                            accum = []
                            for _ in range(N_SHUFFLE_DRAWS):
                                pv = rng.permutation(vec)
                                pipe, _m = build_and_fit(model_name, fresh(), arm, pv, Xb, yb)
                                accum.append(score(pipe, X_t, y_t))
                            f1, auc, pr = tuple(np.mean(accum, axis=0))
                        else:
                            v = {"original": uniform_vec, "shap_prior": vec,
                                 "uniform": uniform_vec}[arm]
                            pipe, _m = build_and_fit(model_name, fresh(), arm, v, Xb, yb)
                            f1, auc, pr = score(pipe, X_t, y_t)

                        rows.append({"ecosystem": eco, "target": target, "model": model_name,
                                     "arm": arm, "boot_seed": seed,
                                     "f1": f1, "auc": auc, "prauc": pr})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "shap_prior_zeroshot_scores.csv", index=False)
    pd.DataFrame(prior_audit).to_csv(OUT_DIR / "shap_prior_group_audit.csv", index=False)
    print(f"\nrows={len(df)} (expected 1920)  targets={df.target.nunique()}  models={df.model.nunique()}")

    # ---- significance: shap_prior vs uniform and vs shuffled ----
    test_rows = []
    for (eco, target, model), g in df.groupby(["ecosystem", "target", "model"]):
        g = g.sort_values("boot_seed")
        for metric in ["f1", "auc", "prauc"]:
            sp = g[g.arm == "shap_prior"][metric].values
            for ctrl in ["uniform", "shuffled"]:
                cv = g[g.arm == ctrl][metric].values
                if len(sp) != len(cv) or np.all(sp == cv):
                    p = 1.0
                else:
                    try:
                        p = wilcoxon(sp, cv).pvalue
                    except ValueError:
                        p = 1.0
                test_rows.append({"ecosystem": eco, "target": target, "model": model,
                                  "metric": metric, "comparison": f"prior_vs_{ctrl}",
                                  "mean_A": sp.mean(), "mean_B": cv.mean(),
                                  "mean_diff": (sp - cv).mean(), "wilcoxon_p": p})
    t = pd.DataFrame(test_rows)
    adj, sig = P.holm_bonferroni(t["wilcoxon_p"].values)
    t["holm_corrected_p"] = adj
    t["significant_at_0.05"] = sig
    t.to_csv(OUT_DIR / "shap_prior_zeroshot_tests.csv", index=False)

    s = t[t["significant_at_0.05"]]
    wu = s[(s.comparison == "prior_vs_uniform") & (s.mean_diff > 0)]
    lu = s[(s.comparison == "prior_vs_uniform") & (s.mean_diff < 0)]
    ws = s[(s.comparison == "prior_vs_shuffled") & (s.mean_diff > 0)]
    ls = s[(s.comparison == "prior_vs_shuffled") & (s.mean_diff < 0)]
    passed = bool(len(wu) >= 2 and len(ws) >= 2 and len(lu) == 0 and len(ls) == 0)

    print(f"\nsignificant: {len(s)}/{len(t)}")
    print(f"  vs uniform : {len(wu)} wins, {len(lu)} losses")
    print(f"  vs shuffled: {len(ws)} wins, {len(ls)} losses")
    print("\n=== GATE 4 ===")
    print("  PASS" if passed else "  FAIL -- reported as the completing negative result.")
    if len(s):
        print(s[["target", "model", "metric", "comparison", "mean_diff",
                 "holm_corrected_p"]].to_string(index=False))

    pd.DataFrame([{"wins_uniform": len(wu), "losses_uniform": len(lu),
                   "wins_shuffled": len(ws), "losses_shuffled": len(ls),
                   "passed": passed,
                   "all_priors_nonconstant": bool(all(a["nonconstant"] for a in prior_audit)),
                   "setting": "zero-shot", "n_targets": int(df.target.nunique()),
                   "n_rows": len(df)}]).to_csv(
        OUT_DIR / "shap_prior_zeroshot_verdict.csv", index=False)
    return df, t


if __name__ == "__main__":
    run()
    print("\nDone. Written to", OUT_DIR)
