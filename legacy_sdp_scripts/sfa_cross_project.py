"""
Phase 3: Shapley-based feature AUGMENTATION (SFA) under distribution shift.

WHAT ANTWARG ET AL. ESTABLISHED (Information Fusion 96 (2023) 92-102).
SFA is a two-stage learner. Stage 1 produces out-of-fold predictions and their
per-instance Shapley values; these are appended to the original features; stage 2
trains on the augmented matrix. On 20 OpenML datasets with three decision forests
they showed SFA beats `base`, Featuretools and PCA-Augment, and -- crucially --
they ran the ablation that isolates the Shapley contribution, `P augmented`
(original features + OOF predictions, NO Shapley values). SFA beat it
significantly for XGBoost and LightGBM (but not Random Forest).

That is a solid positive result and this file does not dispute it. Their design
is entirely WITHIN-distribution: train and test come from the same dataset, and
out-of-fold explanations of the same data-generating process are available at
inference. Whether the benefit survives distribution shift is the question their
paper leaves open, and it is the question this file answers.

CROSS-PROJECT ADAPTATION. In zero-shot CPDP there are no target labels, so
stage 2 cannot be trained on the target. The augmentation FUNCTION transfers
instead of the augmented data:
    stage 1  fit on pooled SOURCE; produce OOF predictions and OOF Shapley
             values FOR THE SOURCE, and (refit on all of source) predictions and
             Shapley values FOR THE TARGET.
    stage 2  train on the augmented SOURCE, predict the augmented TARGET.
No target label is used at any point.

ARMS (Antwarg's own ablation ladder, transplanted)
    base        X                      -- no augmentation
    p_aug       X + prediction         -- their control: isolates Shapley
    shap_aug    X + Shapley values
    ps_aug      X + prediction + Shapley values   -- the full SFA representation

PRE-REGISTERED DECISION RULES (frozen before execution)
    H1 (replication under shift) ps_aug beats base.
    H2 (THE DECISIVE ONE)        ps_aug beats p_aug -- i.e. the Shapley values
                                 add something beyond the prediction alone.
                                 This is Antwarg's own isolating comparison.
    "SHAP augmentation transfers" is claimed iff H2 yields >= 2 Holm-significant
    wins AND 0 Holm-significant losses. H1 alone is NOT sufficient: augmenting
    with a prediction is ordinary stacking and carries no claim about
    explainability.

Run: python sfa_cross_project.py
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import cross_ecosystem as XE
import pipeline as P

warnings.filterwarnings("ignore")

OUT_DIR = P.OUT_DIR
# Models with a tractable explainer for this design (SHAP must be computed for
# every source AND target instance). Antwarg's three decision forests are all
# here, plus two more tree families and a linear model -- which begins to
# address their open question FW2 on base-learner generality.
MODELS = ["RandomForest", "XGBoost", "LightGBM", "ExtraTrees",
          "GradientBoosting", "LogisticRegression"]
ARMS = ["base", "p_aug", "shap_aug", "ps_aug"]
TARGETS = {"eclipse": ["equinox", "lucene", "pde"],
           "apache": ["xalan-2.6", "poi-3.0", "velocity-1.6"]}
SOURCES = {"eclipse": P.PHASE1, "apache": XE.XE_SOURCES}
N_FOLDS = 5


def load_any(eco, name):
    return P.load_xy(name) if eco == "eclipse" else XE.load_promise(name)


def pooled_source(eco):
    Xs, ys, common = [], [], None
    for n in SOURCES[eco]:
        X, y, cols = load_any(eco, n)
        common = set(cols) if common is None else common & set(cols)
        Xs.append(X); ys.append(y)
    common = sorted(common)
    return (pd.concat([x[common] for x in Xs], ignore_index=True),
            pd.concat(ys, ignore_index=True), common)


def stage1_artifacts(X_src, y_src, X_tgt, model_name, tuned):
    """Out-of-fold predictions and Shapley values for the SOURCE, plus
    predictions and Shapley values for the TARGET from a model refit on all of
    the source. Out-of-fold is used on the source side so stage 2 never sees a
    stage-1 model that was trained on the row it is describing."""
    n, F = X_src.shape
    oof_pred = np.zeros(n)
    oof_shap = np.zeros((n, F))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=P.SEED)

    for tr, va in skf.split(X_src, y_src):
        clf = P.build_classifier(model_name, tuned, y_src[tr])
        pipe = P.make_pipeline_for(model_name, clf)
        pipe.fit(X_src[tr], y_src[tr])
        oof_pred[va] = pipe.predict_proba(X_src[va])[:, 1]
        Xtr_p = P.transform_only(pipe, X_src[tr])
        Xva_p = P.transform_only(pipe, X_src[va])
        oof_shap[va] = P.compute_shap_values(pipe, Xtr_p, Xva_p, model_name)

    full = P.build_classifier(model_name, tuned, y_src)
    fpipe = P.make_pipeline_for(model_name, full)
    fpipe.fit(X_src, y_src)
    tgt_pred = fpipe.predict_proba(X_tgt)[:, 1]
    Xs_p = P.transform_only(fpipe, X_src)
    Xt_p = P.transform_only(fpipe, X_tgt)
    tgt_shap = P.compute_shap_values(fpipe, Xs_p, Xt_p, model_name)
    return oof_pred, oof_shap, tgt_pred, np.asarray(tgt_shap)


def assemble(arm, X, pred, shap):
    if arm == "base":
        return X
    if arm == "p_aug":
        return np.hstack([X, pred.reshape(-1, 1)])
    if arm == "shap_aug":
        return np.hstack([X, shap])
    return np.hstack([X, pred.reshape(-1, 1), shap])


def run():
    print("=== Phase 3: SFA (Shapley feature augmentation) under distribution shift ===")
    rows = []

    for eco in ["eclipse", "apache"]:
        X_src_df, y_src_s, src_cols = pooled_source(eco)
        for target in TARGETS[eco]:
            X_tgt_df, y_tgt_s, t_cols = load_any(eco, target)
            use = [c for c in src_cols if c in t_cols]
            X_src = X_src_df[use].values
            y_src = y_src_s.values
            X_tgt = X_tgt_df[use].values
            y_tgt = y_tgt_s.values
            tuned = P.tune_hyperparams(X_src_df[use], y_src_s, f"sfa_{eco}_{target}")
            print(f"\n  [{eco}/{target}] F={len(use)} n_src={len(X_src)} n_tgt={len(X_tgt)}")

            for model_name in MODELS:
                op, os_, tp, ts = stage1_artifacts(X_src, y_src, X_tgt, model_name, tuned)
                print(f"    {model_name:<19} stage-1 done "
                      f"(src shap {os_.shape}, tgt shap {ts.shape})")

                for arm in ARMS:
                    A_src = assemble(arm, X_src, op, os_)
                    A_tgt = assemble(arm, X_tgt, tp, ts)
                    for seed in P.repeated_seeds():
                        sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8,
                                                     random_state=seed)
                        tr, _ = next(sss.split(A_src, y_src))
                        clf = P.build_classifier(model_name, tuned, y_src[tr])
                        pipe = P.make_pipeline_for(model_name, clf)
                        pipe.fit(A_src[tr], y_src[tr])
                        pred = pipe.predict(A_tgt)
                        proba = pipe.predict_proba(A_tgt)[:, 1]
                        rows.append({
                            "ecosystem": eco, "target": target, "model": model_name,
                            "arm": arm, "boot_seed": seed,
                            "f1": f1_score(y_tgt, pred, average="macro"),
                            "auc": roc_auc_score(y_tgt, proba),
                            "prauc": average_precision_score(y_tgt, proba)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "sfa_cross_project_scores.csv", index=False)
    print(f"\nrows={len(df)} (expected {6*len(MODELS)*len(ARMS)*20})")

    print("\n--- Mean by arm ---")
    print(df.groupby(["ecosystem", "arm"])[["f1", "auc", "prauc"]].mean().round(4).to_string())

    # ---- significance ----
    test_rows = []
    for (eco, target, model), g in df.groupby(["ecosystem", "target", "model"]):
        g = g.sort_values("boot_seed")
        for metric in ["f1", "auc", "prauc"]:
            v = {a: g[g.arm == a][metric].values for a in ARMS}
            for name, A, B in [("ps_vs_base", v["ps_aug"], v["base"]),
                               ("ps_vs_p", v["ps_aug"], v["p_aug"]),
                               ("shap_vs_base", v["shap_aug"], v["base"]),
                               ("p_vs_base", v["p_aug"], v["base"])]:
                if len(A) != len(B) or np.all(A == B):
                    p = 1.0
                else:
                    try:
                        p = wilcoxon(A, B).pvalue
                    except ValueError:
                        p = 1.0
                test_rows.append({"ecosystem": eco, "target": target, "model": model,
                                  "metric": metric, "comparison": name,
                                  "mean_A": A.mean(), "mean_B": B.mean(),
                                  "mean_diff": (A - B).mean(), "wilcoxon_p": p})
    t = pd.DataFrame(test_rows)
    adj, sig = P.holm_bonferroni(t["wilcoxon_p"].values)
    t["holm_corrected_p"] = adj
    t["significant_at_0.05"] = sig
    t.to_csv(OUT_DIR / "sfa_cross_project_tests.csv", index=False)

    s = t[t["significant_at_0.05"]]
    print(f"\n--- {len(s)} of {len(t)} significant after Holm ---")
    for c in ["p_vs_base", "shap_vs_base", "ps_vs_base", "ps_vs_p"]:
        sc = s[s.comparison == c]
        print(f"  {c:<14} {len(sc):>3} significant  "
              f"({int((sc.mean_diff > 0).sum())} wins, {int((sc.mean_diff < 0).sum())} losses)")

    h1w = s[(s.comparison == "ps_vs_base") & (s.mean_diff > 0)]
    h2w = s[(s.comparison == "ps_vs_p") & (s.mean_diff > 0)]
    h2l = s[(s.comparison == "ps_vs_p") & (s.mean_diff < 0)]
    h2 = bool(len(h2w) >= 2 and len(h2l) == 0)

    print("\n=== PRE-REGISTERED VERDICT ===")
    print(f"  H1 ps_aug beats base : {len(h1w)} significant wins")
    print(f"  H2 ps_aug beats p_aug: {len(h2w)} wins, {len(h2l)} losses  "
          f"-> {'SUPPORTED' if h2 else 'NOT supported'}")
    if h2:
        print("  CLAIM: Shapley augmentation carries information beyond the")
        print("  prediction alone, and that benefit survives distribution shift.")
    else:
        print("  Shapley values add nothing measurable beyond the stage-1 prediction")
        print("  once the model is transferred across projects. Antwarg et al.'s")
        print("  within-distribution result does not extend to this setting.")
    if len(h2w) or len(h2l):
        print(s[s.comparison == "ps_vs_p"][["target", "model", "metric", "mean_diff",
                                            "holm_corrected_p"]].to_string(index=False))

    pd.DataFrame([{"H1_wins": len(h1w), "H2_wins": len(h2w), "H2_losses": len(h2l),
                   "H2_supported": h2, "n_rows": len(df),
                   "n_models": df.model.nunique(),
                   "n_targets": df.target.nunique()}]).to_csv(
        OUT_DIR / "sfa_cross_project_verdict.csv", index=False)
    return df, t


if __name__ == "__main__":
    run()
    print("\nDone. Written to", OUT_DIR)
