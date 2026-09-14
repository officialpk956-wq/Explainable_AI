"""
The decisive control: is the SHAP-weighting effect actually caused by SHAP?

Fixing the feature-count dependence (scale_invariant_fix.py) removed the harm on
Apache AND the benefit on Eclipse -- leaving the transform inert. That is only
consistent with one hypothesis: the measured effect comes from the MAGNITUDE of
the multiplier interacting with log1p, not from the information SHAP provides
about which features matter.

This tests that directly with two magnitude-matched ablations that carry no
usable SHAP signal:

  uniform_sum1   every feature gets weight 1/F. Identical sum, identical mean,
                 ZERO importance information -- a pure scaling control.
  shuffled_shap  the real SHAP weights, randomly permuted across features.
                 Identical multiset of magnitudes, but each weight is attached
                 to the wrong feature, so the importance mapping is destroyed.

If either reproduces the published variant's effect, the SHAP attributions are
doing no work and the reported gains are a scaling artifact.

Run: python ablation_control.py
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
N_SHUFFLES = 5  # average over several permutations so one unlucky shuffle can't decide it


def run_ablation(load_fn, targets, weight_map, label):
    rows = []
    for name in targets:
        X, y, cols = load_fn(name)
        use = [c for c in cols if c in weight_map]
        X = X[use]
        F = len(use)
        w_shap = np.array([weight_map[c] for c in use])
        w_shap = w_shap / w_shap.sum()
        w_uniform = np.full(F, 1.0 / F)          # same sum, no information

        X_tr0, _, y_tr0, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=P.SEED)
        tuned = P.tune_hyperparams(X_tr0, y_tr0, f"abl_{label}_{name}")
        print(f"  {label}/{name}: F={F}, tuned; 20 splits x 8 models...")

        for seed in P.repeated_seeds():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed)
            for model_name in P.make_classifiers():
                def run(mult):
                    Xtr = X_train.values if mult is None else (X_train * mult).values
                    Xte = X_test.values if mult is None else (X_test * mult).values
                    clf = P.build_classifier(model_name, tuned, y_train.values)
                    pipe = P.make_pipeline_for(model_name, clf)
                    pipe.fit(Xtr, y_train.values)
                    pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
                    return (f1_score(y_test, pred, average="macro"),
                            roc_auc_score(y_test, proba),
                            average_precision_score(y_test, proba))

                variants = {"original": run(None),
                            "shap_sum1": run(w_shap),
                            "uniform_sum1": run(w_uniform)}

                # shuffled SHAP: same magnitudes, wrong features, averaged over
                # N_SHUFFLES permutations seeded off the split seed
                sh = []
                for k in range(N_SHUFFLES):
                    rng = np.random.RandomState(1000 * seed + k)
                    sh.append(run(rng.permutation(w_shap)))
                variants["shuffled_shap"] = tuple(np.mean(sh, axis=0))

                for variant, (f1, auc, pr) in variants.items():
                    rows.append({"ecosystem": label, "dataset": name, "model": model_name,
                                 "variant": variant, "seed": seed,
                                 "f1": f1, "auc": auc, "prauc": pr})
    return rows


def test_ablation(df):
    rows = []
    for eco in df.ecosystem.unique():
        e = df[df.ecosystem == eco]
        for dataset in e.dataset.unique():
            for model in e.model.unique():
                for metric in ["f1", "auc", "prauc"]:
                    s = e[(e.dataset == dataset) & (e.model == model)].sort_values("seed")
                    v = {k: s[s.variant == k][metric].values
                         for k in ["original", "shap_sum1", "uniform_sum1", "shuffled_shap"]}
                    comps = [
                        ("shap_vs_original", v["shap_sum1"], v["original"]),
                        ("uniform_vs_original", v["uniform_sum1"], v["original"]),
                        ("shuffled_vs_original", v["shuffled_shap"], v["original"]),
                        # the money comparison: does real SHAP beat magnitude-matched noise?
                        ("shap_vs_uniform", v["shap_sum1"], v["uniform_sum1"]),
                        ("shap_vs_shuffled", v["shap_sum1"], v["shuffled_shap"]),
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
    print("=== ABLATION: does the SHAP information do any work? ===")
    apache_w = dict(pd.read_csv(OUT_DIR / "xe_shap_weights.csv")[["Feature", "Weight"]].values)
    eclipse_w = dict(pd.read_csv(OUT_DIR / "shap_weights.csv")[["Feature", "Weight"]].values)

    rows = []
    rows += run_ablation(XE.load_promise, XE.XE_TARGETS, apache_w, "apache")
    rows += run_ablation(P.load_xy, P.PHASE2, eclipse_w, "eclipse")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ablation_scores.csv", index=False)
    res = test_ablation(df)
    res.to_csv(OUT_DIR / "ablation_tests.csv", index=False)

    print("\n--- Mean F1 by variant (magnitude-matched controls) ---")
    print(df.groupby(["ecosystem", "variant"])["f1"].mean().reset_index().to_string(index=False))

    sig = res[res["significant_at_0.05"]]
    print(f"\n--- {len(sig)} of {len(res)} comparisons significant after Holm ---")
    for comp in ["shap_vs_original", "uniform_vs_original", "shuffled_vs_original",
                 "shap_vs_uniform", "shap_vs_shuffled"]:
        s = sig[sig.comparison == comp]
        print(f"  {comp:<24} {len(s):>2} significant "
              f"({int((s.mean_diff > 0).sum())} improvements, {int((s.mean_diff < 0).sum())} degradations)")

    print("\n--- VERDICT: is SHAP information necessary? ---")
    n_beats_uniform = len(sig[(sig.comparison == "shap_vs_uniform")])
    n_beats_shuffled = len(sig[(sig.comparison == "shap_vs_shuffled")])
    if n_beats_uniform == 0 and n_beats_shuffled == 0:
        print("  NO. Real SHAP weights are statistically indistinguishable from")
        print("  magnitude-matched uniform and shuffled controls. The measured effect")
        print("  is attributable to the multiplier's MAGNITUDE, not to SHAP attributions.")
    else:
        print(f"  YES -- SHAP beats uniform in {n_beats_uniform} and shuffled in "
              f"{n_beats_shuffled} comparisons; the attributions carry real signal.")
    print("\nDone. Written to", OUT_DIR)
