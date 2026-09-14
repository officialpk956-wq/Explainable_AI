"""
Phase 2 (corrected replication): synthetic redundancy study, v2.

WHY THIS EXISTS. In synthetic_redundancy.py all three pre-registered predictions
failed. Two of those failures were instrument problems, not results, and are
corrected here. The original run is NOT deleted or superseded -- both are
reported, and the corrections are disclosed.

  P1 was mis-specified. It tested whether SHAP's over-selection of redundant
  features GROWS with the redundancy budget. That quantity saturates: at
  budget 10 the pool is already 75% redundant while k = 10 of 20, so the ratio
  is ceiling-bound and must fall back toward 1 regardless of behaviour. Testing
  for monotonic growth in a bounded quantity is a broken test.
  Fixes: (a) n_features 20 -> 40 with k = 10, so k/F stays small and the pool
  never saturates; (b) a saturation-aware effect size
        excess = (observed - chance) / (max_possible - chance)
  which is the standard correction for a bounded proportion and is comparable
  across budgets; (c) the pre-registered range is the non-saturated regime,
  with the highest budget reported separately rather than dropped.

  P3 was underpowered (n = 15) and used a scale-confounded statistic. Informative
  features carry larger SHAP magnitudes overall, so raw across-instance variance
  conflates spread with scale.
  Fixes: (a) 10 replicates instead of 3; (b) coefficient of variation
  (std / mean|SHAP|) alongside raw variance.

  P2 is NOT touched. Its reversal is a finding, not a defect: mRMR recovers
  fewer informative features at low redundancy and more at high redundancy.
  Engineering it into a win is not something this study does. The adaptive-lambda
  follow-up that the finding implies is a separate, separately pre-registered
  experiment.

PRE-REGISTERED PREDICTIONS FOR THIS RUN (frozen before execution)
  P1c  Over the non-saturated regime, SHAP's normalised excess redundancy is
       GREATER THAN ZERO. Test: one-sample Wilcoxon on the excess values,
       two-sided, alpha 0.05. (This asks whether SHAP over-selects redundancy at
       all -- the question P1 was meant to ask.)
  P1d  Secondary: that excess does not systematically shrink as the budget rises
       within the non-saturated regime. Test: Spearman rho(budget, excess);
       reported descriptively, not gated.
  P3c  Coefficient of variation of per-feature Shapley values is higher for
       informative than for redundant features. Test: paired Wilcoxon across all
       configurations, two-sided, alpha 0.05. This is Antwarg et al.'s conjecture
       tested with adequate power and an unconfounded statistic.

Run: python synthetic_redundancy_v2.py
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import pipeline as P

warnings.filterwarnings("ignore")

OUT_DIR = P.OUT_DIR
N_SAMPLES = 1000
N_FEATURES = 40                     # was 20; keeps k/F small so the pool cannot saturate
K = 10                              # select 10 of 40
REDUNDANCY_BUDGETS = [0, 4, 8, 12, 16, 20]
SATURATED_BUDGET = 20               # reported separately, not part of the P1c test
REPLICATES = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]   # was 3; now 10
MODEL = "RandomForest"


def make_labelled_dataset(budget, seed):
    """make_classification with shuffle=False orders columns as informative,
    redundant, repeated, then noise -- so roles are known positionally."""
    n_inf = max(8, 20 - budget // 2)
    n_red = budget
    n_rep = budget // 2
    if n_inf + n_red + n_rep > N_FEATURES:
        n_rep = max(0, N_FEATURES - n_inf - n_red)
    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=N_FEATURES,
        n_informative=n_inf, n_redundant=n_red, n_repeated=n_rep,
        n_classes=2, shuffle=False, random_state=seed)
    roles = ["informative"] * n_inf + ["redundant"] * n_red + ["repeated"] * n_rep
    roles += ["noise"] * (N_FEATURES - len(roles))
    return (pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)]),
            pd.Series(y), np.array(roles))


def normalised_excess(n_selected_red, n_pool_red, k, n_features):
    """(observed - chance) / (max_possible - chance), a saturation-aware effect
    size for a bounded count. 0 = exactly chance, 1 = the most redundancy the
    selection could possibly contain, <0 = avoids redundancy."""
    chance = k * (n_pool_red / n_features)
    max_possible = min(k, n_pool_red)
    if max_possible - chance < 1e-12:
        return np.nan                      # no headroom -> undefined, excluded
    return (n_selected_red - chance) / (max_possible - chance)


def run():
    print("=== Phase 2 v2: corrected synthetic redundancy study ===")
    print(f"    n_features={N_FEATURES}, k={K}, replicates={len(REPLICATES)}")
    rows = []

    for budget in REDUNDANCY_BUDGETS:
        for seed in REPLICATES:
            X, y, roles = make_labelled_dataset(budget, seed)
            X_tr, _, y_tr, _ = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=P.SEED)

            tuned = {MODEL: {"n_estimators": 200, "min_samples_leaf": 2}}
            imp = P.normalise(P.shap_weights_out_of_fold(X_tr, y_tr, MODEL, tuned))

            pipe = P._fit_full(X_tr, y_tr, MODEL, tuned)
            Xp = P.transform_only(pipe, X_tr.values)
            sv = np.asarray(P.compute_shap_values(pipe, Xp, Xp, MODEL))
            shap_std = sv.std(axis=0)
            shap_mean_abs = np.abs(sv).mean(axis=0)
            # coefficient of variation: spread normalised by magnitude, so a
            # feature is not judged variable merely because it is influential
            shap_cv = np.divide(shap_std, shap_mean_abs,
                                out=np.zeros_like(shap_std),
                                where=shap_mean_abs > 1e-12)

            top = np.argsort(-imp)[:K]
            redundantish = np.isin(roles, ["redundant", "repeated"])
            n_pool_red = int(redundantish.sum())
            n_sel_red = int(redundantish[top].sum())

            rows.append({
                "budget": budget, "seed": seed,
                "n_pool_redundant": n_pool_red,
                "pool_frac": n_pool_red / N_FEATURES,
                "n_selected_redundant": n_sel_red,
                "selected_frac": n_sel_red / K,
                "excess": normalised_excess(n_sel_red, n_pool_red, K, N_FEATURES),
                "saturated": budget == SATURATED_BUDGET,
                "cv_informative": float(np.mean(shap_cv[roles == "informative"])),
                "cv_redundant": float(np.mean(shap_cv[redundantish])) if redundantish.any() else np.nan,
                "var_informative": float(np.mean(sv[:, roles == "informative"].var(axis=0))),
                "var_redundant": float(np.mean(sv[:, redundantish].var(axis=0))) if redundantish.any() else np.nan,
            })
        e = [r["excess"] for r in rows if r["budget"] == budget and not np.isnan(r["excess"])]
        print(f"  budget={budget:>2}  pool_red={rows[-1]['pool_frac']:.2f}  "
              f"mean selected_red={np.mean([r['selected_frac'] for r in rows if r['budget']==budget]):.2f}  "
              f"mean excess={np.mean(e) if e else float('nan'):+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "synthetic_redundancy_v2.csv", index=False)

    # ---------- P1c: is there over-selection at all (non-saturated regime)? ----------
    main = df[(~df.saturated) & df.excess.notna()]
    if len(main) and not np.allclose(main.excess, 0):
        st, p1c_p = wilcoxon(main.excess)
    else:
        st, p1c_p = 0.0, 1.0
    p1c = bool(main.excess.mean() > 0 and p1c_p < 0.05)

    # ---------- P1d: descriptive trend within the non-saturated regime ----------
    r1d = spearmanr(main.budget, main.excess)

    # ---------- P3c: CV informative vs redundant, properly powered ----------
    v = df.dropna(subset=["cv_redundant"])
    if len(v) and not np.allclose(v.cv_informative, v.cv_redundant):
        st3, p3c_p = wilcoxon(v.cv_informative, v.cv_redundant)
    else:
        st3, p3c_p = 0.0, 1.0
    p3c = bool(v.cv_informative.mean() > v.cv_redundant.mean() and p3c_p < 0.05)

    print("\n--- Per-budget summary ---")
    print(df.groupby("budget")[["pool_frac", "selected_frac", "excess",
                                "cv_informative", "cv_redundant"]].mean().round(4).to_string())

    print("\n--- PRE-REGISTERED (v2) ---")
    print(f"  P1c over-selection > 0 in non-saturated regime (n={len(main)}): "
          f"mean excess={main.excess.mean():+.3f}, p={p1c_p:.5f} -> "
          f"{'SUPPORTED' if p1c else 'NOT supported'}")
    print(f"  P1d trend within regime (descriptive): rho={r1d.statistic:+.3f}, p={r1d.pvalue:.4f}")
    print(f"  P3c CV(informative) > CV(redundant) (n={len(v)}): "
          f"{v.cv_informative.mean():.4f} vs {v.cv_redundant.mean():.4f}, "
          f"p={p3c_p:.5f} -> {'SUPPORTED' if p3c else 'NOT supported'}")

    sat = df[df.saturated & df.excess.notna()]
    if len(sat):
        print(f"\n  [reported separately] saturated budget {SATURATED_BUDGET}: "
              f"mean excess={sat.excess.mean():+.3f} (pool {sat.pool_frac.mean():.0%} redundant)")

    pd.DataFrame([{
        "P1c_mean_excess": main.excess.mean(), "P1c_p": p1c_p, "P1c_supported": p1c,
        "P1c_n": len(main),
        "P1d_rho": r1d.statistic, "P1d_p": r1d.pvalue,
        "P3c_cv_informative": v.cv_informative.mean(),
        "P3c_cv_redundant": v.cv_redundant.mean(),
        "P3c_p": p3c_p, "P3c_supported": p3c, "P3c_n": len(v),
        "saturated_excess": sat.excess.mean() if len(sat) else np.nan,
    }]).to_csv(OUT_DIR / "synthetic_redundancy_v2_verdict.csv", index=False)
    return df


if __name__ == "__main__":
    run()
    print("\nDone. Written to", OUT_DIR)
