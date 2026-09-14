"""
Phase 2: does SHAP-guided selection attend to redundant features?
A controlled synthetic study answering an open question posed by Antwarg et al.

MOTIVATION (their words, Information Fusion 96 (2023) 92-102, Section 5):
    "(4) examine the contribution of Shapley values further using a synthetic
    dataset, in which feature variance and noise are controlled. This allows
    exploration of whether Shapley values may be used to trace instances with
    problematic feature values, and whether features for which there is high
    variance in the Shapley values distribution are given more attention than
    redundant features."

Their implicit hypothesis is that Shapley-based methods attend to genuinely
informative features and away from redundant ones. On real data we observe the
opposite: SHAP-top-k sets sit at the 88th-97th percentile of random subsets by
internal correlation on all three 20-feature projects. Real data cannot settle
the question because the ground truth is unknown -- which features are
"genuinely" informative is not observable. Synthetic data fixes that:
sklearn's make_classification labels every feature by construction as
informative, redundant (a random linear combination of informative ones),
repeated (an exact duplicate), or noise.

DESIGN
  n_samples 1000, n_features 20, fixed across all configurations.
  Sweep the redundancy budget: for r in {0, 2, 4, 6, 8, 10}, allocate
      n_informative = 10 - r//2      (kept >= 5)
      n_redundant   = r
      n_repeated    = r // 2
      remainder     = pure noise
  3 replicate datasets (seeds 42, 43, 44) x 6 configurations = 18 datasets.
  Model: RandomForest (TreeExplainer, exact and fast). SHAP computed
  out-of-fold on the training split via pipeline.shap_weights_out_of_fold so
  the importance vector never sees held-out labels.

PRE-REGISTERED PREDICTIONS (frozen before running; see preregistration note)
  P1  As the redundancy budget rises, the proportion of the SHAP-top-k set that
      is redundant-or-repeated rises FASTER than the proportion of such features
      in the pool (i.e. SHAP over-selects redundancy relative to chance).
      Test: Spearman rho(redundancy budget, over-selection ratio) > 0, p < 0.05.
  P2  SHAP-guided top-k selection recovers FEWER distinct informative signals
      than redundancy-aware (mRMR) selection at equal k.
      Test: paired Wilcoxon over the 18 datasets, mRMR > SHAP, p < 0.05.
  P3  THEIR hypothesis, tested fairly: the across-instance variance of a
      feature's Shapley values is higher for informative than for redundant
      features. Test: paired Wilcoxon over the 18 datasets. We expect NO
      systematic difference; a positive result would support their conjecture.

Run: python synthetic_redundancy.py
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
N_FEATURES = 20
REDUNDANCY_BUDGETS = [0, 2, 4, 6, 8, 10]
REPLICATES = [42, 43, 44]
MODEL = "RandomForest"
MRMR_LAMBDA = 1.0


def make_labelled_dataset(budget, seed):
    """Build a dataset whose feature roles are known by construction.
    make_classification orders columns as: informative, redundant, repeated,
    then noise -- so roles can be assigned positionally."""
    n_inf = max(5, 10 - budget // 2)
    n_red = budget
    n_rep = budget // 2
    if n_inf + n_red + n_rep > N_FEATURES:
        n_rep = max(0, N_FEATURES - n_inf - n_red)
    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=N_FEATURES,
        n_informative=n_inf, n_redundant=n_red, n_repeated=n_rep,
        n_classes=2, shuffle=False, random_state=seed)
    roles = (["informative"] * n_inf + ["redundant"] * n_red +
             ["repeated"] * n_rep)
    roles += ["noise"] * (N_FEATURES - len(roles))
    cols = [f"f{i}" for i in range(N_FEATURES)]
    return pd.DataFrame(X, columns=cols), pd.Series(y), roles, n_inf


def mrmr_select(imp, corr, k, lam=MRMR_LAMBDA):
    """Greedy: relevance minus lambda * mean |corr| with the already-selected."""
    chosen = [int(np.argmax(imp))]
    while len(chosen) < k:
        best, best_score = None, -np.inf
        for j in range(len(imp)):
            if j in chosen:
                continue
            penalty = np.mean([abs(corr[j, c]) for c in chosen])
            s = imp[j] - lam * penalty
            if s > best_score:
                best, best_score = j, s
        chosen.append(best)
    return np.array(chosen)


def run():
    print("=== Phase 2: synthetic redundancy study (answers Antwarg et al. FW4) ===")
    rows = []

    for budget in REDUNDANCY_BUDGETS:
        for seed in REPLICATES:
            X, y, roles, n_inf = make_labelled_dataset(budget, seed)
            roles = np.array(roles)
            X_tr, _, y_tr, _ = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=P.SEED)

            tuned = {MODEL: {"n_estimators": 200, "min_samples_leaf": 2}}
            imp = P.shap_weights_out_of_fold(X_tr, y_tr, MODEL, tuned)
            imp = P.normalise(imp)

            # per-feature across-instance Shapley variance (for P3) via a
            # single fit on the training split
            pipe = P._fit_full(X_tr, y_tr, MODEL, tuned)
            Xp = P.transform_only(pipe, X_tr.values)
            sv = P.compute_shap_values(pipe, Xp, Xp, MODEL)
            shap_var = np.var(np.asarray(sv), axis=0)

            k = N_FEATURES // 2
            corr = np.nan_to_num(np.abs(np.corrcoef(X_tr.values.T)), nan=0.0)
            top_shap = np.argsort(-imp)[:k]
            top_mrmr = mrmr_select(imp, corr, k)

            redundantish = np.isin(roles, ["redundant", "repeated"])
            pool_frac = redundantish.mean()
            shap_frac = redundantish[top_shap].mean()
            mrmr_frac = redundantish[top_mrmr].mean()
            # >1 means SHAP picks redundancy more often than the pool implies
            over_sel = shap_frac / pool_frac if pool_frac > 0 else np.nan

            rows.append({
                "budget": budget, "seed": seed, "n_informative": n_inf,
                "n_redundant_pool": int(redundantish.sum()),
                "pool_frac_redundant": pool_frac,
                "shap_frac_redundant": shap_frac,
                "mrmr_frac_redundant": mrmr_frac,
                "shap_over_selection": over_sel,
                "shap_n_informative_recovered": int((roles[top_shap] == "informative").sum()),
                "mrmr_n_informative_recovered": int((roles[top_mrmr] == "informative").sum()),
                "shap_set_mean_abs_corr": float(np.mean(
                    [abs(corr[i, j]) for ii, i in enumerate(top_shap) for j in top_shap[ii + 1:]])),
                "mrmr_set_mean_abs_corr": float(np.mean(
                    [abs(corr[i, j]) for ii, i in enumerate(top_mrmr) for j in top_mrmr[ii + 1:]])),
                "shap_var_informative": float(np.mean(shap_var[roles == "informative"])),
                "shap_var_redundant": float(np.mean(shap_var[redundantish]))
                if redundantish.any() else np.nan,
            })
            print(f"  budget={budget:>2} seed={seed}  pool_red={pool_frac:.2f} "
                  f"shap_red={shap_frac:.2f} over={over_sel if np.isnan(over_sel) else round(over_sel,2)} "
                  f"inf_recovered shap={rows[-1]['shap_n_informative_recovered']} "
                  f"mrmr={rows[-1]['mrmr_n_informative_recovered']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "synthetic_redundancy.csv", index=False)

    print("\n--- Aggregated by redundancy budget ---")
    agg = df.groupby("budget")[["pool_frac_redundant", "shap_frac_redundant",
                                "mrmr_frac_redundant", "shap_over_selection",
                                "shap_n_informative_recovered",
                                "mrmr_n_informative_recovered"]].mean()
    print(agg.to_string())

    # ---- P1 ----
    sub = df.dropna(subset=["shap_over_selection"])
    r1 = spearmanr(sub["budget"], sub["shap_over_selection"])
    p1 = bool(r1.statistic > 0 and r1.pvalue < 0.05)

    # ---- P2 ----
    a = df["mrmr_n_informative_recovered"].values
    b = df["shap_n_informative_recovered"].values
    if np.all(a == b):
        p2_p, p2_stat = 1.0, 0.0
    else:
        p2_stat, p2_p = wilcoxon(a, b)
    p2 = bool(a.mean() > b.mean() and p2_p < 0.05)

    # ---- P3 (their hypothesis) ----
    v = df.dropna(subset=["shap_var_redundant"])
    if len(v) and not np.allclose(v["shap_var_informative"], v["shap_var_redundant"]):
        p3_stat, p3_p = wilcoxon(v["shap_var_informative"], v["shap_var_redundant"])
    else:
        p3_stat, p3_p = 0.0, 1.0
    p3 = bool(v["shap_var_informative"].mean() > v["shap_var_redundant"].mean() and p3_p < 0.05)

    print("\n--- PRE-REGISTERED PREDICTIONS ---")
    print(f"  P1 SHAP over-selects redundancy as budget rises: rho={r1.statistic:+.3f} "
          f"p={r1.pvalue:.4f} -> {'SUPPORTED' if p1 else 'NOT supported'}")
    print(f"  P2 mRMR recovers more informative features than SHAP: "
          f"mrmr={a.mean():.2f} vs shap={b.mean():.2f} p={p2_p:.4f} -> "
          f"{'SUPPORTED' if p2 else 'NOT supported'}")
    print(f"  P3 (Antwarg conjecture) Shapley variance higher for informative "
          f"than redundant: {v['shap_var_informative'].mean():.3e} vs "
          f"{v['shap_var_redundant'].mean():.3e} p={p3_p:.4f} -> "
          f"{'SUPPORTED' if p3 else 'NOT supported'}")

    pd.DataFrame([{
        "P1_rho": r1.statistic, "P1_p": r1.pvalue, "P1_supported": p1,
        "P2_mrmr_mean": a.mean(), "P2_shap_mean": b.mean(), "P2_p": p2_p,
        "P2_supported": p2,
        "P3_var_informative": v["shap_var_informative"].mean(),
        "P3_var_redundant": v["shap_var_redundant"].mean(), "P3_p": p3_p,
        "P3_supported": p3,
    }]).to_csv(OUT_DIR / "synthetic_redundancy_verdict.csv", index=False)
    return df


if __name__ == "__main__":
    run()
    print("\nDone. Written to", OUT_DIR)
