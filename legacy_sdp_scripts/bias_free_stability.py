"""
Phase I + J: a stability metric free of LIME's linear-surrogate bias, and a test
of whether it changes the model ranking.

WHY. The Phase-G Explainability Score ranks LogisticRegression first on both
projects, but its stability component is sigma_bar -- the variance of LIME's
weights across seeds. LIME fits a LINEAR local surrogate, and measured surrogate
R^2 is 0.92-0.96 for LogisticRegression versus 0.18-0.59 for everything else.
So a linear model may score as "stable" simply because LIME can fit it, not
because its explanations are genuinely more reliable. That caveat is unresolved.

PHASE I. Local Lipschitz estimate (Alvarez-Melis & Jaakkola, 2018): for an
instance x, sample neighbours x' in a small ball and take

    L(x) = max_{x'}  || attr(x) - attr(x') ||_2  /  || x - x' ||_2

Low L means the attribution does not swing wildly under tiny input changes. It
is computed here on SHAP values, which involve no surrogate model at all, so it
cannot be gamed by matching LIME's linear assumption.

PHASE J. Rank the 8 classifiers three ways -- LIME sigma_bar (potentially
biased), SHAP-Lipschitz (bias-free), AOPC faithfulness (bias-free) -- and
measure agreement with Kendall's tau. If the LIME-based ranking disagrees with
both bias-free rankings, then model selection based on LIME stability reaches
different conclusions than bias-free measures.

Run: python bias_free_stability.py
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split

import pipeline as P

warnings.filterwarnings("ignore")

OUT_DIR = P.OUT_DIR
N_NEIGHBORS = 5      # perturbations per instance
EPS = 0.10           # neighbourhood radius, as a fraction of each feature's train std


def local_lipschitz_shap(pipe, X_tr_p, X_te_p, idxs, model_name,
                         eps=EPS, n_neighbors=N_NEIGHBORS, seed=P.SEED):
    """Mean over instances of the max attribution-change / input-change ratio.
    All neighbours of an instance are batched into a single SHAP call."""
    rng = np.random.RandomState(seed)
    scale = X_tr_p.std(axis=0)
    scale = np.where(scale < 1e-12, 1e-12, scale)
    per_instance = []

    for i in idxs:
        x = X_te_p[i]
        noise = rng.normal(0, eps, size=(n_neighbors, len(x))) * scale
        batch = np.vstack([x.reshape(1, -1), x + noise])
        attrs = np.asarray(P.compute_shap_values(pipe, X_tr_p, batch, model_name), dtype=float)
        a0, an = attrs[0], attrs[1:]
        num = np.linalg.norm(an - a0, axis=1)
        den = np.linalg.norm(batch[1:] - x, axis=1)
        den = np.where(den < 1e-12, np.nan, den)
        ratios = num / den
        if np.all(np.isnan(ratios)):
            continue
        per_instance.append(np.nanmax(ratios))

    return float(np.mean(per_instance)) if per_instance else np.nan


def phase_i_bias_free_stability():
    print("=== Phase I: bias-free (SHAP local-Lipschitz) stability for all 8 classifiers ===")
    rows = []
    for name in P.PHASE1:
        X, y, cols = P.load_xy(name)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=P.SEED)
        cv = pd.read_csv(OUT_DIR / f"phase1_cv_{name}.csv")
        best = cv.iloc[0]["Model"]

        # same instances every model is judged on, as in Phase F/G
        base = P.xai_core(X_train, X_test, y_train, best, None)
        idxs = base["idxs"]
        print(f"\n  [{name}] {len(idxs)} shared instances, {N_NEIGHBORS} neighbours each, eps={EPS}")

        for model_name in P.make_classifiers():
            pipe = P._fit_full(X_train, y_train, model_name, None)
            X_tr_p = P.transform_only(pipe, X_train.values)
            X_te_p = P.transform_only(pipe, X_test.values)
            L = local_lipschitz_shap(pipe, X_tr_p, X_te_p, idxs, model_name)
            rows.append({"dataset": name, "model": model_name, "shap_lipschitz": L})
            print(f"    {model_name:<19} SHAP-Lipschitz = {L:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "bias_free_stability.csv", index=False)
    return df


def phase_j_ranking_agreement(lip_df):
    print("\n=== Phase J: do the biased and bias-free rankings agree? ===")
    expl = pd.read_csv(OUT_DIR / "explainability_scores.csv")
    r2 = pd.read_csv(OUT_DIR / "surrogate_bias.csv")[
        ["dataset", "model", "lime_surrogate_r2"]]
    df = (expl[["dataset", "model", "stability_sigma_bar", "faithfulness_gain",
                "explainability_score", "F1_mean"]]
          .merge(lip_df, on=["dataset", "model"])
          .merge(r2, on=["dataset", "model"]))

    # rank 1 = best on each criterion (low sigma_bar, low Lipschitz, high AOPC)
    out = []
    for name, g in df.groupby("dataset"):
        g = g.copy()
        g["rank_lime_sigma"] = g["stability_sigma_bar"].rank()
        g["rank_shap_lipschitz"] = g["shap_lipschitz"].rank()
        g["rank_faithfulness"] = g["faithfulness_gain"].rank(ascending=False)
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    df.to_csv(OUT_DIR / "ranking_agreement.csv", index=False)

    print("\n--- Rankings per dataset (1 = best) ---")
    print(df[["dataset", "model", "rank_lime_sigma", "rank_shap_lipschitz",
              "rank_faithfulness", "lime_surrogate_r2"]].to_string(index=False))

    print("\n--- Kendall tau between rankings ---")
    tau_rows = []
    pairs = [("LIME sigma_bar", "rank_lime_sigma", "SHAP-Lipschitz", "rank_shap_lipschitz"),
             ("LIME sigma_bar", "rank_lime_sigma", "AOPC faithfulness", "rank_faithfulness"),
             ("SHAP-Lipschitz", "rank_shap_lipschitz", "AOPC faithfulness", "rank_faithfulness")]
    for name, g in df.groupby("dataset"):
        for la, ca, lb, cb in pairs:
            t = kendalltau(g[ca], g[cb])
            tau_rows.append({"dataset": name, "ranking_a": la, "ranking_b": lb,
                             "kendall_tau": t.statistic, "p_value": t.pvalue})
            print(f"  {name:<8} {la:<18} vs {lb:<18} tau={t.statistic:+.3f}  p={t.pvalue:.3f}")
    pd.DataFrame(tau_rows).to_csv(OUT_DIR / "ranking_agreement_tau.csv", index=False)

    print("\n--- Does the LIME-stability bias change who wins? ---")
    for name, g in df.groupby("dataset"):
        w_lime = g.loc[g["rank_lime_sigma"].idxmin(), "model"]
        w_lip = g.loc[g["rank_shap_lipschitz"].idxmin(), "model"]
        w_faith = g.loc[g["rank_faithfulness"].idxmin(), "model"]
        r2_lime_winner = float(g.loc[g["model"] == w_lime, "lime_surrogate_r2"].iloc[0])
        print(f"  {name}: LIME-stable winner = {w_lime} (surrogate R^2={r2_lime_winner:.2f}) | "
              f"bias-free stable winner = {w_lip} | most faithful = {w_faith}")
        if w_lime != w_lip:
            print(f"      -> the two stability measures DISAGREE on the winner")
        else:
            print(f"      -> both stability measures agree")

    # Direct test of the Phase-H idea, now on the bias-free metric: if LIME
    # sigma_bar were purely surrogate fit, the bias-free metric should NOT show
    # the same relationship to R^2.
    print("\n--- Is the bias-free metric also related to surrogate fit? (it should not be) ---")
    from scipy.stats import spearmanr
    for scope in list(P.PHASE1) + ["POOLED"]:
        g = df if scope == "POOLED" else df[df.dataset == scope]
        r_lime = spearmanr(g["lime_surrogate_r2"], g["stability_sigma_bar"])
        r_lip = spearmanr(g["lime_surrogate_r2"], g["shap_lipschitz"])
        print(f"  {scope:<8} rho(R^2, LIME sigma_bar)={r_lime.statistic:+.3f} (p={r_lime.pvalue:.3f})  |  "
              f"rho(R^2, SHAP-Lipschitz)={r_lip.statistic:+.3f} (p={r_lip.pvalue:.3f})")
    return df


if __name__ == "__main__":
    lip = phase_i_bias_free_stability()
    phase_j_ranking_agreement(lip)
    print("\nDone. Written to", OUT_DIR)
