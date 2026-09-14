"""
Statistical power for the null result: what could this design actually have detected?

WHY THIS IS REQUIRED. The central claim of the study is a NULL -- SHAP-derived importance
is indistinguishable from magnitude-matched controls. A null is only meaningful if the
design had the power to detect an effect had one existed; otherwise "no significant
difference" reduces to "we did not look hard enough". The positive control (CORAL
achieving 51 significant wins in the same harness) already argues this qualitatively.
This file makes it quantitative.

METHOD. For every (dataset, model, metric) cell we have 20 paired differences between the
SHAP-weighted variant and its control. Rather than assume normality, we resample the
OBSERVED difference distribution -- preserving its real shape, skew and variance -- and
shift it by a candidate true effect delta:

    d_centred = d - mean(d)                        # null-centred, real noise structure
    for each candidate delta:
        draw B bootstrap samples of size 20 from (d_centred + delta)
        run the same paired Wilcoxon used in the study
        power(delta) = fraction of samples with p < 0.05

The minimum detectable effect (MDE) is the smallest delta reaching 80% power. Comparing
the MDE against the observed effect tells us whether the null is informative: if the MDE
is far below any effect worth caring about, the design was capable and the null stands.

We report the MDE against two reference points:
  - the observed |effect| for SHAP vs its controls (should be far BELOW the MDE)
  - the effect CORAL achieved in the same harness (should be far ABOVE the MDE, since it
    was in fact detected)

Run: python power_analysis.py
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import pipeline as P

warnings.filterwarnings("ignore")

OUT_DIR = P.OUT_DIR
N_BOOT = 1000
ALPHA = 0.05
TARGET_POWER = 0.80
DELTA_GRID = np.round(np.arange(0.0025, 0.1001, 0.0025), 4)
RNG = np.random.RandomState(P.SEED)


def power_at(d_centred, delta, n_boot=N_BOOT):
    """Fraction of bootstrap resamples in which a shifted copy of the observed
    difference distribution is detected by the paired Wilcoxon test."""
    n = len(d_centred)
    hits = 0
    for _ in range(n_boot):
        s = RNG.choice(d_centred, size=n, replace=True) + delta
        if np.allclose(s, 0):
            continue
        try:
            if wilcoxon(s).pvalue < ALPHA:
                hits += 1
        except ValueError:
            pass
    return hits / n_boot


def mde(d_centred):
    """Smallest delta on the grid reaching TARGET_POWER; NaN if none does."""
    for delta in DELTA_GRID:
        if power_at(d_centred, delta) >= TARGET_POWER:
            return delta
    return np.nan


def collect_pairs(df, arm_a, arm_b, keys, metrics=("f1", "auc", "prauc")):
    out = []
    for key, g in df.groupby(list(keys)):
        g = g.sort_values([c for c in ("seed", "boot_seed") if c in g.columns][0])
        for m in metrics:
            a = g[g.arm == arm_a][m].values if "arm" in g.columns else None
            b = g[g.arm == arm_b][m].values if "arm" in g.columns else None
            if a is None or b is None or len(a) != len(b) or len(a) < 5:
                continue
            out.append({"cell": " / ".join(map(str, key if isinstance(key, tuple) else (key,))),
                        "metric": m, "diff": a - b})
    return out


def run():
    print("=== Power analysis for the null result ===")
    print(f"    {N_BOOT} bootstrap resamples per delta, alpha={ALPHA}, "
          f"target power={TARGET_POWER:.0%}, n=20 paired splits\n")

    abl = pd.read_csv(OUT_DIR / "ablation_scores.csv")
    variants = set(abl.variant.unique())
    abl = abl.rename(columns={"variant": "arm"})
    print(f"  ablation arms found: {sorted(variants)}")

    rows = []
    for ctrl in ["uniform_sum1", "shuffled_shap"]:
        pairs = collect_pairs(abl, "shap_sum1", ctrl,
                              keys=[c for c in ["ecosystem", "dataset", "model"] if c in abl.columns])
        print(f"  computing MDE for shap vs {ctrl} ({len(pairs)} cells)...")
        for rec in pairs:
            d = rec["diff"]
            rows.append({"comparison": f"shap_vs_{ctrl}", "cell": rec["cell"],
                         "metric": rec["metric"], "n": len(d),
                         "observed_effect": d.mean(), "sd_of_diff": d.std(ddof=1),
                         "mde_80": mde(d - d.mean())})

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "power_analysis.csv", index=False)

    print("\n--- Minimum detectable effect at 80% power ---")
    print(res.groupby(["comparison", "metric"]).agg(
        median_MDE=("mde_80", "median"),
        median_abs_observed=("observed_effect", lambda x: np.median(np.abs(x))),
        cells=("mde_80", "size")).round(4).to_string())

    med_mde = res.mde_80.median()
    med_obs = np.median(np.abs(res.observed_effect))
    print(f"\n  Overall median MDE            : {med_mde:.4f}")
    print(f"  Overall median |observed effect|: {med_obs:.4f}")
    print(f"  Ratio (MDE / observed)         : {med_mde/med_obs:.1f}x" if med_obs > 0 else "")

    # ---- reference point: what CORAL achieved in the same harness ----
    try:
        cp = pd.read_csv(OUT_DIR / "cpdp_significance.csv")
        coral = cp[(cp.arm == "coral_aligned") & (cp["significant_at_0.05"]) & (cp.mean_diff > 0)]
        coral_eff = coral.mean_diff.median()
        print(f"\n  Reference: CORAL's median detected effect in the same harness = {coral_eff:.4f}")
        print(f"  CORAL effect is {coral_eff/med_mde:.1f}x the MDE -> detectable, and was detected.")
    except Exception as e:
        coral_eff = np.nan
        print(f"  (CORAL reference unavailable: {e})")

    informative = bool(med_mde < 0.05 and med_obs < med_mde)
    print("\n=== VERDICT ===")
    if informative:
        print(f"  The null is INFORMATIVE. The design detects a true difference of")
        print(f"  ~{med_mde:.3f} with 80% power, while the observed SHAP-vs-control effect")
        print(f"  is {med_obs:.4f} -- roughly {med_mde/med_obs:.0f}x smaller than the detection")
        print(f"  floor. An effect large enough to matter would have been found.")
    else:
        print("  The design is UNDERPOWERED for the effect sizes in question; the null")
        print("  should be reported as inconclusive rather than as evidence of absence.")

    pd.DataFrame([{"median_MDE_80": med_mde, "median_abs_observed_effect": med_obs,
                   "ratio_mde_to_observed": med_mde / med_obs if med_obs > 0 else np.nan,
                   "coral_median_detected_effect": coral_eff,
                   "n_cells": len(res), "n_boot": N_BOOT,
                   "null_is_informative": informative}]).to_csv(
        OUT_DIR / "power_analysis_verdict.csv", index=False)
    return res


if __name__ == "__main__":
    run()
    print("\nDone. Written to", OUT_DIR)
