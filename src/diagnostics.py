"""
diagnostics.py
Statistical assumption testing and data quality diagnostics.
Runs on raw data — no preprocessing applied here.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from scipy.stats import shapiro, mannwhitneyu, kruskal, ks_2samp
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

OUTPUT_DIR = Path("D:/Xai/outputs")


def shapiro_wilk_analysis(X: pd.DataFrame, y: pd.Series, name: str) -> pd.DataFrame:
    """
    Shapiro-Wilk normality test per feature, for each class.
    """
    rows = []
    for col in X.columns:
        g1 = X[col][y == 1].dropna().values
        g0 = X[col][y == 0].dropna().values
        std = X[col].std()
        if std == 0:
            rows.append({"Feature": col, "SW_p_buggy": np.nan,
                         "Normal_buggy": False, "SW_p_clean": np.nan,
                         "Normal_clean": False, "ZeroVar": True,
                         "Skewness": 0.0, "Kurtosis": 0.0})
            continue
        _, p1 = shapiro(g1[:5000]) if len(g1) >= 3 else (np.nan, np.nan)
        _, p0 = shapiro(g0[:5000]) if len(g0) >= 3 else (np.nan, np.nan)
        rows.append({
            "Feature":      col,
            "SW_p_buggy":   round(float(p1), 4) if not np.isnan(p1) else np.nan,
            "Normal_buggy": bool(p1 > 0.05)     if not np.isnan(p1) else False,
            "SW_p_clean":   round(float(p0), 4) if not np.isnan(p0) else np.nan,
            "Normal_clean": bool(p0 > 0.05)     if not np.isnan(p0) else False,
            "ZeroVar":      False,
            "Skewness":     round(float(X[col].skew()), 3),
            "Kurtosis":     round(float(X[col].kurtosis()), 3),
        })
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print(f"  Shapiro-Wilk — {name}")
    print(f"{'='*60}")
    print(f"  Non-normal (buggy class) : {(~df['Normal_buggy']).sum()}")
    print(f"  Non-normal (clean class) : {(~df['Normal_clean']).sum()}")
    print(f"  Zero-variance features   : {df['ZeroVar'].sum()}")
    return df


def multicollinearity_diagnostics(X: pd.DataFrame, name: str,
                                  output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """
    VIF, condition number, and correlation heatmap.
    """
    valid = X.loc[:, X.std() > 0].copy()
    print(f"\n{'='*60}")
    print(f"  Multicollinearity — {name}")
    print(f"{'='*60}")

    if valid.shape[1] < 2:
        print(f"  Only {valid.shape[1]} non-zero variance feature(s). VIF skipped.")
        return pd.DataFrame()

    vif_vals = [variance_inflation_factor(valid.values.astype(float), i)
                for i in range(valid.shape[1])]
    vif_df = pd.DataFrame({
        "Feature": valid.columns,
        "VIF": [round(v, 3) for v in vif_vals],
        "Severity": ["Severe" if v > 10 else ("Moderate" if v > 5 else "Low")
                     for v in vif_vals],
    })

    X_mat = add_constant(valid.values.astype(float))
    _, s, _ = np.linalg.svd(X_mat)
    cond = float(s.max() / s.min()) if s.min() > 1e-10 else np.inf
    cond_label = "Severe (>30)" if cond > 30 else ("Moderate (15-30)" if cond > 15 else "Low (<15)")
    print(f"  Condition Number : {cond:.2f}  [{cond_label}]")
    print(vif_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(max(6, valid.shape[1]), max(5, valid.shape[1] - 1)))
    sns.heatmap(valid.corr(), annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title(f"Correlation Heatmap — {name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f"corr_heatmap_{name.lower()}.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    return vif_df


def breusch_pagan_test(X: pd.DataFrame, y: pd.Series, name: str) -> dict:
    """
    Breusch-Pagan test for homoskedasticity using OLS residuals.
    Uses binary target as dependent variable (residual variance diagnostic).
    """
    valid = X.loc[:, X.std() > 0].copy()
    print(f"\n{'='*60}")
    print(f"  Breusch-Pagan Homoskedasticity — {name}")
    print(f"{'='*60}")

    if valid.shape[1] == 0:
        print("  No valid features. Test skipped.")
        return {}

    X_const = add_constant(valid.values.astype(float))
    ols_model = OLS(y.astype(float).values, X_const).fit()
    lm_stat, lm_p, f_stat, f_p = het_breuschpagan(ols_model.resid, X_const)

    result = {"LM_stat": round(lm_stat, 4), "LM_p": round(lm_p, 4),
               "F_stat": round(f_stat, 4), "F_p": round(f_p, 4),
               "Heteroskedastic": lm_p < 0.05}
    print(f"  LM stat={lm_stat:.4f}  p={lm_p:.4f}")
    print(f"  F  stat={f_stat:.4f}  p={f_p:.4f}")
    verdict = "HETEROSKEDASTIC (variance not constant)" if lm_p < 0.05 \
              else "HOMOSKEDASTIC (variance approximately constant)"
    print(f"  Result: {verdict}")
    return result


def skewness_kurtosis_summary(X: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Per-feature skewness, kurtosis, and outlier counts (IQR method).
    """
    rows = []
    for col in X.columns:
        vals = X[col].dropna().values
        q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
        iqr = q3 - q1
        n_out = int(((vals < q1 - 1.5*iqr) | (vals > q3 + 1.5*iqr)).sum())
        rows.append({
            "Feature":     col,
            "Mean":        round(float(np.mean(vals)), 4),
            "Std":         round(float(np.std(vals)), 4),
            "Skewness":    round(float(scipy_skew(vals)), 3),
            "Kurtosis":    round(float(scipy_kurtosis(vals)), 3),
            "Outliers(IQR)": n_out,
            "Outlier_%":   round(100 * n_out / max(len(vals), 1), 2),
            "NeedLog":     abs(scipy_skew(vals)) > 1.0,
        })
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print(f"  Skewness / Kurtosis / Outliers — {name}")
    print(f"{'='*60}")
    print(f"  Features needing log transform (|skew|>1): {df['NeedLog'].sum()}")
    return df


def class_imbalance_stats(y: pd.Series, name: str) -> dict:
    vc = y.value_counts().sort_index()
    n0, n1 = int(vc.get(0, 0)), int(vc.get(1, 0))
    ratio = n0 / max(n1, 1)
    print(f"\n{'='*60}")
    print(f"  Class Imbalance — {name}")
    print(f"{'='*60}")
    print(f"  Non-buggy (0) : {n0}  ({100*n0/(n0+n1):.1f}%)")
    print(f"  Buggy     (1) : {n1}  ({100*n1/(n0+n1):.1f}%)")
    print(f"  Ratio         : {ratio:.2f}:1")
    print(f"  SMOTE needed  : {'YES' if ratio >= 3.0 else 'No'}")
    return {"n0": n0, "n1": n1, "ratio": ratio, "smote_needed": ratio >= 3.0}


def feature_dependence_analysis(X: pd.DataFrame, y: pd.Series, name: str) -> pd.DataFrame:
    """
    Mann-Whitney U + Kruskal-Wallis per feature.
    Flags potential leakage if a single feature perfectly separates classes.
    """
    rows = []
    for col in X.columns:
        g1 = X[col][y == 1].values
        g0 = X[col][y == 0].values
        std = X[col].std()
        if std == 0:
            rows.append({"Feature": col, "MW_p": np.nan, "MW_sig": False,
                         "KW_p": np.nan, "KW_sig": False,
                         "LeakageRisk": "Low (zero variance)"})
            continue
        _, mw_p = mannwhitneyu(g1, g0, alternative="two-sided")
        _, kw_p = kruskal(g1, g0)
        overlap = len(set(g1) & set(g0))
        leakage = "HIGH — no overlap" if overlap == 0 else \
                  ("Moderate" if mw_p < 1e-10 else "Low")
        rows.append({
            "Feature":     col,
            "MW_p":        round(float(mw_p), 6),
            "MW_sig":      mw_p < 0.05,
            "KW_p":        round(float(kw_p), 6),
            "KW_sig":      kw_p < 0.05,
            "LeakageRisk": leakage,
        })
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print(f"  Feature Dependence & Leakage Check — {name}")
    print(f"{'='*60}")
    print(f"  Significant features (MW p<0.05) : {df['MW_sig'].sum()}")
    high_risk = df[df["LeakageRisk"].str.startswith("HIGH")]
    if not high_risk.empty:
        print(f"  HIGH leakage risk features: {high_risk['Feature'].tolist()}")
    return df


def distribution_comparison(X_train: pd.DataFrame, X_test: pd.DataFrame,
                             name: str) -> pd.DataFrame:
    """
    KS test per feature to detect train/test distribution shift.
    """
    rows = []
    for col in X_train.columns:
        stat, p = ks_2samp(X_train[col].values, X_test[col].values)
        rows.append({
            "Feature":   col,
            "KS_stat":   round(float(stat), 4),
            "p_value":   round(float(p), 4),
            "Shift":     "YES" if p < 0.05 else "No",
        })
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print(f"  Train/Test Distribution (KS Test) — {name}")
    print(f"{'='*60}")
    print(f"  Features with significant shift (p<0.05): "
          f"{(df['Shift']=='YES').sum()} / {len(df)}")
    return df
