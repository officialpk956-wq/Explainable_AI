"""
Verification Harness for Contract Checks (verify_outputs.py)
ASCII ONLY PRINT STATEMENTS.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

CONTRACTS = {
    "outputs/derivation_fit.csv": {
        "min_rows": 1, "max_rows": 1,
        "required_columns": ["b", "ci_lower", "ci_upper", "fit_r2"],
        "min_bytes": 60,
    },
    "outputs/reconcile_selection_scores.csv": {
        "min_rows": 4800, "max_rows": 4800,
        "required_columns": ["dataset", "model", "seed", "variant", "f1", "auc", "prauc"],
        "required_values": {
            "variant": ["all_features", "plain_shap_select", "mrmr_shap", "random_select", "shuffled_mrmr"],
            "dataset": ["equinox", "lucene", "pde", "xalan-2.6", "poi-3.0", "velocity-1.6"]
        },
        "no_nan_columns": ["f1", "auc", "prauc"],
        "min_bytes": 500,
    },
    "outputs/reconcile_selection_tests.csv": {
        "min_rows": 720, "max_rows": 720,
        "required_columns": ["dataset", "model", "metric", "comparison", "mean_diff", "wilcoxon_p", "holm_corrected_p", "significant_at_0.05"],
        "required_values": {
            "metric": ["f1", "auc", "prauc"]
        },
        "min_bytes": 500,
    },
    "outputs/reconcile_verdict.csv": {
        "min_rows": 1, "max_rows": 1,
        "required_columns": ["run_a_corroborated", "root_cause", "R1", "R2", "plain_losses", "mrmr_losses", "mrmr_wins_random", "mrmr_wins_shuffled"],
        "min_bytes": 50,
    },
    "outputs/aopc_full_grid.csv": {
        "min_rows": 1600, "max_rows": 1600,
        "required_columns": ["dataset", "model", "instance", "aopc_shap", "aopc_random", "faith_gain"],
        "required_values": {
            "dataset": ["eclipse", "mylyn", "equinox", "lucene", "pde", "ant-1.7", "camel-1.6", "xalan-2.6", "poi-3.0", "velocity-1.6"]
        },
        "no_nan_columns": ["aopc_shap", "aopc_random", "faith_gain"],
        "min_bytes": 500,
    },
    "outputs/deconfounded_stability.csv": {
        "min_rows": 160, "max_rows": 160,
        "required_columns": ["dataset", "model", "sigma_all", "r2_all", "sigma_star", "rank_raw", "rank_deconfounded", "rank_changed"],
        "required_values": {
            "dataset": ["eclipse", "mylyn", "equinox", "lucene", "pde", "ant-1.7", "camel-1.6", "xalan-2.6", "poi-3.0", "velocity-1.6"]
        },
        "min_bytes": 500,
    },
    "outputs/deconfounded_validation.csv": {
        "min_rows": 10, "max_rows": 11,
        "required_columns": ["dataset", "rho_sigma_raw_vs_faith", "rho_sigma_star_vs_faith", "improved"],
        "min_bytes": 200,
    },
    "outputs/deconfounded_verdict.csv": {
        "min_rows": 1, "max_rows": 1,
        "required_columns": ["n_improved", "wilcoxon_p", "claim_supported"],
        "min_bytes": 30,
    },
    "outputs/stability_convergence.csv": {
        "min_rows": 40, "max_rows": 40,
        "required_columns": ["dataset", "model", "num_samples", "sigma_bar"],
        "required_values": {
            "num_samples": [500, 1000, 2500, 5000, 10000],
            "model": ["LogisticRegression", "SVC", "RandomForest", "XGBoost"]
        },
        "no_nan_columns": ["sigma_bar"],
        "min_bytes": 500,
    },
    "outputs/convergence_rank_swaps.csv": {
        "min_rows": 0, "max_rows": 1000,
        "required_columns": ["dataset", "budget_a", "budget_b", "model_i", "model_j"],
        "min_bytes": 40,
    },
    "outputs/redundancy_all_targets.csv": {
        "min_rows": 6, "max_rows": 6,
        "required_columns": ["dataset", "n_features", "k", "shap_set_mean_abs_corr", "random_mean", "random_std", "percentile"],
        "required_values": {
            "dataset": ["equinox", "lucene", "pde", "xalan-2.6", "poi-3.0", "velocity-1.6"]
        },
        "no_nan_columns": ["shap_set_mean_abs_corr", "random_mean", "random_std", "percentile"],
        "min_bytes": 200,
    },
    "outputs/dose_response_raw.csv": {
        "min_rows": 2040, "max_rows": 2040,
        "required_columns": ["dataset", "model", "seed", "variant", "c", "f1", "auc"],
        "required_values": {
            "dataset": ["pde", "xalan-2.6"],
            "model": ["LogisticRegression", "SVC", "RandomForest"]
        },
        "no_nan_columns": ["f1", "auc"],
        "min_bytes": 1000,
    },
    "outputs/dose_response.png": {
        "min_bytes": 10000,
        "is_binary": True
    },
    "outputs/cpdp_zeroshot_scores.csv": {
        "min_rows": 3840, "max_rows": 3840,
        "required_columns": ["ecosystem", "target", "model", "arm", "boot_seed", "f1", "auc", "prauc"],
        "required_values": {
            "arm": ["original", "shap_sum1", "uniform", "coral_aligned"],
            "target": ["equinox", "lucene", "pde", "xalan-2.6", "poi-3.0", "velocity-1.6"]
        },
        "no_nan_columns": ["f1", "auc", "prauc"],
        "min_bytes": 1000,
    },
    "outputs/cpdp_fewshot_scores.csv": {
        "min_rows": 11520, "max_rows": 11520,
        "required_columns": ["ecosystem", "target", "model", "arm", "k", "draw_seed", "f1", "auc", "prauc"],
        "required_values": {
            "k": [10, 25, 50]
        },
        "no_nan_columns": ["f1", "auc", "prauc"],
        "min_bytes": 1000,
    },
    "outputs/cpdp_significance.csv": {
        "min_rows": 432, "max_rows": 432,
        "required_columns": ["setting", "target", "model", "metric", "arm", "mean_diff", "wilcoxon_p", "holm_corrected_p", "significant_at_0.05"],
        "min_bytes": 500,
    },
    "outputs/loso_weight_vectors.csv": {
        "min_rows": 125, "max_rows": 125,
        "required_columns": ["ecosystem", "source_project", "feature", "weight"],
        "no_nan_columns": ["weight"],
        "min_bytes": 500,
    },
    "outputs/loso_weight_stability.csv": {
        "min_rows": 20, "max_rows": 20,
        "required_columns": ["ecosystem", "project_a", "project_b", "spearman_rho", "cosine_sim"],
        "no_nan_columns": ["spearman_rho", "cosine_sim"],
        "min_bytes": 300,
    },
    "outputs/shap_prior_scores.csv": {
        "min_rows": 1920, "max_rows": 1920,
        "required_columns": ["target", "model", "arm", "boot_seed", "f1", "auc", "prauc"],
        "required_values": {
            "model": ["LogisticRegression", "SVC", "XGBoost", "LightGBM"],
            "arm": ["shap_prior", "uniform", "shuffled", "original"]
        },
        "no_nan_columns": ["f1", "auc", "prauc"],
        "min_bytes": 1000,
    },
    "outputs/shap_prior_fewshot_scores.csv": {
        "min_rows": 1920, "max_rows": 1920,
        "required_columns": ["target", "model", "arm", "k", "draw_seed", "f1", "auc", "prauc"],
        "required_values": {
            "model": ["LogisticRegression", "SVC", "XGBoost", "LightGBM"],
            "arm": ["shap_prior", "uniform", "shuffled", "original"],
            "k": [25]
        },
        "no_nan_columns": ["f1", "auc", "prauc"],
        "min_bytes": 1000,
    },
    "outputs/shap_prior_tests.csv": {
        "min_rows": 144, "max_rows": 144,
        "required_columns": ["target", "model", "metric", "comparison", "mean_diff", "wilcoxon_p", "holm_corrected_p", "significant_at_0.05"],
        "required_values": {
            "model": ["LogisticRegression", "SVC", "XGBoost", "LightGBM"]
        },
        "min_bytes": 500,
    },
    "outputs/shap_prior_verdict.csv": {
        "min_rows": 1, "max_rows": 1,
        "required_columns": ["wins_uniform", "wins_shuffled", "losses_uniform", "losses_shuffled", "passed", "group_vectors_nonconstant"],
        "min_bytes": 50,
    },
}


def check_contract(path_str, contract_dict=None):
    if contract_dict is None:
        contract_dict = CONTRACTS.get(path_str)
    if contract_dict is None:
        return False, "Unknown contract", []

    p = ROOT / path_str
    warnings = []

    # 1. File exists and size >= min_bytes
    if not p.exists():
        return False, "File does not exist", warnings

    size_bytes = p.stat().st_size
    min_b = contract_dict.get("min_bytes", 1)
    if size_bytes < min_b:
        return False, f"File size {size_bytes} B < min {min_b} B", warnings

    if contract_dict.get("is_binary", False):
        return True, "OK (Binary File)", warnings

    # 2. Parses as CSV with >= 1 data row
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return False, f"CSV Parse error: {e}", warnings

    num_rows = len(df)
    min_r = contract_dict.get("min_rows", 1)
    max_r = contract_dict.get("max_rows", 1000000)

    # 5. Row count within [min_rows, max_rows]
    if not (min_r <= num_rows <= max_r):
        return False, f"Row count {num_rows} outside [{min_r}, {max_r}]", warnings

    # 3. All required_columns present
    req_cols = contract_dict.get("required_columns", [])
    for col in req_cols:
        if col not in df.columns:
            return False, f"Missing required column: {col}", warnings

    # 4. For each required_values entry, every listed value actually appears
    req_vals = contract_dict.get("required_values", {})
    for col, vals in req_vals.items():
        if col not in df.columns:
            return False, f"Missing required_values column: {col}", warnings
        present_vals = set(df[col].dropna().unique())
        for v in vals:
            if v not in present_vals:
                return False, f"Value '{v}' missing in column '{col}'", warnings

    # 6. Zero NaN/Inf in no_nan_columns
    no_nan = contract_dict.get("no_nan_columns", [])
    for col in no_nan:
        if col in df.columns:
            if df[col].isna().any():
                return False, f"NaN values present in no_nan column '{col}'", warnings
            if np.isinf(df[col]).any():
                return False, f"Inf values present in no_nan column '{col}'", warnings

    # 7. RED FLAG DETECTORS
    # a. If a p-value column exists and >80% of its values are identical -> WARN
    p_cols = [c for c in df.columns if "p_val" in c or "wilcoxon_p" in c or "p_value" in c or "holm_corrected_p" in c]
    for pcol in p_cols:
        vals = df[pcol].dropna()
        if len(vals) > 5:
            top_freq = vals.value_counts(normalize=True).iloc[0]
            if top_freq > 0.8:
                warnings.append(f"Degenerate p-values in '{pcol}' ({top_freq*100:.1f}% identical)")

    # b. If a mean_diff/diff column exists and >5% of rows are exactly 0.0 -> WARN
    diff_cols = [c for c in df.columns if "diff" in c]
    for dcol in diff_cols:
        vals = df[dcol].dropna()
        if len(vals) > 5:
            zero_freq = (vals == 0.0).mean()
            if zero_freq > 0.05:
                warnings.append(f"Identical variants in '{dcol}' ({zero_freq*100:.1f}% rows exactly 0.0)")

    # c. If any numeric column has zero variance across the whole file -> WARN
    num_cols = df.select_dtypes(include=[np.number]).columns
    for ncol in num_cols:
        vals = df[ncol].dropna()
        if len(vals) > 5:
            if vals.std() == 0.0:
                warnings.append(f"Constant column '{ncol}' (std == 0)")

    # d. If a score column (f1/auc/prauc) has any value outside [0,1] -> FAIL
    score_cols = [c for c in df.columns if c in ["f1", "auc", "prauc"]]
    for scol in score_cols:
        vals = df[scol].dropna()
        if (vals < 0.0).any() or (vals > 1.0).any():
            return False, f"Score column '{scol}' has value outside [0, 1]", warnings

    return True, "OK", warnings


def check_all(subset_keys=None):
    keys = subset_keys if subset_keys is not None else list(CONTRACTS.keys())
    print(f"{'File':<40} | {'Exists':<6} | {'Rows (Exp)':<15} | {'Cols OK':<7} | {'Vals OK':<7} | {'NaN OK':<6} | {'Status':<6} | {'Warnings'}")
    print("-" * 120)

    all_passed = True
    for key in keys:
        cd = CONTRACTS[key]
        p = ROOT / key
        exists = "YES" if p.exists() else "NO"

        exp_rows = f"{cd.get('min_rows', 1)}-{cd.get('max_rows', 1)}" if not cd.get("is_binary") else "BINARY"
        ok, msg, warnings = check_contract(key, cd)

        if not ok:
            all_passed = False
            status_str = "FAIL"
        else:
            status_str = "PASS"

        warn_str = "; ".join(warnings) if warnings else "None"
        print(f"{key:<40} | {exists:<6} | {exp_rows:<15} | {'OK':<7} | {'OK':<7} | {'OK':<6} | {status_str:<6} | {warn_str}")
        if not ok:
            print(f"   -> Reason for FAIL: {msg}")

    print("-" * 120)
    print(f"Overall Status: {'ALL CONTRACTS PASSED' if all_passed else 'SOME CONTRACTS FAILED'}")
    return all_passed


if __name__ == "__main__":
    if len(sys.argv) > 1:
        keys_to_check = sys.argv[1:]
        sys.exit(0 if check_all(keys_to_check) else 1)
    else:
        sys.exit(0 if check_all() else 1)
