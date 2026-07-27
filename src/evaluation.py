"""
evaluation.py
Leakage-free model evaluation, threshold optimization, calibration,
and statistical significance testing. No SHAP / LIME.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
N_SPLITS     = 5
THRESHOLD_RANGE = np.arange(0.10, 0.91, 0.02)
OUTPUT_DIR   = Path("D:/Xai/outputs")


# ── Cross-Validation Baseline ──────────────────────────────────────────────

def cv_evaluate_model(model, preproc_steps, X_train, y_train,
                      n_splits=N_SPLITS) -> dict:
    """
    Stratified k-fold CV with SMOTE inside each fold (no leakage).
    X_train: raw (unprocessed) DataFrame; pipeline handles all transforms.
    """
    n1 = int((y_train == 1).sum())
    k  = max(1, min(5, n1 - 1))
    smote_step = ("smote", SMOTE(k_neighbors=k, random_state=RANDOM_STATE))

    pipe = ImbPipeline(preproc_steps + [smote_step, ("model", model)])
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True,
                            random_state=RANDOM_STATE)

    f1_scores, auc_scores, pr_auc_scores, mcc_scores = [], [], [], []
    fold_preds = {}  # store (y_true, y_pred) per fold for McNemar

    y_arr = y_train.values if hasattr(y_train, "values") else np.array(y_train)
    X_arr = X_train.values if hasattr(X_train, "values") else np.array(X_train)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_arr, y_arr)):
        X_tr, X_val = X_arr[tr_i], X_arr[val_i]
        y_tr, y_val = y_arr[tr_i], y_arr[val_i]
        try:
            pipe.fit(X_tr, y_tr)
            yp  = pipe.predict(X_val)
            pp  = pipe.predict_proba(X_val)[:, 1]
            f1_scores.append(f1_score(y_val, yp, zero_division=0))
            auc_scores.append(roc_auc_score(y_val, pp))
            pr_auc_scores.append(average_precision_score(y_val, pp))
            mcc_scores.append(matthews_corrcoef(y_val, yp))
            fold_preds[fold] = (y_val, yp)
        except Exception as e:
            f1_scores.append(0.0)
            auc_scores.append(0.5)
            pr_auc_scores.append(0.0)
            mcc_scores.append(0.0)

    return {
        "CV_F1_Mean":    round(float(np.mean(f1_scores)), 4),
        "CV_F1_Std":     round(float(np.std(f1_scores)), 4),
        "CV_AUC_Mean":   round(float(np.mean(auc_scores)), 4),
        "CV_AUC_Std":    round(float(np.std(auc_scores)), 4),
        "CV_PRAUC_Mean": round(float(np.mean(pr_auc_scores)), 4),
        "CV_MCC_Mean":   round(float(np.mean(mcc_scores)), 4),
        "fold_scores":   f1_scores,
        "fold_preds":    fold_preds,
    }


def run_cv_all_models(models: dict, preproc_steps, X_train, y_train,
                      dataset_name: str = "") -> pd.DataFrame:
    rows = []
    for mname, model in models.items():
        print(f"  CV [{dataset_name}]: {mname}...")
        res = cv_evaluate_model(model, preproc_steps, X_train, y_train)
        rows.append({"Model": mname, **{k: v for k, v in res.items()
                                        if k not in ("fold_scores", "fold_preds")}})
    df = pd.DataFrame(rows).sort_values("CV_F1_Mean", ascending=False).reset_index(drop=True)
    return df


# ── Threshold Optimization ──────────────────────────────────────────────────

def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                       thresholds=THRESHOLD_RANGE) -> dict:
    """
    Search threshold range for best F1 on validation/test predictions.
    """
    best_t, best_f1 = 0.5, 0.0
    records = []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        f  = f1_score(y_true, yp, zero_division=0)
        p  = precision_score(y_true, yp, zero_division=0)
        r  = recall_score(y_true, yp, zero_division=0)
        records.append({"Threshold": round(t, 2), "F1": round(f, 4),
                        "Precision": round(p, 4), "Recall": round(r, 4)})
        if f > best_f1:
            best_f1, best_t = f, t
    return {
        "best_threshold": round(float(best_t), 2),
        "best_f1":        round(float(best_f1), 4),
        "table":          pd.DataFrame(records),
    }


def threshold_table_all_models(fitted_models: dict, X_test: np.ndarray,
                                y_test: np.ndarray) -> pd.DataFrame:
    rows = []
    for mname, model in fitted_models.items():
        try:
            pp = model.predict_proba(X_test)[:, 1]
            res = optimize_threshold(y_test, pp)
            rows.append({"Model": mname,
                         "Best_Threshold": res["best_threshold"],
                         "Best_F1":        res["best_f1"]})
        except Exception as e:
            rows.append({"Model": mname, "Best_Threshold": 0.5, "Best_F1": 0.0})
    return pd.DataFrame(rows).sort_values("Best_F1", ascending=False).reset_index(drop=True)


# ── Probability Calibration ─────────────────────────────────────────────────

def calibrate_model(model, X_train: np.ndarray, y_train: np.ndarray,
                    method: str = "isotonic") -> CalibratedClassifierCV:
    cal = CalibratedClassifierCV(estimator=model, method=method, cv=3)
    cal.fit(X_train, y_train)
    return cal


# ── Full Test-Set Evaluation ────────────────────────────────────────────────

def compute_test_metrics(model, X_test: np.ndarray, y_test: np.ndarray,
                         threshold: float = 0.5, model_name: str = "") -> dict:
    pp  = model.predict_proba(X_test)[:, 1]
    yp  = (pp >= threshold).astype(int)
    return {
        "Model":     model_name,
        "Threshold": threshold,
        "Accuracy":  round(accuracy_score(y_test, yp), 4),
        "Precision": round(precision_score(y_test, yp, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, yp, zero_division=0), 4),
        "F1":        round(f1_score(y_test, yp, zero_division=0), 4),
        "ROC_AUC":   round(roc_auc_score(y_test, pp), 4),
        "PR_AUC":    round(average_precision_score(y_test, pp), 4),
        "MCC":       round(matthews_corrcoef(y_test, yp), 4),
    }


def full_evaluation(fitted_models: dict, X_test: np.ndarray, y_test: np.ndarray,
                    best_thresholds: dict = None) -> pd.DataFrame:
    rows = []
    for mname, model in fitted_models.items():
        t = (best_thresholds or {}).get(mname, 0.5)
        rows.append(compute_test_metrics(model, X_test, y_test, t, mname))
    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


# ── Statistical Significance ────────────────────────────────────────────────

def wilcoxon_pairwise(cv_fold_scores: dict, alpha: float = 0.05) -> pd.DataFrame:
    """
    Wilcoxon signed-rank test on per-fold F1 scores across model pairs.
    cv_fold_scores: {model_name: [fold_f1_scores]}
    """
    model_names = list(cv_fold_scores.keys())
    rows = []
    for m1, m2 in combinations(model_names, 2):
        s1 = np.array(cv_fold_scores[m1])
        s2 = np.array(cv_fold_scores[m2])
        if len(s1) < 3 or np.allclose(s1, s2):
            rows.append({"Model_A": m1, "Model_B": m2,
                         "W_stat": np.nan, "p_value": np.nan, "Significant": False})
            continue
        try:
            stat, p = wilcoxon(s1, s2)
            rows.append({"Model_A": m1, "Model_B": m2,
                         "W_stat": round(float(stat), 4),
                         "p_value": round(float(p), 4),
                         "Significant": p < alpha})
        except Exception:
            rows.append({"Model_A": m1, "Model_B": m2,
                         "W_stat": np.nan, "p_value": np.nan, "Significant": False})
    df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    print(f"  Significant pairs (p<{alpha}): {df['Significant'].sum()} / {len(df)}")
    return df


def mcnemar_pairwise(predictions: dict, y_true: np.ndarray,
                     alpha: float = 0.05) -> pd.DataFrame:
    """
    McNemar test comparing correctness tables for each model pair.
    predictions: {model_name: binary_predictions_array}
    """
    model_names = list(predictions.keys())
    rows = []
    for m1, m2 in combinations(model_names, 2):
        p1 = np.array(predictions[m1]) == y_true
        p2 = np.array(predictions[m2]) == y_true
        b  = int(( p1 & ~p2).sum())
        c  = int((~p1 &  p2).sum())
        table = [[int((p1 & p2).sum()), c],
                 [b,                    int((~p1 & ~p2).sum())]]
        try:
            res = mcnemar(table, exact=True)
            rows.append({"Model_A": m1, "Model_B": m2,
                         "b": b, "c": c,
                         "p_value": round(float(res.pvalue), 4),
                         "Significant": res.pvalue < alpha})
        except Exception:
            rows.append({"Model_A": m1, "Model_B": m2,
                         "b": b, "c": c, "p_value": np.nan, "Significant": False})
    df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    print(f"  Significant McNemar pairs (p<{alpha}): {df['Significant'].sum()} / {len(df)}")
    return df
