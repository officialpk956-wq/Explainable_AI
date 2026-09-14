"""
create_notebook_rf.py - Simple RF approach
Normalize data -> Random Forest with class_weight='balanced' + increased depth
"""
import json, uuid

def uid():  return str(uuid.uuid4())[:8]
def md(src): return {"cell_type":"markdown","id":uid(),"metadata":{},"source":src}
def code(src): return {"cell_type":"code","execution_count":None,"id":uid(),
                       "metadata":{},"outputs":[],"source":src}
def L(s):
    parts = s.strip("\n").split("\n")
    return [p+"\n" if i<len(parts)-1 else p for i,p in enumerate(parts)]

C = []

# ── TITLE ─────────────────────────────────────────────────────────────────
C.append(md(L("""
# Simple Random Forest Approach
## Lucene & PDE — Normalized RF Classifier with Class Imbalance Handling

**Approach:**
1. Load & fix both datasets (zero-variance, duplicates, skewness, outliers)
2. Normalize features (StandardScaler)
3. Random Forest with:
   - `class_weight='balanced'` (handle imbalance)
   - `max_depth=15` (increased from default 10)
   - `n_estimators=500`
   - `min_samples_leaf=2`
4. 75/25 train-test split (no SMOTE initially)
5. Compare F1-scores on both datasets
""")))

# ── IMPORTS ───────────────────────────────────────────────────────────────
C.append(code(L("""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro, levene, mannwhitneyu, kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs("D:/Xai/outputs", exist_ok=True)
sns.set_style("whitegrid")
print("Libraries loaded.")
""")))

# ── SEC 1: LOAD DATA ──────────────────────────────────────────────────────
C.append(md(L("""
---
## Section 1 — Load & Prepare Datasets
""")))

C.append(code(L("""
df_lucene_raw = pd.read_csv("D:/Xai/data/lucene-bug-metrics.csv", sep=";")
df_lucene_raw.columns = df_lucene_raw.columns.str.strip()
df_lucene_raw = df_lucene_raw.loc[:, df_lucene_raw.columns != ""]

df_pde_raw = pd.read_csv("D:/Xai/data/pde-bug-metrics.csv", sep=";")
df_pde_raw.columns = df_pde_raw.columns.str.strip()
df_pde_raw = df_pde_raw.loc[:, df_pde_raw.columns != ""]

print(f"Lucene shape: {df_lucene_raw.shape}")
print(f"PDE shape: {df_pde_raw.shape}")
""")))

C.append(code(L("""
HIST_COLS = [
    "numberOfBugsFoundUntil:",
    "numberOfNonTrivialBugsFoundUntil:",
    "numberOfMajorBugsFoundUntil:",
    "numberOfCriticalBugsFoundUntil:",
    "numberOfHighPriorityBugsFoundUntil:"
]

def make_X_y(df_raw, name):
    avail = [c for c in HIST_COLS if c in df_raw.columns]
    X = df_raw[avail].copy().astype(float)
    y = (df_raw["bugs"] > 0).astype(int)
    print(f"{name}: X={X.shape}, class={dict(y.value_counts())}")
    return X, y

X_lucene_raw, y_lucene = make_X_y(df_lucene_raw, "Lucene")
X_pde_raw,    y_pde    = make_X_y(df_pde_raw,    "PDE")
""")))

# ── SEC 2: FIX DATASETS ───────────────────────────────────────────────────
C.append(md(L("""
---
## Section 2 — Fix Datasets
Drop zero-variance, duplicates; log1p transform; winsorize outliers
""")))

C.append(code(L("""
def fix_dataset(X_raw, y, name):
    X = X_raw.copy()

    # Remove zero-variance
    zv_mask = X.std() == 0
    zv_cols = X.columns[zv_mask].tolist()
    X.drop(columns=zv_cols, inplace=True)
    print(f"[{name}] Dropped zero-variance: {zv_cols}")

    if X.shape[1] == 0:
        return X, y

    # Remove duplicates (|r| > 0.99)
    corr_m = X.corr().abs()
    upper  = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
    dup_cols = [c for c in upper.columns if upper[c].max() > 0.99]
    X.drop(columns=dup_cols, inplace=True)
    print(f"[{name}] Dropped duplicates: {dup_cols}")

    # Log1p for skewed
    skewed = [c for c in X.columns if abs(X[c].skew()) > 1.0]
    for c in skewed:
        X[c] = np.log1p(X[c])
    print(f"[{name}] Log1p applied: {skewed}")

    # Winsorize outliers
    for c in X.columns:
        p1, p99 = X[c].quantile(0.01), X[c].quantile(0.99)
        X[c] = X[c].clip(p1, p99)
    print(f"[{name}] Winsorized (p1, p99)")

    print(f"[{name}] Final shape: {X.shape}")
    return X, y

X_lucene, y_lucene = fix_dataset(X_lucene_raw, y_lucene, "Lucene")
X_pde,    y_pde    = fix_dataset(X_pde_raw,    y_pde,    "PDE")
""")))

# ── SEC 3: TRAIN-TEST SPLIT & NORMALIZE ───────────────────────────────────
C.append(md(L("""
---
## Section 3 — Train-Test Split (75/25) & Normalize
""")))

C.append(code(L("""
def prepare_data(X, y, name):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_tr_norm = scaler.fit_transform(X_tr)
    X_te_norm = scaler.transform(X_te)

    n0, n1 = (y_tr == 0).sum(), (y_tr == 1).sum()
    imbalance = n0 / max(n1, 1)

    print(f"{name}:")
    print(f"  Train: {X_tr_norm.shape}, class dist: {dict(pd.Series(y_tr).value_counts())}")
    print(f"  Test:  {X_te_norm.shape}, class dist: {dict(pd.Series(y_te).value_counts())}")
    print(f"  Imbalance ratio: {imbalance:.1f}:1")

    return X_tr_norm, X_te_norm, y_tr, y_te

X_tr_luc, X_te_luc, y_tr_luc, y_te_luc = prepare_data(X_lucene, y_lucene, "Lucene")
X_tr_pde, X_te_pde, y_tr_pde, y_te_pde = prepare_data(X_pde, y_pde, "PDE")
""")))

# ── SEC 4: RANDOM FOREST WITH CLASS IMBALANCE HANDLING ────────────────────
C.append(md(L("""
---
## Section 4 — Random Forest Classifier
Parameters:
- `class_weight='balanced'` — handle class imbalance
- `max_depth=15` — increased depth
- `n_estimators=500` — strong ensemble
- `min_samples_leaf=2` — allow smaller leaves
""")))

C.append(code(L("""
def train_rf(X_tr, X_te, y_tr, y_te, name):
    print(f"\\n{'='*60}")
    print(f"  Random Forest Classifier — {name}")
    print(f"{'='*60}")

    # Create RF with class imbalance handling
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=2,
        min_samples_split=5,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )

    print("Training...")
    rf.fit(X_tr, y_tr)

    # Predictions
    y_pred = rf.predict(X_te)
    y_pred_proba = rf.predict_proba(X_te)[:, 1]

    # Metrics
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    auc = roc_auc_score(y_te, y_pred_proba)

    print(f"\\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")

    cm = confusion_matrix(y_te, y_pred)
    print(f"\\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}, TP={cm[1,1]}")

    print(f"\\n  Classification Report:")
    print(classification_report(y_te, y_pred, zero_division=0))

    # Feature importance
    feat_imp = pd.DataFrame({
        "Feature": X_tr.columns if hasattr(X_tr, 'columns') else [f"F{i}" for i in range(X_tr.shape[1])],
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)

    print(f"\\n  Top-5 Feature Importance:")
    print(feat_imp.head(5).to_string(index=False))

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": auc,
        "rf": rf,
        "y_pred": y_pred,
        "y_proba": y_pred_proba,
        "feat_imp": feat_imp
    }

# Train on both datasets
results_lucene = train_rf(X_tr_luc, X_te_luc, y_tr_luc, y_te_luc, "Lucene")
results_pde = train_rf(X_tr_pde, X_te_pde, y_tr_pde, y_te_pde, "PDE")
""")))

# ── SEC 5: COMPARISON ─────────────────────────────────────────────────────
C.append(md(L("""
---
## Section 5 — Results Summary
""")))

C.append(code(L("""
summary = pd.DataFrame([
    {
        "Dataset": "Lucene",
        "Accuracy": results_lucene["Accuracy"],
        "Precision": results_lucene["Precision"],
        "Recall": results_lucene["Recall"],
        "F1-Score": results_lucene["F1"],
        "ROC-AUC": results_lucene["ROC_AUC"]
    },
    {
        "Dataset": "PDE",
        "Accuracy": results_pde["Accuracy"],
        "Precision": results_pde["Precision"],
        "Recall": results_pde["Recall"],
        "F1-Score": results_pde["F1"],
        "ROC-AUC": results_pde["ROC_AUC"]
    }
])

print("\\n" + "="*70)
print("  RANDOM FOREST RESULTS SUMMARY")
print("="*70)
display(summary)

print(f"\\nAverage F1-Score: {summary['F1-Score'].mean():.4f}")
""")))

C.append(code(L("""
# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

metrics_map = {"Accuracy": "Accuracy", "Precision": "Precision",
               "Recall": "Recall", "F1-Score": "F1", "ROC-AUC": "ROC_AUC"}
metrics_labels = list(metrics_map.keys())
x = np.arange(len(metrics_labels))
w = 0.35

luc_vals = [results_lucene[metrics_map[m]] for m in metrics_labels]
pde_vals = [results_pde[metrics_map[m]] for m in metrics_labels]

axes[0].bar(x - w/2, luc_vals, w, label="Lucene", color="#1565C0", alpha=0.85)
axes[0].bar(x + w/2, pde_vals, w, label="PDE", color="#BF360C", alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_labels, rotation=15, ha="right")
axes[0].set_ylabel("Score", fontsize=11)
axes[0].set_title("Random Forest Performance Metrics", fontsize=12, fontweight="bold")
axes[0].set_ylim(0, 1.0)
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# Feature importance comparison
feat_imp_luc = results_lucene["feat_imp"].head(10)
feat_imp_pde = results_pde["feat_imp"].head(10)

y_pos = np.arange(len(feat_imp_pde))
axes[1].barh(y_pos, feat_imp_pde["Importance"].values, color="#BF360C", alpha=0.85)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(feat_imp_pde["Feature"].values)
axes[1].set_xlabel("Importance", fontsize=11)
axes[1].set_title("Top-10 Feature Importance (PDE)", fontsize=12, fontweight="bold")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("D:/Xai/outputs/rf_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: D:/Xai/outputs/rf_results.png")
""")))

# ── WRITE ─────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"SDP (sdp_env)","language":"python","name":"sdp_env"},
        "language_info": {"name":"python","version":"3.9.0"}
    },
    "cells": C
}
out = "D:/Xai/sdp_rf_simple.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Notebook written: {out}  ({len(C)} cells)")
