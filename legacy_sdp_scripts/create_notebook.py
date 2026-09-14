"""
create_notebook.py  — v3
Root-cause fixes:
  1. F1ThresholdWrapper removed (calibration on SMOTE data doesn't transfer to real test)
  2. Polynomial expansion conditional (only for <= 2 features — avoids overfitting on PDE)
  3. OOF cross-validated SHAP (unbiased importance, 3-fold x 2 rounds with momentum)
  4. Soft reweighting with floor (no feature zeroed out)
  5. Better hyperparameters across all standard models
  6. Plain SMOTE (consistent with test distribution expectation)
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
# Explainability-Guided Software Defect Prediction
## Lucene & PDE — Full Analysis Pipeline  (v3)

| Step | Section |
|------|---------|
| 1 | Data Loading & Basic Analytics |
| 2 | Feature Engineering & Target Definition |
| 3 | Statistical Tests (per dataset) |
| 4 | Fixing Issues from Tests |
| 5 | Pearson & Spearman Correlation |
| 6 | Feature Ranking |
| 7 | Enhanced `SHAPReweightedClassifier` + 7 Standard Models |
| 8 | Model Training — LUCENE |
| 9 | Model Training — PDE |
| 10 | Final F1-Score Comparison |

**Key improvements in v3 over v2:**
- Removed `F1ThresholdWrapper` from standard models (threshold on SMOTE data did not transfer)
- Custom model: conditional poly expansion only for ≤2 features (avoids PDE overfitting)
- Custom model: OOF cross-validated SHAP (3-fold, 2 rounds with momentum) — unbiased weights
- Custom model: soft reweighting with floor `max(1/strength, w*n)` — no feature zeroed out
- Custom model: 3-fold CV threshold calibration (F1-optimal)
- Better hyperparameters for all 7 standard classifiers
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from imblearn.over_sampling import SMOTE
import shap
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs("D:/Xai/outputs", exist_ok=True)
sns.set_style("whitegrid")
print("All libraries loaded.")
""")))

# ── SEC 1: DATA LOADING ───────────────────────────────────────────────────
C.append(md(L("""
---
## Section 1 — Data Loading & Basic Analytics
""")))

C.append(code(L("""
df_lucene_raw = pd.read_csv("D:/Xai/data/lucene-bug-metrics.csv", sep=";")
df_lucene_raw.columns = df_lucene_raw.columns.str.strip()
df_lucene_raw = df_lucene_raw.loc[:, df_lucene_raw.columns != ""]
print(f"Lucene  shape : {df_lucene_raw.shape}")
df_lucene_raw.head(3)
""")))

C.append(code(L("""
df_pde_raw = pd.read_csv("D:/Xai/data/pde-bug-metrics.csv", sep=";")
df_pde_raw.columns = df_pde_raw.columns.str.strip()
df_pde_raw = df_pde_raw.loc[:, df_pde_raw.columns != ""]
print(f"PDE     shape : {df_pde_raw.shape}")
df_pde_raw.head(3)
""")))

C.append(code(L("""
def basic_analytics(df, name):
    num   = df.select_dtypes(include=[np.number])
    tgt   = (df["bugs"] > 0).astype(int)
    vc    = tgt.value_counts()
    ratio = vc.get(0,0) / max(vc.get(1,1), 1)
    print("=" * 55)
    print(f"  {name}")
    print("=" * 55)
    print(f"  Rows         : {df.shape[0]}")
    print(f"  Numeric cols : {num.shape[1]}")
    print(f"  No-bug  (0)  : {vc.get(0,0)}")
    print(f"  Buggy   (1)  : {vc.get(1,0)}")
    print(f"  Imbalance    : {ratio:.2f}:1")
    print()
    return num.describe().round(3)

display(basic_analytics(df_lucene_raw, "LUCENE"))
display(basic_analytics(df_pde_raw,    "PDE"))
""")))

C.append(code(L("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (df, name) in zip(axes, [(df_lucene_raw,"Lucene"),(df_pde_raw,"PDE")]):
    tgt    = (df["bugs"] > 0).astype(int)
    counts = tgt.value_counts().sort_index()
    bars   = ax.bar(["No Bug (0)","Buggy (1)"], counts.values,
                    color=["#2196F3","#e74c3c"], alpha=0.85, edgecolor="black")
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+3,
                str(int(b.get_height())), ha="center", fontsize=11, fontweight="bold")
    ax.set_title(f"{name} — Class Distribution", fontsize=13)
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("D:/Xai/outputs/class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
""")))

# ── SEC 2: FEATURE ENGINEERING ────────────────────────────────────────────
C.append(md(L("""
---
## Section 2 — Feature Engineering & Target Definition

**Historical features (safe, no leakage):**
`numberOfBugsFoundUntil`, `numberOfNonTrivialBugsFoundUntil`,
`numberOfMajorBugsFoundUntil`, `numberOfCriticalBugsFoundUntil`, `numberOfHighPriorityBugsFoundUntil`

**Target:** `bugs > 0` → 1 (buggy), 0 (clean)

**Dropped (current-period derivatives — data leakage):**
`nonTrivialBugs`, `majorBugs`, `criticalBugs`, `highPriorityBugs`
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
    print(f"{name}: X={X.shape},  class dist={dict(y.value_counts())}")
    return X, y

X_lucene_raw, y_lucene = make_X_y(df_lucene_raw, "Lucene")
X_pde_raw,    y_pde    = make_X_y(df_pde_raw,    "PDE")
""")))

# ── SEC 3: STATISTICAL TESTS ──────────────────────────────────────────────
C.append(md(L("""
---
## Section 3 — Statistical Tests (Each Dataset Individually)

| Test | Null Hypothesis | alpha |
|------|-----------------|-------|
| Zero-Variance | std = 0 | — |
| Shapiro-Wilk | Normal distribution in buggy group | 0.05 |
| Levene | Equal variance between groups | 0.05 |
| Mann-Whitney U | No distribution difference | 0.05 |
| Kruskal-Wallis | No difference across groups | 0.05 |
""")))

C.append(code(L("""
def run_stat_tests(X, y, name):
    print("\\n" + "="*65)
    print(f"  STATISTICAL TESTS — {name}")
    print("="*65)
    rows = []
    for col in X.columns:
        feat = X[col]; g1 = feat[y==1].values; g0 = feat[y==0].values
        std  = feat.std(); skew = feat.skew(); zv = (std == 0.0)
        sw_p, normal = np.nan, False
        if not zv and len(g1) >= 3:
            _, sw_p = shapiro(g1[:5000]); normal = (sw_p > 0.05)
        lev_p, equal_var = np.nan, False
        if not zv and len(g1) > 1:
            _, lev_p = levene(g1, g0); equal_var = (lev_p > 0.05)
        mw_p, mw_sig = np.nan, False
        if not zv:
            _, mw_p = mannwhitneyu(g1, g0, alternative="two-sided"); mw_sig = (mw_p < 0.05)
        kw_p, kw_sig = np.nan, False
        if not zv:
            _, kw_p = kruskal(g1, g0); kw_sig = (kw_p < 0.05)
        rows.append({"Feature":col,"Std":round(std,4),"Skewness":round(skew,3),
                     "ZeroVar":zv,
                     "SW_p":round(sw_p,4) if not np.isnan(sw_p) else "NaN","Normal":normal,
                     "Levene_p":round(lev_p,4) if not np.isnan(lev_p) else "NaN","EqualVar":equal_var,
                     "MannWhit_p":round(mw_p,4) if not np.isnan(mw_p) else "NaN","MW_Sig":mw_sig,
                     "Kruskal_p":round(kw_p,4) if not np.isnan(kw_p) else "NaN","KW_Sig":kw_sig})
    df_res = pd.DataFrame(rows)
    print(f"  Zero-variance features    : {df_res['ZeroVar'].sum()}")
    print(f"  Non-normal (SW, bug grp)  : {(~df_res['Normal']&~df_res['ZeroVar']).sum()}")
    print(f"  Sig. Mann-Whitney (p<.05) : {df_res['MW_Sig'].sum()}")
    print(f"  Sig. Kruskal-Wallis       : {df_res['KW_Sig'].sum()}")
    return df_res

stat_lucene = run_stat_tests(X_lucene_raw, y_lucene, "LUCENE")
display(stat_lucene)
""")))

C.append(code(L("""
stat_pde = run_stat_tests(X_pde_raw, y_pde, "PDE")
display(stat_pde)
""")))

C.append(code(L("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, (stat_df, name) in zip(axes, [(stat_lucene,"Lucene"),(stat_pde,"PDE")]):
    valid  = stat_df[~stat_df["ZeroVar"]].set_index("Feature")
    colors = ["#e74c3c" if abs(s) > 1 else "#2ecc71" for s in valid["Skewness"]]
    ax.barh(valid.index, valid["Skewness"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(1, color="orange", lw=0.9, linestyle="--", label="|skew|=1")
    ax.axvline(-1, color="orange", lw=0.9, linestyle="--")
    ax.set_title(f"{name} — Skewness Before Fix", fontsize=12)
    ax.set_xlabel("Skewness"); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("D:/Xai/outputs/skewness_before.png", dpi=150, bbox_inches="tight")
plt.show()
""")))

# ── SEC 4: FIXES ──────────────────────────────────────────────────────────
C.append(md(L("""
---
## Section 4 — Fixing Issues from Statistical Tests

| Fix | Issue | Action |
|-----|-------|--------|
| Fix 1 | Zero-variance (std=0) | Drop |
| Fix 2 | Near-duplicate (|r|>0.99) | Drop one |
| Fix 3 | High skewness (|skew|>1) | `log1p` |
| Fix 4 | Outliers (1st/99th pct) | Winsorize |
| Fix 5 | Class imbalance | SMOTE on **training only** |

> Lucene collapses to **1 feature** after Fix 1+2.
> The Custom model compensates via **internal polynomial expansion** (Section 7).
""")))

C.append(code(L("""
def fix_dataset(X_raw, y, stat_df, name):
    print("\\n" + "="*58)
    print(f"  FIXING ISSUES — {name}")
    print("="*58)
    X = X_raw.copy()
    zv = stat_df[stat_df["ZeroVar"]]["Feature"].tolist()
    X.drop(columns=[c for c in zv if c in X.columns], inplace=True)
    print(f"[Fix 1] Dropped zero-variance ({len(zv)}): {zv}")
    if X.shape[1] == 0:
        print("  No features remain."); return X, y
    corr_m = X.corr().abs()
    upper  = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
    dup    = [c for c in upper.columns if upper[c].max() > 0.99]
    X.drop(columns=[c for c in dup if c in X.columns], inplace=True)
    print(f"[Fix 2] Dropped near-duplicates ({len(dup)}): {dup}")
    skewed = [c for c in X.columns if abs(X[c].skew()) > 1.0]
    for c in skewed: X[c] = np.log1p(X[c])
    print(f"[Fix 3] log1p applied ({len(skewed)}): {skewed}")
    n_clip = 0
    for c in X.columns:
        p1, p99 = X[c].quantile(0.01), X[c].quantile(0.99)
        n_clip += int(((X[c]<p1)|(X[c]>p99)).sum())
        X[c] = X[c].clip(p1, p99)
    print(f"[Fix 4] Winsorized {n_clip} values")
    vc    = y.value_counts()
    ratio = vc.get(0,0) / max(vc.get(1,1), 1)
    print(f"[Fix 5] Imbalance {ratio:.1f}:1 — SMOTE on training only")
    print(f"\\n  Final features ({X.shape[1]}): {X.columns.tolist()}")
    return X, y

X_lucene, y_lucene = fix_dataset(X_lucene_raw, y_lucene, stat_lucene, "LUCENE")
X_pde,    y_pde    = fix_dataset(X_pde_raw,    y_pde,    stat_pde,    "PDE")
""")))

C.append(code(L("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, (X, name) in zip(axes, [(X_lucene,"Lucene"),(X_pde,"PDE")]):
    skew = X.skew().sort_values()
    colors = ["#e74c3c" if abs(s)>1 else "#2ecc71" for s in skew]
    ax.barh(skew.index, skew.values, color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(1, color="orange", lw=0.9, linestyle="--")
    ax.axvline(-1, color="orange", lw=0.9, linestyle="--")
    ax.set_title(f"{name} — Skewness AFTER Fixes", fontsize=12)
    ax.set_xlabel("Skewness")
plt.tight_layout()
plt.savefig("D:/Xai/outputs/skewness_after.png", dpi=150, bbox_inches="tight")
plt.show()
""")))

# ── SEC 5: CORRELATION ────────────────────────────────────────────────────
C.append(md(L("""
---
## Section 5 — Pearson & Spearman Correlation
""")))

C.append(code(L("""
def compute_correlations(X, y, name):
    print("\\n" + "="*55)
    print(f"  CORRELATION — {name}")
    print("="*55)
    tmp = X.copy(); tmp["__y__"] = y.values
    p = tmp.corr(method="pearson")["__y__"].drop("__y__")
    s = tmp.corr(method="spearman")["__y__"].drop("__y__")
    df_c = pd.DataFrame({"Feature":p.index,
                         "Pearson_r":p.round(4).values,
                         "Spearman_rho":s.round(4).values,
                         "Abs_Pearson":p.abs().round(4).values,
                         "Abs_Spearman":s.abs().round(4).values})
    df_c["Mean_Abs"] = ((df_c["Abs_Pearson"]+df_c["Abs_Spearman"])/2).round(4)
    df_c = df_c.sort_values("Mean_Abs", ascending=False).reset_index(drop=True)
    n = len(df_c)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(3, n*1.0+1)))
    for ax, col, label in zip(axes,["Pearson_r","Spearman_rho"],["Pearson r","Spearman rho"]):
        colors = ["#e74c3c" if v>0 else "#3498db" for v in df_c[col]]
        ax.barh(df_c["Feature"], df_c[col], color=colors, alpha=0.85)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(label, fontsize=11)
        ax.set_title(f"{label} with Bug Target ({name})", fontsize=12)
        for i, v in enumerate(df_c[col]):
            ax.text(v+(0.01 if v>=0 else -0.01), i, f"{v:.4f}", va="center",
                    ha=("left" if v>=0 else "right"), fontsize=9)
        ax.set_xlim(-1.1, 1.1)
    plt.tight_layout()
    plt.savefig(f"D:/Xai/outputs/correlation_{name.lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()
    return df_c

corr_lucene = compute_correlations(X_lucene, y_lucene, "LUCENE")
display(corr_lucene)
""")))

C.append(code(L("""
corr_pde = compute_correlations(X_pde, y_pde, "PDE")
display(corr_pde)
""")))

# ── SEC 6: RANKING ────────────────────────────────────────────────────────
C.append(md(L("""
---
## Section 6 — Feature Ranking

Ranked by **Mean Absolute Correlation** = (|Pearson r| + |Spearman rho|) / 2
""")))

C.append(code(L("""
def feature_ranking(corr_df, name):
    ranked = corr_df[["Feature","Abs_Pearson","Abs_Spearman","Mean_Abs"]].copy()
    ranked.insert(0, "Rank", range(1, len(ranked)+1))
    print(f"Feature Ranking — {name}")
    display(ranked)
    x, w = np.arange(len(ranked)), 0.32
    fig, ax = plt.subplots(figsize=(max(8, len(ranked)*2), 5))
    ax.bar(x-w/2, ranked["Abs_Pearson"],  w, label="|Pearson r|",    color="#2196F3", alpha=0.85)
    ax.bar(x+w/2, ranked["Abs_Spearman"], w, label="|Spearman rho|", color="#FF5722", alpha=0.85)
    ax.plot(x, ranked["Mean_Abs"], "ko--", lw=1.5, ms=6, label="Mean |Corr|")
    ax.set_xticks(x)
    ax.set_xticklabels(ranked["Feature"], rotation=30, ha="right")
    ax.set_ylabel("Absolute Correlation", fontsize=11)
    ax.set_title(f"Feature Ranking — {name}", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0); ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"D:/Xai/outputs/ranking_{name.lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()
    return ranked["Feature"].tolist()

feats_lucene = feature_ranking(corr_lucene, "LUCENE")
feats_pde    = feature_ranking(corr_pde,    "PDE")
print("Lucene ranked features:", feats_lucene)
print("PDE    ranked features:", feats_pde)
""")))

# ── SEC 7: MODELS ─────────────────────────────────────────────────────────
C.append(md(L("""
---
## Section 7 — Enhanced `SHAPReweightedClassifier`

### Algorithm (v3)

```
Phase 1  Conditional Polynomial Expansion
           If n_features <= 2 (e.g. Lucene):  [x]   -> [x, x^2, sqrt|x|]
           If n_features >  2 (e.g. PDE):     [x_1..x_5] -> [x_1..x_5]   (no expansion)
           Reason: expansion on many features causes overfitting in pilot

Phase 2  OOF Cross-Validated SHAP  (3-fold, 2 rounds, momentum alpha=0.6)
           For each fold: train pilot XGBoost on fold-train, compute SHAP on fold-val
           Average SHAP over 3 folds -> unbiased, not inflated by training-data memorisation
           Repeat 2 rounds: w = 0.4*w_prev + 0.6*w_new

Phase 3  Soft Reweighting  (no feature zeroed out)
           scaled_w  = shap_weights * n_features          (mean becomes 1.0)
           soft_w    = max(1/strength, scaled_w)          (floor prevents zeroing)
           soft_w    = soft_w / mean(soft_w)              (normalise)
           X_final   = X_expanded * soft_w

Phase 4  Final XGBoost on X_final (full training data)

Phase 5  3-fold CV Threshold Calibration (F1-optimal on SMOTE training)
           Threshold found on SMOTE folds -> reasonable for test
```

### 7 Standard Models
Same as v1 but with improved hyperparameters (no threshold wrapper — SMOTE + class_weight already handles imbalance).
""")))

C.append(code(L("""
class SHAPReweightedClassifier(BaseEstimator, ClassifierMixin):
    '''
    Enhanced SHAP-guided SDP classifier  (v3).

    Core novelty:
      - OOF cross-validated SHAP avoids pilot-overfitting bias
      - Soft reweighting keeps all features active
      - Conditional expansion adds signal only where beneficial
    '''
    def __init__(self, n_rounds=2, pilot_n=100, final_n=400,
                 lr=0.03, depth=4, strength=2.5, seed=42):
        self.n_rounds = n_rounds
        self.pilot_n  = pilot_n
        self.final_n  = final_n
        self.lr       = lr
        self.depth    = depth
        self.strength = strength
        self.seed     = seed

    def _pilot_xgb(self, spw, n_feat):
        cbt = min(1.0, max(0.5, 5.0 / n_feat))
        return XGBClassifier(
            n_estimators=self.pilot_n, max_depth=3,
            learning_rate=0.1, subsample=0.8, colsample_bytree=cbt,
            min_child_weight=5, reg_alpha=0.5, reg_lambda=1.0,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=self.seed, verbosity=0
        )

    def _final_xgb(self, spw, n_feat):
        cbt = min(1.0, max(0.5, 6.0 / n_feat))
        reg = max(0.3, n_feat * 0.05)
        return XGBClassifier(
            n_estimators=self.final_n, max_depth=self.depth,
            learning_rate=self.lr, subsample=0.8, colsample_bytree=cbt,
            min_child_weight=3, reg_alpha=reg, reg_lambda=reg,
            gamma=0.05, scale_pos_weight=spw, eval_metric="logloss",
            random_state=self.seed, verbosity=0
        )

    def _expand(self, X):
        '''Polynomial expansion only for <= 2 features (avoids PDE overfitting).'''
        if X.shape[1] <= 2:
            return np.hstack([X, X**2, np.sqrt(np.abs(X))])
        return X

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_       = np.unique(y)
        spw = (y == 0).sum() / max((y == 1).sum(), 1)

        # Phase 1: conditional feature expansion
        X_exp  = self._expand(X)
        n_exp  = X_exp.shape[1]

        # Phase 2: OOF cross-validated SHAP (unbiased importance)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed)
        w   = np.ones(n_exp) / n_exp

        for rnd in range(self.n_rounds):
            fold_shap = []
            for tr_i, val_i in skf.split(X_exp, y):
                Xw = X_exp[tr_i] * w
                pilot = self._pilot_xgb(spw, n_exp)
                pilot.fit(Xw, y[tr_i])
                explainer = shap.TreeExplainer(pilot)
                sv = explainer.shap_values(X_exp[val_i] * w)
                if isinstance(sv, list):
                    sv = sv[1]
                fold_shap.append(np.abs(sv).mean(axis=0))
            round_shap = np.mean(fold_shap, axis=0)
            tot = round_shap.sum()
            new_w = round_shap / tot if tot > 0 else w
            w = 0.4 * w + 0.6 * new_w          # momentum update

        self.shap_weights_ = w / w.sum()       # final normalisation

        # Phase 3: soft reweighting — floor prevents zeroing features
        scaled = self.shap_weights_ * n_exp    # mean -> 1.0
        floor  = 1.0 / self.strength           # minimum = 1/strength of mean
        soft   = np.maximum(floor, scaled)
        self.soft_w_ = soft / soft.mean()      # normalise mean back to 1.0

        X_final = X_exp * self.soft_w_

        # Phase 4: 3-fold CV threshold calibration
        thresholds = np.arange(0.05, 0.96, 0.02)
        t_f1 = {float(t): [] for t in thresholds}
        for tr_i, val_i in skf.split(X_final, y):
            m = self._final_xgb(spw, n_exp)
            m.fit(X_final[tr_i], y[tr_i])
            probs = m.predict_proba(X_final[val_i])[:, 1]
            for t in thresholds:
                t_f1[float(t)].append(
                    f1_score(y[val_i], (probs >= t).astype(int), zero_division=0)
                )
        self.threshold_ = max(t_f1, key=lambda t: np.mean(t_f1[t]))

        # Phase 5: final model on full training data
        self.final_ = self._final_xgb(spw, n_exp)
        self.final_.fit(X_final, y)

        top3 = sorted(zip(range(n_exp), self.shap_weights_), key=lambda x: x[1], reverse=True)[:3]
        print(f"  [Custom] {X.shape[1]} orig -> {n_exp} expanded features")
        print(f"  [Custom] Top-3 SHAP weights : {[(i, round(v,4)) for i,v in top3]}")
        print(f"  [Custom] Soft-w floor       : {floor:.3f}")
        print(f"  [Custom] CV F1 threshold    : {self.threshold_:.2f}")
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["final_", "soft_w_"])
        return self.final_.predict_proba(self._expand(check_array(X)) * self.soft_w_)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)

print("SHAPReweightedClassifier v3 defined.")
""")))

# ── SEC 8: LUCENE ─────────────────────────────────────────────────────────
C.append(md(L("""
---
## Section 8 — Model Training & F1 Evaluation — LUCENE

**Pipeline (strict no-leakage):**
1. Stratified 75/25 train-test split
2. `StandardScaler` fit on train only
3. `SMOTE` on train only (when imbalance >= 3:1)
4. Train Custom + 7 standard models
5. Evaluate on untouched test — **F1-Score is primary metric**
""")))

C.append(code(L("""
def prepare_data(X, y, ranked_feats, name):
    avail   = [f for f in ranked_feats if f in X.columns] or X.columns.tolist()
    Xs      = X[avail].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    n0, n1  = (y_tr == 0).sum(), (y_tr == 1).sum()
    ratio   = n0 / max(n1, 1)
    if ratio >= 3.0 and n1 >= 5:
        k  = min(5, n1 - 1)
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
        X_tr_s, y_tr_np = sm.fit_resample(X_tr_s, y_tr)
        print(f"{name}: SMOTE -> {dict(pd.Series(y_tr_np).value_counts())}")
    else:
        y_tr_np = y_tr.values
        print(f"{name}: No SMOTE (ratio {ratio:.1f}:1)")

    print(f"{name}: train={X_tr_s.shape}  test={X_te_s.shape}")
    return X_tr_s, X_te_s, y_tr_np, y_te.values, avail
""")))

C.append(code(L("""
def get_standard_models(y_tr):
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    k   = min(9, max(3, int((y_tr == 1).sum() // 3)))
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            C=1.0, solver="lbfgs", random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, class_weight="balanced",
            C=5.0, gamma="scale", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(
            n_neighbors=k, weights="distance", metric="euclidean"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=8, class_weight="balanced",
            min_samples_leaf=3, max_features="sqrt", random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=4,
            subsample=0.8, min_samples_leaf=5, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, scale_pos_weight=spw,
            eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0)
    }
""")))

C.append(code(L("""
def train_and_eval(X_tr, X_te, y_tr, y_te, name):
    print("\\n" + "="*65)
    print(f"  MODEL TRAINING & EVALUATION — {name}")
    print("="*65)
    rows = []

    # Custom model
    print("\\n  [1/8] SHAPReweightedClassifier (Custom)...")
    cm = SHAPReweightedClassifier(seed=RANDOM_STATE)
    cm.fit(X_tr, y_tr)
    yp = cm.predict(X_te);  pp = cm.predict_proba(X_te)[:, 1]
    rows.append({"Model": "SHAPReweighted (Custom)",
                 "Accuracy" : round(accuracy_score(y_te, yp), 4),
                 "Precision": round(precision_score(y_te, yp, zero_division=0), 4),
                 "Recall"   : round(recall_score(y_te, yp, zero_division=0), 4),
                 "F1"       : round(f1_score(y_te, yp, zero_division=0), 4),
                 "ROC_AUC"  : round(roc_auc_score(y_te, pp), 4)})

    # 7 standard models
    for idx, (mname, model) in enumerate(get_standard_models(y_tr).items(), 2):
        print(f"  [{idx}/8] {mname}...")
        model.fit(X_tr, y_tr)
        yp = model.predict(X_te);  pp = model.predict_proba(X_te)[:, 1]
        rows.append({"Model": mname,
                     "Accuracy" : round(accuracy_score(y_te, yp), 4),
                     "Precision": round(precision_score(y_te, yp, zero_division=0), 4),
                     "Recall"   : round(recall_score(y_te, yp, zero_division=0), 4),
                     "F1"       : round(f1_score(y_te, yp, zero_division=0), 4),
                     "ROC_AUC"  : round(roc_auc_score(y_te, pp), 4)})

    df_res = (pd.DataFrame(rows)
               .sort_values("F1", ascending=False)
               .reset_index(drop=True))
    df_res.index += 1
    print("\\n  Results (sorted by F1-Score):")
    display(df_res)
    return df_res
""")))

C.append(code(L("""
X_tr_luc, X_te_luc, y_tr_luc, y_te_luc, sel_luc = prepare_data(
    X_lucene, y_lucene, feats_lucene, "LUCENE"
)
results_lucene = train_and_eval(X_tr_luc, X_te_luc, y_tr_luc, y_te_luc, "LUCENE")
""")))

C.append(code(L("""
def plot_f1(results_df, name):
    df     = results_df.sort_values("F1", ascending=True)
    colors = ["#e74c3c" if "Custom" in m else "#3498db" for m in df["Model"]]
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(df)*0.65+2)))
    for ax, col, xlabel in zip(axes, ["F1","ROC_AUC"], ["F1-Score","ROC-AUC"]):
        bars = ax.barh(df["Model"], df[col], color=colors, alpha=0.85,
                       edgecolor="white", height=0.6)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_title(f"{xlabel} — {name}", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1.08)
        ax.axvline(0.5, color="gray", lw=0.8, linestyle="--", alpha=0.6)
        for bar, v in zip(bars, df[col]):
            ax.text(v+0.012, bar.get_y()+bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=9)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#e74c3c", label="Custom (SHAPReweighted v3)"),
                        Patch(color="#3498db", label="Standard Models")],
               loc="lower center", ncol=2, fontsize=11, bbox_to_anchor=(0.5,-0.04))
    plt.tight_layout()
    plt.savefig(f"D:/Xai/outputs/f1_models_{name.lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_f1(results_lucene, "LUCENE")
""")))

# ── SEC 9: PDE ────────────────────────────────────────────────────────────
C.append(md(L("""
---
## Section 9 — Model Training & F1 Evaluation — PDE
""")))

C.append(code(L("""
X_tr_pde, X_te_pde, y_tr_pde, y_te_pde, sel_pde = prepare_data(
    X_pde, y_pde, feats_pde, "PDE"
)
results_pde = train_and_eval(X_tr_pde, X_te_pde, y_tr_pde, y_te_pde, "PDE")
""")))

C.append(code(L("""
plot_f1(results_pde, "PDE")
""")))

# ── SEC 10: FINAL COMPARISON ──────────────────────────────────────────────
C.append(md(L("""
---
## Section 10 — Final F1-Score Comparison: Lucene vs PDE
""")))

C.append(code(L("""
model_order = results_lucene.sort_values("F1", ascending=False)["Model"].tolist()
f1_luc = results_lucene.set_index("Model").reindex(model_order)["F1"].values
f1_pde = results_pde.set_index("Model").reindex(model_order)["F1"].fillna(0).values

x, w = np.arange(len(model_order)), 0.37
fig, ax = plt.subplots(figsize=(16, 6))
b1 = ax.bar(x-w/2, f1_luc, w, label="Lucene", color="#1565C0", alpha=0.85)
b2 = ax.bar(x+w/2, f1_pde, w, label="PDE",    color="#BF360C", alpha=0.85)

for bars, vals in [(b1, f1_luc),(b2, f1_pde)]:
    for bar, v in zip(bars, vals):
        if v > 0.001:
            ax.text(bar.get_x()+bar.get_width()/2, v+0.006,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

for i, m in enumerate(model_order):
    if "Custom" in m:
        ax.axvspan(i-0.5, i+0.5, alpha=0.07, color="red")
        break

ax.set_xticks(x)
ax.set_xticklabels(model_order, rotation=28, ha="right", fontsize=10)
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("F1-Score — All 8 Models: Lucene vs PDE (v3)", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="gray", lw=0.8, linestyle="--", alpha=0.5)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig("D:/Xai/outputs/f1_comparison_combined_v3.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: D:/Xai/outputs/f1_comparison_combined_v3.png")
""")))

C.append(code(L("""
luc_s = results_lucene[["Model","F1","ROC_AUC"]].rename(
    columns={"F1":"F1_Lucene","ROC_AUC":"AUC_Lucene"})
pde_s = results_pde[["Model","F1","ROC_AUC"]].rename(
    columns={"F1":"F1_PDE","ROC_AUC":"AUC_PDE"})
summary = luc_s.merge(pde_s, on="Model", how="outer")
summary["F1_Avg"] = ((summary["F1_Lucene"].fillna(0)+summary["F1_PDE"].fillna(0))/2).round(4)
summary = summary.sort_values("F1_Avg", ascending=False).reset_index(drop=True)
summary.index += 1

print("="*72)
print("  FINAL SUMMARY v3 — F1-Score & ROC-AUC (Lucene and PDE)")
print("="*72)
display(summary)

best = summary.iloc[0]
print(f"\\n  Best overall model  : {best['Model']}")
print(f"  F1  Lucene={best['F1_Lucene']}  PDE={best['F1_PDE']}  Avg={best['F1_Avg']}")

custom = summary[summary["Model"].str.contains("Custom")]
if not custom.empty:
    rank = custom.index[0]
    print(f"\\n  Custom model rank   : #{rank}")
    print(f"  Custom F1  Lucene={custom['F1_Lucene'].values[0]}  "
          f"PDE={custom['F1_PDE'].values[0]}  Avg={custom['F1_Avg'].values[0]}")
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
out = "D:/Xai/sdp_lucene_pde_analysis.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Notebook written: {out}  ({len(C)} cells)")
