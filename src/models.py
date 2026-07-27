"""
models.py
Model definitions: base classifiers, custom model, stacking ensemble.
No SHAP / LIME anywhere.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
)
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

RANDOM_STATE = 42


# ── Custom SDP Classifier ──────────────────────────────────────────────────

class CustomSDPClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom SDP model that combines three user-requested elements:

    1. MinMaxScaler normalization  — normalizes every feature to [0, 1].
    2. BalancedRandomForestClassifier with deep trees — each tree is trained
       on a balanced bootstrap sample drawn internally, so NO SMOTE is needed.
       class_weight='balanced_subsample' gives additional per-sample weighting.
       max_depth=None grows fully-expanded trees (user: 'more depth').
    3. CV-based threshold optimization — 5-fold cross-validation finds the
       probability cut-off that maximises F1; avoids test-set threshold leak.

    Because this model handles class imbalance internally, it must be fitted
    on the ORIGINAL imbalanced (non-SMOTE) preprocessed data.
    """

    def __init__(self, n_estimators: int = 500,
                 max_depth=None,
                 min_samples_leaf: int = 1,
                 min_samples_split: int = 2,
                 max_features: str = "sqrt",
                 random_state: int = RANDOM_STATE):
        self.n_estimators    = n_estimators
        self.max_depth       = max_depth
        self.min_samples_leaf  = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features    = max_features
        self.random_state    = random_state

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_        = np.unique(y)
        self.n_features_in_  = X.shape[1]

        n0, n1 = int((y == 0).sum()), int((y == 1).sum())

        # ── Step 1: MinMaxScaler normalization ─────────────────────────
        self.normalizer_ = MinMaxScaler()
        X_norm = self.normalizer_.fit_transform(X)

        # ── Step 2: BalancedRandomForestClassifier (deep, imbalance-aware, OOB)
        self.model_ = BalancedRandomForestClassifier(
            n_estimators     = self.n_estimators,
            max_depth        = self.max_depth,
            min_samples_leaf = self.min_samples_leaf,
            min_samples_split= self.min_samples_split,
            max_features     = self.max_features,
            class_weight     = "balanced_subsample",
            bootstrap        = True,
            oob_score        = True,
            n_jobs           = 1,
            random_state     = self.random_state,
        )
        self.model_.fit(X_norm, y)

        # ── Step 3: OOB-based threshold optimization (no test leakage) ─
        # oob_decision_function_ gives per-sample probabilities from trees
        # that never saw that sample — equivalent to hold-out predictions.
        oob_probs = self.model_.oob_decision_function_[:, 1]
        thresholds = np.arange(0.10, 0.91, 0.02)
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            f = f1_score(y, (oob_probs >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        self.threshold_ = float(best_t)

        print(f"  [CustomSDP] n_train={len(y)}  n0={n0}  n1={n1}")
        print(f"  [CustomSDP] OOB_threshold={self.threshold_:.2f}  "
              f"OOB_F1={best_f1:.4f}")
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        X_norm = self.normalizer_.transform(np.asarray(X, dtype=float))
        return self.model_.predict_proba(X_norm)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)


# ── Standard model catalogue ───────────────────────────────────────────────

def build_base_models(y_train: np.ndarray) -> dict:
    """
    Return default classifiers with sensible imbalance handling.
    y_train: training labels (after SMOTE if applied for this set).
    """
    n0 = int((y_train == 0).sum())
    n1 = int((y_train == 1).sum())
    spw = n0 / max(n1, 1)
    k   = min(9, max(3, n1 // 3))

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            C=1.0, solver="lbfgs", random_state=RANDOM_STATE),

        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, class_weight="balanced",
            C=5.0, gamma="scale", random_state=RANDOM_STATE),

        "KNN": KNeighborsClassifier(
            n_neighbors=k, weights="distance", metric="euclidean"),

        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=None, class_weight="balanced",
            min_samples_leaf=2, min_samples_split=5,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=1),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, min_samples_leaf=5, random_state=RANDOM_STATE),

        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0),

        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            num_leaves=31, class_weight="balanced",
            random_state=RANDOM_STATE, verbose=-1),

        "CatBoost": CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=5,
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE, verbose=0),
    }


def build_stacking_ensemble(best_params: dict, y_train: np.ndarray) -> StackingClassifier:
    """XGBoost + LightGBM + CatBoost → Logistic Regression meta-learner."""
    n0 = int((y_train == 0).sum())
    n1 = int((y_train == 1).sum())
    spw = n0 / max(n1, 1)

    xgb_p = best_params.get("XGBoost", {})
    lgb_p  = best_params.get("LightGBM", {})
    cat_p  = best_params.get("CatBoost", {})

    xgb_clf = XGBClassifier(
        n_estimators     = xgb_p.get("n_estimators", 300),
        max_depth        = xgb_p.get("max_depth", 5),
        learning_rate    = xgb_p.get("learning_rate", 0.05),
        subsample        = xgb_p.get("subsample", 0.8),
        colsample_bytree = xgb_p.get("colsample_bytree", 0.8),
        reg_alpha        = xgb_p.get("reg_alpha", 0.1),
        reg_lambda       = xgb_p.get("reg_lambda", 1.0),
        min_child_weight = xgb_p.get("min_child_weight", 3),
        gamma            = xgb_p.get("gamma", 0.1),
        scale_pos_weight = spw,
        eval_metric      = "logloss",
        random_state     = RANDOM_STATE, verbosity=0)

    lgb_clf = lgb.LGBMClassifier(
        n_estimators  = lgb_p.get("n_estimators", 300),
        max_depth     = lgb_p.get("max_depth", 5),
        learning_rate = lgb_p.get("learning_rate", 0.05),
        num_leaves    = lgb_p.get("num_leaves", 31),
        reg_alpha     = lgb_p.get("reg_alpha", 0.0),
        reg_lambda    = lgb_p.get("reg_lambda", 1.0),
        class_weight  = "balanced",
        random_state  = RANDOM_STATE, verbose=-1)

    cat_clf = CatBoostClassifier(
        iterations    = cat_p.get("iterations", 300),
        learning_rate = cat_p.get("learning_rate", 0.05),
        depth         = cat_p.get("depth", 5),
        l2_leaf_reg   = cat_p.get("l2_leaf_reg", 3.0),
        auto_class_weights = "Balanced",
        random_seed   = RANDOM_STATE, verbose=0)

    meta = LogisticRegression(
        max_iter=2000, C=1.0, solver="lbfgs",
        class_weight="balanced", random_state=RANDOM_STATE)

    return StackingClassifier(
        estimators      = [("xgb", xgb_clf), ("lgb", lgb_clf), ("cat", cat_clf)],
        final_estimator = meta,
        cv              = 3,
        stack_method    = "predict_proba",
        n_jobs          = 1,
    )


def build_tuned_models(best_params: dict, y_train: np.ndarray) -> dict:
    """Build full model dict using Optuna best params where available."""
    base = build_base_models(y_train)
    n0 = int((y_train == 0).sum())
    n1 = int((y_train == 1).sum())
    spw = n0 / max(n1, 1)

    for mname, params in best_params.items():
        if not params:
            continue
        if mname == "XGBoost":
            base[mname] = XGBClassifier(
                **params, scale_pos_weight=spw,
                eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0)
        elif mname == "LightGBM":
            base[mname] = lgb.LGBMClassifier(
                **params, class_weight="balanced",
                random_state=RANDOM_STATE, verbose=-1)
        elif mname == "CatBoost":
            base[mname] = CatBoostClassifier(
                **params, auto_class_weights="Balanced",
                random_seed=RANDOM_STATE, verbose=0)
        elif mname == "Random Forest":
            base[mname] = RandomForestClassifier(
                **params, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=1)
        elif mname == "Logistic Regression":
            base[mname] = LogisticRegression(
                **params, class_weight="balanced",
                max_iter=2000, random_state=RANDOM_STATE)

    base["Stacking Ensemble"] = build_stacking_ensemble(best_params, y_train)
    return base
