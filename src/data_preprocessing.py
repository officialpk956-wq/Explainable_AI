"""
data_preprocessing.py
Leakage-free preprocessing transformers and pipeline builder.
All fit() calls happen strictly on training data only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import skew as scipy_skew

RANDOM_STATE = 42


class ZeroVarianceDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.keep_mask_ = X.std(axis=0) > 0
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_mask_]


class CorrelationDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.99):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[1]
        corr = np.corrcoef(X.T) if n > 1 else np.array([[1.0]])
        self.keep_mask_ = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if self.keep_mask_[i] and self.keep_mask_[j]:
                    if abs(corr[i, j]) > self.threshold:
                        self.keep_mask_[j] = False
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_mask_]


class SkewedLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.skewed_idx_ = [
            i for i in range(X.shape[1])
            if abs(scipy_skew(X[:, i])) > self.threshold
        ]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        for i in self.skewed_idx_:
            X[:, i] = np.log1p(np.abs(X[:, i]))
        return X


class PolynomialExpander(BaseEstimator, TransformerMixin):
    """
    Expand features with x^2 and sqrt(|x|) when n_features_after_fit <= max_features.
    Helps datasets where preprocessing reduces to very few features (e.g. Lucene → 1 feature).
    fit() records the feature count from training data; transform() applies consistently.
    """
    def __init__(self, max_features=3):
        self.max_features = max_features

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.apply_ = X.shape[1] <= self.max_features
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if not self.apply_:
            return X
        return np.hstack([X, X ** 2, np.sqrt(np.abs(X))])


class InteractionExpander(BaseEstimator, TransformerMixin):
    """
    Add pairwise feature products when n_features >= min_features.
    Activates for PDE (5 features → 5 + 10 interactions = 15) but not
    Lucene (3 features after PolynomialExpander, below threshold).
    """
    def __init__(self, min_features=4):
        self.min_features = min_features

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.apply_ = X.shape[1] >= self.min_features
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if not self.apply_:
            return X
        n = X.shape[1]
        cols = [X]
        for i in range(n):
            for j in range(i + 1, n):
                cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack(cols)


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_pct=1.0, upper_pct=99.0):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.percentile(X, self.lower_pct, axis=0)
        self.upper_ = np.percentile(X, self.upper_pct, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def make_preprocessing_steps():
    """
    Ordered preprocessing steps — all fit on training data only.
    - PolynomialExpander: activates for ≤3 features (Lucene: 1→3).
    - InteractionExpander: activates for ≥4 features (PDE: 5→15).
    """
    return [
        ('zero_var',  ZeroVarianceDropper()),
        ('corr_drop', CorrelationDropper(threshold=0.99)),
        ('log_skew',  SkewedLogTransformer(threshold=1.0)),
        ('poly_exp',  PolynomialExpander(max_features=3)),
        ('interact',  InteractionExpander(min_features=4)),
        ('winsorize', Winsorizer(lower_pct=1.0, upper_pct=99.0)),
        ('scaler',    RobustScaler()),
    ]


def split_data(X, y, test_size=0.25, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=random_state)


def fit_preprocess(X_train, X_test, y_train, apply_smote=True,
                   random_state=RANDOM_STATE):
    """
    Fit preprocessing on X_train only; transform both sets.
    SMOTE applied to training data only — never to test.
    Returns (X_train_proc, X_test_proc, y_train_proc, fitted_pipeline).
    """
    steps = make_preprocessing_steps()
    preproc = ImbPipeline(steps)
    preproc.fit(X_train, y_train)

    X_tr = preproc.transform(X_train)
    X_te = preproc.transform(X_test)

    y_tr_raw = y_train.values if hasattr(y_train, 'values') else np.array(y_train)

    if apply_smote:
        n1 = int((y_tr_raw == 1).sum())
        k  = max(1, min(5, n1 - 1))
        smote = SMOTE(k_neighbors=k, random_state=random_state)
        X_tr, y_tr_out = smote.fit_resample(X_tr, y_tr_raw)
    else:
        y_tr_out = y_tr_raw

    return X_tr, X_te, np.array(y_tr_out), preproc


def load_dataset(path, sep=";"):
    df = pd.read_csv(path, sep=sep)
    df.columns = df.columns.str.strip()
    df = df.loc[:, df.columns != ""]
    return df


HIST_COLS = [
    "numberOfBugsFoundUntil:",
    "numberOfNonTrivialBugsFoundUntil:",
    "numberOfMajorBugsFoundUntil:",
    "numberOfCriticalBugsFoundUntil:",
    "numberOfHighPriorityBugsFoundUntil:",
]


def make_X_y(df_raw, name=""):
    avail = [c for c in HIST_COLS if c in df_raw.columns]
    X = df_raw[avail].copy().astype(float)
    y = (df_raw["bugs"] > 0).astype(int)
    if name:
        print(f"{name}: X={X.shape}  "
              f"class_dist={dict(y.value_counts().sort_index())}")
    return X, y
