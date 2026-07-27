"""
optimization.py
Optuna hyperparameter optimization.
Uses 3-fold inner CV to evaluate each trial — fast but unbiased.
SMOTE is applied inside each fold via ImbPipeline (no leakage).
"""
import numpy as np
import optuna
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_TRIALS_DEFAULT = 40
INNER_CV_SPLITS  = 3


def _cv_score(pipeline, X, y, metric="f1"):
    skf = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)
    scores = []
    for tr_i, val_i in skf.split(X, y):
        X_tr, X_val = X[tr_i], X[val_i]
        y_tr, y_val = y[tr_i], y[val_i]
        try:
            pipeline.fit(X_tr, y_tr)
            if metric == "f1":
                yp = pipeline.predict(X_val)
                scores.append(f1_score(y_val, yp, zero_division=0))
            else:
                pp = pipeline.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, pp))
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores))


def _smote_step(y_train):
    n1 = int((y_train == 1).sum())
    k  = max(1, min(5, n1 - 1))
    return ("smote", SMOTE(k_neighbors=k, random_state=RANDOM_STATE))


def make_xgb_objective(preproc_steps, X_train, y_train):
    n0 = int((y_train == 0).sum())
    n1 = int((y_train == 1).sum())
    spw = n0 / max(n1, 1)
    smote_step = _smote_step(y_train)

    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 600),
            max_depth        = trial.suggest_int("max_depth", 3, 8),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            gamma            = trial.suggest_float("gamma", 0.0, 1.0),
            # scale_pos_weight added in build_tuned_models — kept out of best_params
        )
        model = XGBClassifier(**params, scale_pos_weight=spw,
                               eval_metric="logloss",
                               random_state=RANDOM_STATE, verbosity=0)
        pipe = ImbPipeline(preproc_steps + [smote_step, ("model", model)])
        return _cv_score(pipe, X_train, y_train)
    return objective


def make_lgb_objective(preproc_steps, X_train, y_train):
    smote_step = _smote_step(y_train)

    def objective(trial):
        params = dict(
            n_estimators  = trial.suggest_int("n_estimators", 100, 600),
            max_depth     = trial.suggest_int("max_depth", 3, 8),
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves    = trial.suggest_int("num_leaves", 15, 63),
            reg_alpha     = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda    = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            subsample     = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
            class_weight  = "balanced",
        )
        model = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE, verbose=-1)
        pipe = ImbPipeline(preproc_steps + [smote_step, ("model", model)])
        return _cv_score(pipe, X_train, y_train)
    return objective


def make_cat_objective(preproc_steps, X_train, y_train):
    smote_step = _smote_step(y_train)

    def objective(trial):
        params = dict(
            iterations    = trial.suggest_int("iterations", 100, 600),
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            depth         = trial.suggest_int("depth", 3, 8),
            l2_leaf_reg   = trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            auto_class_weights = "Balanced",
        )
        model = CatBoostClassifier(**params, random_seed=RANDOM_STATE, verbose=0)
        pipe = ImbPipeline(preproc_steps + [smote_step, ("model", model)])
        return _cv_score(pipe, X_train, y_train)
    return objective


def make_rf_objective(preproc_steps, X_train, y_train):
    smote_step = _smote_step(y_train)

    def objective(trial):
        params = dict(
            n_estimators  = trial.suggest_int("n_estimators", 100, 500),
            max_depth     = trial.suggest_int("max_depth", 3, 15),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
            max_features  = trial.suggest_categorical("max_features",
                                                       ["sqrt", "log2", None]),
            class_weight  = "balanced",
        )
        model = RandomForestClassifier(**params, random_state=RANDOM_STATE)
        pipe = ImbPipeline(preproc_steps + [smote_step, ("model", model)])
        return _cv_score(pipe, X_train, y_train)
    return objective


def make_lr_objective(preproc_steps, X_train, y_train):
    smote_step = _smote_step(y_train)

    def objective(trial):
        C       = trial.suggest_float("C", 1e-3, 100.0, log=True)
        solver  = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        penalty = "l2"
        params  = dict(C=C, solver=solver, penalty=penalty,
                       max_iter=2000, class_weight="balanced",
                       random_state=RANDOM_STATE)
        model = LogisticRegression(**params)
        pipe = ImbPipeline(preproc_steps + [smote_step, ("model", model)])
        return _cv_score(pipe, X_train, y_train)
    return objective


OBJECTIVES = {
    "XGBoost":             make_xgb_objective,
    "LightGBM":            make_lgb_objective,
    "CatBoost":            make_cat_objective,
    "Random Forest":       make_rf_objective,
    "Logistic Regression": make_lr_objective,
}


def run_optuna_study(model_name: str, preproc_steps: list,
                     X_train: np.ndarray, y_train: np.ndarray,
                     n_trials: int = N_TRIALS_DEFAULT,
                     random_state: int = RANDOM_STATE) -> dict:
    if model_name not in OBJECTIVES:
        print(f"  No Optuna objective for {model_name}. Skipping.")
        return {}

    objective_fn = OBJECTIVES[model_name](preproc_steps, X_train, y_train)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)

    print(f"  {model_name}: best F1={study.best_value:.4f}  "
          f"params={study.best_params}")
    return study.best_params


def run_all_optuna(preproc_steps: list,
                   X_train: np.ndarray, y_train: np.ndarray,
                   n_trials: int = N_TRIALS_DEFAULT,
                   models_to_tune: list = None) -> dict:
    """
    Run Optuna for each requested model.
    Returns dict: model_name -> best_params.
    """
    if models_to_tune is None:
        models_to_tune = list(OBJECTIVES.keys())

    best_params = {}
    for mname in models_to_tune:
        print(f"\nOptimizing {mname} ({n_trials} trials)...")
        best_params[mname] = run_optuna_study(
            mname, preproc_steps, X_train, y_train, n_trials=n_trials)
    return best_params
