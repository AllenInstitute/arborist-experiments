"""Nested-CV elastic-net classifier, shared by every morphology benchmark in this repo.

One implementation, two datasets. Patch-seq and MICrONS ran separate copies of this for a while,
on the theory that the Patch-seq numbers were a baseline that editing would put under suspicion.
That was wrong: the objective is the reference, not whichever optimizer happened to run first.
`solver="gpu"` (FISTA proximal gradient, gpu_logreg) and `solver="saga"` minimize the same
elastic-net multinomial objective, the equivalence tests assert `objective(gpu) <= objective(saga)`
to within 1e-6 relative, and at the C=10 corner FISTA reaches a *lower* objective in 2.5 s than
saga does in 195 s. So gpu is the default everywhere and saga is kept only as the test fixture.

Grid, inner-loop scoring, metrics, PCA cap and fold seeds are identical to what both packages had,
so nothing about the numbers moves except the solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import warnings

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    log_loss, matthews_corrcoef,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .gpu_logreg import TorchElasticNetLogReg


N_JOBS = max(1, (os.cpu_count() or 1) - 1)

GRID = {
    "clf__C":        [0.1, 1.0, 10.0],
    "clf__l1_ratio": [0.0, 0.5, 1.0],
}

JOBLIB_CACHE = "/root/capsule/scratch/cv_joblib_cache"


@dataclass
class CVResult:
    metrics: dict[str, list[float]]        # metric_name -> list of per-outer-fold values
    confusion: np.ndarray                  # aggregated across outer folds
    best_params: list[dict]                # per outer fold
    n_samples: int
    n_features: int
    n_classes: int
    class_names: list[str]
    # The per-cell out-of-fold record and the unaggregated confusion matrices. Optional because
    # run_gaussian_cv and the TreeMoCo fine-tune path build a CVResult by hand and fill what they
    # have; a result carrying only the aggregate is still valid, just lossy.
    #
    # oof_* rows are in the order the classifier was handed them. The outer StratifiedKFold test
    # sets are disjoint and cover every sample, so run_cv leaves no row unfilled.
    oof_proba: np.ndarray | None = None
    oof_pred: np.ndarray | None = None
    oof_fold: np.ndarray | None = None
    confusion_per_fold: np.ndarray | None = None


def boundary_hits(best_params: list[dict], grid: dict | None = None) -> list[str]:
    """Grid-edge selections as data, for the metrics CSV.

    Same test as _boundary_warnings, which only ever printed. OQ-1 says the suite must be rerun
    with a wider grid if C=10 starts winning at full n, and that tripwire cannot live only in a
    log line nobody greps.
    """
    if grid is None:
        grid = GRID
    hits: list[str] = []
    for fold, params in enumerate(best_params):
        for key, values in grid.items():
            v = params.get(key)
            if v is None:
                continue
            if v == max(values) or v == min(values):
                edge = "max" if v == max(values) else "min"
                hits.append(f"fold{fold}:{key}={v}@{edge}")
    return hits


def _make_pipeline(
    n_samples: int, n_features: int, random_state: int,
    solver: str = "saga", dtype: str = "float32", memory=None,
) -> Pipeline:
    """The scored pipeline. `solver` defaults to saga *here* but to gpu in run_cv, deliberately:
    the direct callers of this function (the TreeMoCo probes' per-fold fitter) drive their own
    GridSearchCV with n_jobs=N_JOBS, and forked workers cannot inherit a CUDA context.
    """
    pca_cap = max(2, min(n_samples // 2, 200, n_features))
    if solver == "gpu":
        clf = TorchElasticNetLogReg(dtype=dtype, random_state=random_state)
    elif solver == "saga":
        clf = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000,
                                 random_state=random_state)
    else:
        raise ValueError(f"Unknown solver {solver!r}")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_cap, random_state=random_state)),
        ("clf", clf),
    ], memory=memory)


def _boundary_warnings(params: dict, grid: dict, fold: int, log=None) -> None:
    """Flag a selected hyperparameter sitting on a grid edge.

    At 31x the Patch-seq sample size the optimum should move toward less regularization. If
    C=10 starts winning the grid is too narrow and the suite needs rerunning with a wider one,
    so it should be visible in the run log rather than discovered at review time.
    """
    for key, values in grid.items():
        v = params.get(key)
        if v is None:
            continue
        if v == max(values) or v == min(values):
            edge = "max" if v == max(values) else "min"
            msg = f"  fold {fold}: {key}={v} is the {edge} of the grid {values}"
            print(msg, flush=True)
            if log is not None:
                log(msg)


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    outer_k: int = 5,
    inner_k: int = 3,
    grid: dict | None = None,
    seed: int = 0,
    solver: str = "gpu",
    dtype: str = "float32",
    log=None,
) -> CVResult:
    if grid is None:
        grid = GRID

    n_samples, n_features = X.shape
    classes = np.unique(y)
    outer = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=seed)

    # One CUDA context cannot be shared by forked workers, and spawned ones would serialize on
    # the single GPU anyway. Instead of parallel fits, memoize the scaler+PCA prefix: it is
    # constant across the 9 grid points, so this fits PCA once per inner split rather than 9x.
    if solver == "gpu":
        n_jobs = 1
        os.makedirs(JOBLIB_CACHE, exist_ok=True)
        memory = joblib.Memory(location=JOBLIB_CACHE, verbose=0)
    else:
        n_jobs, memory = N_JOBS, None

    metrics = {"log_loss": [], "accuracy": [], "balanced_accuracy": [],
               "f1_macro": [], "mcc": []}
    conf = np.zeros((len(classes), len(classes)), dtype=int)
    conf_per_fold = np.zeros((outer_k, len(classes), len(classes)), dtype=int)
    oof_proba = np.full((n_samples, len(classes)), np.nan)
    oof_pred = np.full(n_samples, -1, dtype=np.int64)
    oof_fold = np.full(n_samples, -1, dtype=np.int64)
    best_params: list[dict] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        for fold, (tr, te) in enumerate(outer.split(X, y)):
            pipe = _make_pipeline(len(tr), n_features, seed, solver=solver, dtype=dtype,
                                  memory=memory)
            inner = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=seed)
            gs = GridSearchCV(pipe, grid, cv=inner, scoring="neg_log_loss",
                              n_jobs=n_jobs, refit=True)
            gs.fit(X[tr], y[tr])
            best_params.append(gs.best_params_)
            _boundary_warnings(gs.best_params_, grid, fold, log)

            y_pred = gs.predict(X[te])
            y_proba = gs.predict_proba(X[te])

            metrics["log_loss"].append(float(log_loss(y[te], y_proba, labels=classes)))
            metrics["accuracy"].append(float(accuracy_score(y[te], y_pred)))
            metrics["balanced_accuracy"].append(float(balanced_accuracy_score(y[te], y_pred)))
            metrics["f1_macro"].append(float(f1_score(y[te], y_pred, average="macro",
                                                      labels=classes, zero_division=0.0)))
            metrics["mcc"].append(float(matthews_corrcoef(y[te], y_pred)))
            conf_per_fold[fold] = confusion_matrix(y[te], y_pred, labels=classes)
            conf += conf_per_fold[fold]
            oof_proba[te] = y_proba
            oof_pred[te] = y_pred
            oof_fold[te] = fold

    if memory is not None:
        memory.clear(warn=False)

    return CVResult(
        metrics=metrics,
        confusion=conf,
        best_params=best_params,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=len(classes),
        class_names=class_names,
        oof_proba=oof_proba,
        oof_pred=oof_pred,
        oof_fold=oof_fold,
        confusion_per_fold=conf_per_fold,
    )
