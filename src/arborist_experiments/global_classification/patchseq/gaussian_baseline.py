"""Generative point-cloud baseline: one 3D Gaussian per subclass.

Pools the node coordinates of every training cell in a class and fits a single
multivariate normal to them, then classifies a held-out cell by its mean
per-node log-likelihood under each class Gaussian plus the class log-prior.

Two non-obvious choices:
  - `mean` rather than `sum` over nodes, so the score does not scale with
    reconstruction node count (nodes are not i.i.d. samples -- denser tracings
    would otherwise dominate).
  - a softmax temperature fitted on inner folds, so the resulting posteriors are
    on a comparable footing with the log-loss of the discriminative rows.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    log_loss, matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold

from arborist_experiments.global_classification.cv.classifier import CVResult
from .modality import filter_cell


COV_RIDGE = 1e-3
TEMPERATURES = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class ClassGaussians:
    means: np.ndarray       # (K, D)
    precisions: np.ndarray  # (K, D, D)
    logdets: np.ndarray     # (K,)
    log_priors: np.ndarray  # (K,)


def _fit_one(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    d = points.shape[1]
    mu = points.mean(axis=0)
    cov = np.atleast_2d(np.cov(points, rowvar=False))
    cov = cov + np.eye(d) * COV_RIDGE * max(np.trace(cov) / d, 1e-12)
    _, logdet = np.linalg.slogdet(cov)
    return mu, np.linalg.inv(cov), float(logdet)


def fit_gaussians(clouds: list[np.ndarray], y: np.ndarray, n_classes: int) -> ClassGaussians:
    d = clouds[0].shape[1]
    # A class absent from this fit set falls back to the pooled Gaussian rather
    # than to zero probability, which would make log-loss infinite.
    fb = _fit_one(np.concatenate(clouds, axis=0))

    means = np.zeros((n_classes, d))
    precisions = np.zeros((n_classes, d, d))
    logdets = np.zeros(n_classes)
    counts = np.zeros(n_classes)

    for k in range(n_classes):
        idx = np.flatnonzero(y == k)
        if idx.size == 0:
            means[k], precisions[k], logdets[k] = fb
            continue
        pts = np.concatenate([clouds[i] for i in idx], axis=0)
        means[k], precisions[k], logdets[k] = _fit_one(pts)
        counts[k] = idx.size

    log_priors = np.log((counts + 0.5) / (counts.sum() + 0.5 * n_classes))
    return ClassGaussians(means, precisions, logdets, log_priors)


def mean_loglik(g: ClassGaussians, clouds: list[np.ndarray]) -> np.ndarray:
    n_classes, d = g.means.shape
    out = np.empty((len(clouds), n_classes))
    const = -0.5 * d * LOG_2PI
    for i, pts in enumerate(clouds):
        for k in range(n_classes):
            delta = pts - g.means[k]
            maha = np.einsum("ij,jk,ik->i", delta, g.precisions[k], delta)
            out[i, k] = -0.5 * float(np.mean(maha)) - 0.5 * g.logdets[k] + const
    return out


def posterior(loglik: np.ndarray, log_priors: np.ndarray, temperature: float) -> np.ndarray:
    scores = temperature * loglik + log_priors
    scores = scores - scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    return p / p.sum(axis=1, keepdims=True)


def _pick_temperature(clouds, y, n_classes, inner_k, seed) -> float:
    labels = np.arange(n_classes)
    inner = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=seed)
    losses = np.zeros(len(TEMPERATURES))
    for tr, te in inner.split(np.zeros(len(y)), y):
        g = fit_gaussians([clouds[i] for i in tr], y[tr], n_classes)
        ll = mean_loglik(g, [clouds[i] for i in te])
        for j, t in enumerate(TEMPERATURES):
            losses[j] += log_loss(y[te], posterior(ll, g.log_priors, t), labels=labels)
    return float(TEMPERATURES[int(np.argmin(losses))])


def build_clouds(cohort, modality: str) -> tuple[list[np.ndarray], np.ndarray, list[tuple[str, str]]]:
    clouds: list[np.ndarray] = []
    labels: list[int] = []
    skipped: list[tuple[str, str]] = []
    for cell, lbl in zip(cohort.cells, cohort.labels):
        sub = filter_cell(cell, modality)
        if sub is None:
            skipped.append((cell.specimen_id, f"no {modality} nodes"))
            continue
        clouds.append(np.ascontiguousarray(sub.xyz, dtype=np.float64))
        labels.append(int(lbl))
    return clouds, np.array(labels, dtype=np.int64), skipped


def run_gaussian_cv(
    cohort,
    modality: str,
    outer_k: int = 5,
    inner_k: int = 3,
    seed: int = 0,
) -> tuple[Optional[CVResult], list[tuple[str, str]]]:
    clouds, y_raw, skipped = build_clouds(cohort, modality)
    if len(clouds) < 30:
        return None, skipped

    present = np.unique(y_raw)
    class_names = [cohort.label_names[i] for i in present]
    remap = {int(v): i for i, v in enumerate(present)}
    y = np.array([remap[int(v)] for v in y_raw], dtype=np.int64)
    n_classes = len(present)
    if n_classes < 2:
        return None, skipped

    labels = np.arange(n_classes)
    outer = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=seed)
    metrics = {"log_loss": [], "accuracy": [], "balanced_accuracy": [],
               "f1_macro": [], "mcc": []}
    conf = np.zeros((n_classes, n_classes), dtype=int)
    best_params: list[dict] = []

    for tr, te in outer.split(np.zeros(len(y)), y):
        temperature = _pick_temperature([clouds[i] for i in tr], y[tr], n_classes, inner_k, seed)
        g = fit_gaussians([clouds[i] for i in tr], y[tr], n_classes)
        proba = posterior(mean_loglik(g, [clouds[i] for i in te]), g.log_priors, temperature)
        y_pred = proba.argmax(axis=1)

        best_params.append({"temperature": temperature})
        metrics["log_loss"].append(float(log_loss(y[te], proba, labels=labels)))
        metrics["accuracy"].append(float(accuracy_score(y[te], y_pred)))
        metrics["balanced_accuracy"].append(float(balanced_accuracy_score(y[te], y_pred)))
        metrics["f1_macro"].append(float(f1_score(y[te], y_pred, average="macro", labels=labels, zero_division=0.0)))
        metrics["mcc"].append(float(matthews_corrcoef(y[te], y_pred)))
        conf += confusion_matrix(y[te], y_pred, labels=labels)

    return CVResult(
        metrics=metrics,
        confusion=conf,
        best_params=best_params,
        n_samples=len(clouds),
        n_features=3,
        n_classes=n_classes,
        class_names=class_names,
    ), skipped
