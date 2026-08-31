"""Writers for the per-cell record a scored run produces.

Everything that used to reach disk from a `CVResult` was lossy: five mean/std metric rows and a
confusion PNG. The out-of-fold probability matrix is the sufficient statistic -- the metrics, the
aggregated and per-fold confusion matrices, per-class recall and any cohort split all follow from
`(proba, y, fold, specimen_ids)` -- so that is what gets written, and the confusion matrices go out
as CSV as well as pixels.
"""

from __future__ import annotations

import csv

import numpy as np


def write_oof(path: str, proba: np.ndarray, y: np.ndarray, fold: np.ndarray,
              specimen_ids, class_names) -> str:
    """The out-of-fold record, one row per cell in the order the classifier saw them.

    `specimen_ids` is required, not optional: probabilities that cannot be joined back to cells
    cannot answer the questions this file exists for.
    """
    proba = np.asarray(proba)
    if len(specimen_ids) != len(proba) or len(y) != len(proba) or len(fold) != len(proba):
        raise ValueError(f"oof length mismatch: proba {len(proba)}, y {len(y)}, "
                         f"fold {len(fold)}, ids {len(specimen_ids)}")
    if proba.shape[1] != len(class_names):
        raise ValueError(f"oof width {proba.shape[1]} vs {len(class_names)} class names")
    np.savez(path,
             proba=proba.astype(np.float64),
             y=np.asarray(y, dtype=np.int64),
             fold=np.asarray(fold, dtype=np.int64),
             specimen_ids=np.array([str(s) for s in specimen_ids], dtype=str),
             class_names=np.array([str(s) for s in class_names], dtype=str))
    return path


def write_confusion(path: str, conf: np.ndarray, class_names, per_fold=None) -> str:
    """Confusion counts as data. `fold=all` is the sum the PNG shows; 0..k-1 are the folds.

    The aggregate alone cannot say whether a class is misread consistently or in one fold only,
    and `conf +=` in the CV loop used to destroy the per-fold matrices before anything saw them.
    """
    conf = np.asarray(conf, dtype=np.int64)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "true_label"] + [f"pred_{c}" for c in class_names])
        for i, name in enumerate(class_names):
            w.writerow(["all", name] + [int(v) for v in conf[i]])
        if per_fold is not None:
            per_fold = np.asarray(per_fold, dtype=np.int64)
            for k in range(per_fold.shape[0]):
                for i, name in enumerate(class_names):
                    w.writerow([k, name] + [int(v) for v in per_fold[k, i]])
    return path
