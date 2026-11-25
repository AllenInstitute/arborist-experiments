"""One result per input graph, and the adapters that compact a batch of them.

Results are an aligned list, one per input graph, rather than a compacted batch: the caller still
holds the graphs it passed, so no ids have to travel with the rows (`utils.cells.graph_labels`
recovers them), and a skipped graph keeps its position.

So there is no batch type here. A dense matrix is formed only where one is needed, by
`design_matrix`; nothing that computes a feature, and nothing that post-processes one, has a matrix
in its signature.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

# What a caller wants carried alongside each row. Usually the graph's id, but a driver that also needs
# `y` passes `(id, class_index)` pairs and gets the surviving ones back, so no id-to-label join is
# needed.
T = TypeVar("T")


@dataclass
class FeatureResult:
    """One graph's features, or the reason there are none.

    `feature_names` is carried per element, not per batch, and a skipped element carries it too
    wherever the feature knows its own width from its config: that is what lets a batch in which
    every graph was skipped still report a width.
    """

    X: np.ndarray | None                # 1-D float64, or None when this graph was skipped
    feature_names: list[str]
    skipped: str | None                 # the reason, or None


def batch_names(results: list[FeatureResult]) -> list[str]:
    """The batch's feature names: the first element that has any.

    Every extractor in this package declares its width from its own constants and puts the names
    on every element, skipped ones included, so this reads them even from an all-skipped batch. It
    still scans rather than reading `results[0]` because a caller can build results by hand, and
    because `drop_zero_variance_and_impute` is free to narrow the set.
    """
    for r in results:
        if r.feature_names:
            return r.feature_names
    return []


def width_reason(r: FeatureResult, names: list[str]) -> str | None:
    """Why this result cannot join the batch, or None.

    A row of a different width is dropped rather than raising, since one odd cell must not cost the
    cohort.
    """
    if r.X is None:
        return "no features and no reason"      # a self-inconsistent result, not a real path
    if len(r.X) != len(names):
        return f"width {len(r.X)} vs batch {len(names)}"
    return None


def usable(r: FeatureResult, names: list[str]) -> bool:
    """Does this result contribute a row to the batch?"""
    return r.skipped is None and width_reason(r, names) is None


def design_matrix(
    results: list[FeatureResult], labels: Sequence[T]
) -> tuple[np.ndarray, list[T]]:
    """The surviving rows as a design matrix, with the labels of the graphs they came from.

    `(0, len(batch_names))` when nothing survived, never `(0,)`: a caller that concatenates blocks
    needs the width even when the block is empty. A row whose width disagrees with the batch is
    dropped rather than raising, since one odd cell must not cost the cohort.
    """
    names = batch_names(results)
    rows, kept = [], []
    for label, r in zip(labels, results):
        if not usable(r, names):
            continue
        rows.append(r.X)
        kept.append(label)
    X = np.vstack(rows) if rows else np.zeros((0, len(names)))
    return X, kept


def concat_features(*batches: list[FeatureResult]) -> list[FeatureResult]:
    """Several aligned batches joined column-wise, still one result per graph.

    For a caller who wants several features as one block, e.g. `radial_hist` + `radial_sholl`.

    An element is skipped if it was skipped in any batch, since a block cannot be half a row; the first
    reason is kept. Names concatenate whether or not the element survived, so a batch of features that
    each declare their width still declares the joint one.
    """
    if not batches:
        return []
    n = len(batches[0])
    if any(len(b) != n for b in batches):
        raise ValueError(f"batches are not aligned: {[len(b) for b in batches]}")
    out = []
    for parts in zip(*batches):
        names = [name for p in parts for name in p.feature_names]
        reason = next((p.skipped for p in parts if p.skipped is not None), None)
        X = None if reason is not None else np.concatenate([p.X for p in parts])
        out.append(FeatureResult(X=X, feature_names=names, skipped=reason))
    return out


def skips(results: list[FeatureResult], labels: Sequence[T]) -> list[tuple[T, str]]:
    """Every skipped graph as `(label, reason)`, in input order.

    Includes the width disagreement `design_matrix` drops, so the two together account for every
    input graph exactly once.
    """
    names = batch_names(results)
    out = []
    for label, r in zip(labels, results):
        reason = r.skipped if r.skipped is not None else width_reason(r, names)
        if reason is not None:
            out.append((label, reason))
    return out
