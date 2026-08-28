"""Cohort construction for the MICrONS cells, from the extracted caches.

Both cohorts are built here, but they are never built jointly: "has apical" and "has axon"
almost perfectly separate the two source files, so a combined classifier would be reading the
acquisition pipeline rather than morphology.

Differences from the Patch-seq loader that a reader coming from there will look for:

  - No `_reparent_to_soma`. Every cell in both caches has exactly one soma node and exactly one
    root, and the root *is* the soma, so the trees are already connected. `synthetic_edge` is
    therefore all-False and the fragmentation confound that inflated Patch-seq cable length
    by ~29% does not exist here.
  - Joins are by specimen-id string, never by row index. The excitatory source groups have
    different cardinalities (49803 swc_* vs 49795 meta) because 8 specimens failed the conformal
    stage, so a shared-index assumption would silently misalign coordinates against types. The
    inhibitory cache is keyed by nucleus id, resolved geometrically in `inh_id_map`.
  - Radius is constant 1.0. The excitatory file has none; the inhibitory one does, but it is
    discarded by default so the two cohorts' morphometrics mean the same thing -- the Patch-seq
    `three_fixes` correction showed radius-derived statistics are exactly where a depth-like leak
    hides. `real_radius=True` is available for a separately labelled sensitivity run.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import h5py
import numpy as np

from arborist_experiments.patchseq_morphology_benchmark.data_loading import (
    Cell, Cohort, TYPE_SOMA, TYPE_AXON, TYPE_BASAL, TYPE_APICAL,
)


CACHE_DIR = "/root/capsule/scratch/microns_cache/v1"
CACHE_PATH = os.path.join(CACHE_DIR, "em_exc_swc.h5")
LABELS_PATH = os.path.join(CACHE_DIR, "coarse_labels.csv")
INH_CACHE_PATH = os.path.join(CACHE_DIR, "microns_inh_swc.h5")
INH_LABELS_PATH = os.path.join(CACHE_DIR, "inh_labels.csv")

# The 8 classes surviving exc_only + min_class_size=25 on coarse_label.
EXC_COARSE_CLASSES = ("L2IT", "L3IT", "L4IT", "L5ET", "L5IT", "L5NP", "L6CT", "L6IT")

# The 11 inhibitory classes surviving min_class_size=25 on the 7550 labelled inh cells:
# PV 2967, ITC 1261, MC 1246, DTC 339, NMC 283, NGC 249, L1 234, ITCperi 168, AltBasket 166,
# AltDTC 152, ChC 29 -> 7094 cells, chance balanced accuracy 1/11 = 0.0909. The remaining 456
# labelled cells carry excitatory labels -- the mirror image of the ~747 inhibitory-labelled
# cells in the excitatory file (R-7) -- and are dropped by default.
INH_COARSE_CLASSES = ("AltBasket", "AltDTC", "ChC", "DTC", "ITC", "ITCperi", "L1", "MC",
                      "NGC", "NMC", "PV")

__all__ = [
    "Cell", "Cohort", "SwcCache", "TYPE_SOMA", "TYPE_AXON", "TYPE_BASAL", "TYPE_APICAL",
    "EXC_COARSE_CLASSES", "INH_COARSE_CLASSES", "load_cache", "cell_from_cache", "load_labels",
    "build_cohort", "build_inh_cohort", "build_cohort_from_source", "SOURCES",
    "subsample_proportional", "subsample_per_class", "class_counts", "soma_depth_features",
]


@dataclass
class SwcCache:
    specimens: list[str]
    offsets: np.ndarray
    xyz: np.ndarray
    types: np.ndarray
    parents: np.ndarray
    pia_y: np.ndarray
    wm_y: np.ndarray
    row_by_id: dict[str, int]
    # Present only in the inhibitory cache; the excitatory file has no radius column.
    radius: np.ndarray | None = None


def load_cache(cache_path: str = CACHE_PATH) -> SwcCache:
    with h5py.File(cache_path, "r") as f:
        if not bool(f.attrs.get("complete", False)):
            raise RuntimeError(
                f"{cache_path} is incomplete ({int(f.attrs.get('n_cells_done', 0))} cells); "
                "rerun cache_extract to finish it"
            )
        specimens = [s.decode() for s in f["specimens"][:]]
        cache = SwcCache(
            specimens=specimens,
            offsets=f["offsets"][:],
            xyz=f["xyz"][:],
            types=f["types"][:],
            parents=f["parents"][:],
            pia_y=f["pia_y"][:],
            wm_y=f["wm_y"][:],
            row_by_id={s: i for i, s in enumerate(specimens)},
            radius=f["radius"][:] if "radius" in f else None,
        )
    assert len(cache.row_by_id) == len(specimens), "duplicate specimen ids in cache"
    return cache


def cell_from_cache(cache: SwcCache, row: int, real_radius: bool = False) -> Cell:
    lo, hi = int(cache.offsets[row]), int(cache.offsets[row + 1])
    n = hi - lo
    if real_radius:
        if cache.radius is None:
            raise ValueError("this cache has no radius column; real_radius is unavailable")
        radius = cache.radius[lo:hi].astype(np.float64)
    else:
        radius = np.ones(n, dtype=np.float64)
    return Cell(
        specimen_id=cache.specimens[row],
        types=cache.types[lo:hi].astype(np.int32),
        parents=cache.parents[lo:hi].astype(np.int64),
        xyz=cache.xyz[lo:hi].astype(np.float64),
        radius=radius,
        synthetic_edge=np.zeros(n, dtype=bool),
    )


def load_labels(labels_path: str = LABELS_PATH, label_column: str = "coarse_label") -> dict[str, str]:
    import pandas as pd

    read = pd.read_parquet if labels_path.endswith(".parquet") else pd.read_csv
    df = read(labels_path)
    if label_column not in df.columns:
        raise ValueError(f"{label_column!r} not in {list(df.columns)}")
    ids = df["cell_id"].astype(str)
    labs = df[label_column].astype(str)
    return {i: s for i, s in zip(ids, labs) if s and s != "nan"}


def class_counts(cohort: Cohort) -> dict[str, int]:
    counts = {name: 0 for name in cohort.label_names}
    for k in cohort.labels:
        counts[cohort.label_names[int(k)]] += 1
    return counts


def _assemble(cells: list[Cell], labels_str: list[str], label_by_id: dict[str, str]) -> Cohort:
    label_names = sorted(set(labels_str))
    index = {name: i for i, name in enumerate(label_names)}
    labels = np.array([index[s] for s in labels_str], dtype=np.int64)
    return Cohort(cells=cells, labels=labels, label_names=label_names,
                  subclass_by_id=label_by_id)


def build_cohort(
    cache_path: str = CACHE_PATH,
    labels_path: str = LABELS_PATH,
    label_column: str = "coarse_label",
    min_class_size: int = 25,
    exc_only: bool = True,
    allowed_ids: set[str] | None = None,
    keep_classes: tuple[str, ...] | None = None,
    real_radius: bool = False,
    require_types: tuple[int, ...] | None = None,
    log=print,
) -> Cohort:
    cache = load_cache(cache_path)
    label_by_id = load_labels(labels_path, label_column)
    log(f"cache: {len(cache.specimens)} cells; labels: {len(label_by_id)} ids")

    rows: list[int] = []
    labels_str: list[str] = []
    for row, sid in enumerate(cache.specimens):
        if allowed_ids is not None and sid not in allowed_ids:
            continue
        lab = label_by_id.get(sid)
        if lab is None:
            continue
        rows.append(row)
        labels_str.append(lab)
    log(f"joined on specimen id: {len(rows)} cells")

    if exc_only:
        # EXC_COARSE_CLASSES are coarse_label values, so this filter would drop every cell on
        # any other label column. Fail rather than return an empty cohort.
        if label_column != "coarse_label":
            raise ValueError(
                f"exc_only filters on coarse_label values, so it cannot be combined with "
                f"label_column={label_column!r}; pass exc_only=False"
            )
        keep = [k for k, lab in enumerate(labels_str) if lab in EXC_COARSE_CLASSES]
        log(f"exc_only: dropped {len(rows) - len(keep)} non-excitatory-labelled cells")
        rows = [rows[k] for k in keep]
        labels_str = [labels_str[k] for k in keep]

    if keep_classes is not None:
        if label_column != "coarse_label":
            raise ValueError(
                f"keep_classes holds coarse_label values, so it cannot be combined with "
                f"label_column={label_column!r}; pass keep_classes=None"
            )
        keep = [k for k, lab in enumerate(labels_str) if lab in keep_classes]
        log(f"keep_classes: dropped {len(rows) - len(keep)} cells outside the "
            f"{len(keep_classes)} requested classes")
        rows = [rows[k] for k in keep]
        labels_str = [labels_str[k] for k in keep]

    if require_types:
        # Modality filtering drops cells that have no cable of the requested type, so an `axon`
        # row scores a different (smaller) cohort than a `dendrite` row. Restricting the cohort
        # up front is what makes dendrite-vs-axon-vs-full a matched comparison rather than a
        # comparison confounded with sample size and class prior.
        keep = [k for k, row in enumerate(rows)
                if all((cache.types[int(cache.offsets[row]):int(cache.offsets[row + 1])] == t).any()
                       for t in require_types)]
        log(f"require_types={list(require_types)}: dropped {len(rows) - len(keep)} cells")
        rows = [rows[k] for k in keep]
        labels_str = [labels_str[k] for k in keep]

    counts: dict[str, int] = {}
    for lab in labels_str:
        counts[lab] = counts.get(lab, 0) + 1
    surviving = {lab for lab, n in counts.items() if n >= min_class_size}
    dropped = {lab: n for lab, n in counts.items() if lab not in surviving}
    if dropped:
        log(f"min_class_size={min_class_size}: dropped {dropped}")

    keep = [k for k, lab in enumerate(labels_str) if lab in surviving]
    rows = [rows[k] for k in keep]
    labels_str = [labels_str[k] for k in keep]

    cells = [cell_from_cache(cache, row, real_radius=real_radius) for row in rows]
    cohort = _assemble(cells, labels_str, label_by_id)
    cohort.pia_y = cache.pia_y[rows]
    cohort.wm_y = cache.wm_y[rows]
    log(f"cohort: {len(cohort)} cells, {len(cohort.label_names)} classes {class_counts(cohort)}")
    return cohort


def build_inh_cohort(
    cache_path: str = INH_CACHE_PATH,
    labels_path: str = INH_LABELS_PATH,
    label_column: str = "coarse_label",
    min_class_size: int = 25,
    inh_only: bool = True,
    allowed_ids: set[str] | None = None,
    real_radius: bool = False,
    require_types: tuple[int, ...] | None = None,
    log=print,
) -> Cohort:
    """The inhibitory cohort. Same `Cohort` contract, different cache and class set.

    `pia_y`/`wm_y` come from the affine fits in `inh_id_map.derive_upright` rather than from
    per-cell boundary polylines, so the `soma_depth` control is a smooth function of position
    here. It still orders the laminar labels monotonically on cells that had no part in the fit
    (L1 88 um from pia, ChC 95, up to DTC 510), which is what validates the derived frame.
    """
    return build_cohort(
        cache_path=cache_path, labels_path=labels_path, label_column=label_column,
        min_class_size=min_class_size, exc_only=False, allowed_ids=allowed_ids,
        keep_classes=INH_COARSE_CLASSES if inh_only else None,
        real_radius=real_radius, require_types=require_types, log=log,
    )


SOURCES = {"em_exc": build_cohort, "microns_inh": build_inh_cohort}


def build_cohort_from_source(source: str = "em_exc", **kwargs) -> Cohort:
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(SOURCES)}")
    return SOURCES[source](**kwargs)


def _subset(cohort: Cohort, keep: np.ndarray) -> Cohort:
    cells = [cohort.cells[i] for i in keep]
    labels_str = [cohort.label_names[int(cohort.labels[i])] for i in keep]
    sub = _assemble(cells, labels_str, cohort.subclass_by_id)
    for attr in ("pia_y", "wm_y"):
        if hasattr(cohort, attr):
            setattr(sub, attr, getattr(cohort, attr)[keep])
    return sub


def subsample_per_class(cohort: Cohort, per_class: int, seed: int = 0) -> Cohort:
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for k in range(len(cohort.label_names)):
        idx = np.flatnonzero(cohort.labels == k)
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        keep.extend(sorted(idx.tolist()))
    return _subset(cohort, np.array(sorted(keep), dtype=int))


def subsample_proportional(
    cohort: Cohort, n_cells: int, seed: int = 0, min_class_size: int = 25, log=print,
) -> Cohort:
    """Class-proportional subsample.

    Small classes can fall below min_class_size and get dropped, so the class set at a
    learning-curve point is not guaranteed to match the full cohort's. That is recorded per
    run rather than corrected, since forcing every class to survive would change the prior.
    """
    if n_cells >= len(cohort):
        return cohort
    rng = np.random.default_rng(seed)
    frac = n_cells / len(cohort)
    keep: list[int] = []
    for k in range(len(cohort.label_names)):
        idx = np.flatnonzero(cohort.labels == k)
        take = int(round(len(idx) * frac))
        if take < min_class_size:
            log(f"subsample n={n_cells}: dropping {cohort.label_names[k]} "
                f"({take} < min_class_size {min_class_size})")
            continue
        keep.extend(rng.choice(idx, take, replace=False).tolist())
    sub = _subset(cohort, np.array(sorted(keep), dtype=int))
    log(f"subsampled to {len(sub)} cells, {len(sub.label_names)} classes {class_counts(sub)}")
    return sub


def soma_depth_features(cohort: Cohort):
    """Two-feature control: soma-to-pia and soma-to-wm distance along the cortical axis.

    The coarse labels are laminar, so this is the number every morphology representation must
    be read against -- not chance. Marked as a control, not a representation.
    """
    from .features import FeatureResult

    X = np.stack([np.asarray(cohort.pia_y, dtype=float),
                  np.asarray(cohort.wm_y, dtype=float)], axis=1)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return FeatureResult(
        specimen_ids=[c.specimen_id for c in cohort.cells],
        X=X, feature_names=["soma_to_pia_y", "soma_to_wm_y"], skipped=[],
    )
