"""A labelled collection of cells, and the operations that only need one.

The dataclass is the contract every benchmark is written against. The functions below it are the
operations that read nothing but that contract -- counting a class, rebuilding a cohort from an
index array, drawing a subsample -- so they are independent of any one dataset.
"""

from dataclasses import dataclass, field

import numpy as np

from arborist.skeleton_graph import SkeletonGraph


@dataclass
class Cohort:
    cells: list[SkeletonGraph]
    labels: np.ndarray                 # (n_cells,) int label index
    label_names: list[str]             # class name per label index
    subclass_by_id: dict[str, str] = field(default_factory=dict)
    # Per-cell measurements that are not computable from a morphology: MICrONS carries
    # `pia_y`/`wm_y` from its upright frame, which the `soma_depth` control reads. Each value is
    # (n_cells,) and parallel to `cells`, which is the whole contract -- `subset_cohort` slices
    # every entry, so a new one is carried through a subsample without touching this module.
    per_cell: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self):
        return len(self.cells)


def class_counts(cohort: Cohort) -> dict[str, int]:
    counts = {name: 0 for name in cohort.label_names}
    for k in cohort.labels:
        counts[cohort.label_names[int(k)]] += 1
    return counts


def assemble_cohort(cells: list[SkeletonGraph], labels_str: list[str],
                    label_by_id: dict[str, str]) -> Cohort:
    label_names = sorted(set(labels_str))
    index = {name: i for i, name in enumerate(label_names)}
    labels = np.array([index[s] for s in labels_str], dtype=np.int64)
    return Cohort(cells=cells, labels=labels, label_names=label_names,
                  subclass_by_id=label_by_id)


def subset_cohort(cohort: Cohort, keep: np.ndarray) -> Cohort:
    cells = [cohort.cells[i] for i in keep]
    labels_str = [cohort.label_names[int(cohort.labels[i])] for i in keep]
    sub = assemble_cohort(cells, labels_str, cohort.subclass_by_id)
    # Dropping these would silently change what the soma_depth control measures rather than fail.
    sub.per_cell = {k: v[keep] for k, v in cohort.per_cell.items()}
    return sub


def subsample_per_class(cohort: Cohort, per_class: int, seed: int = 0) -> Cohort:
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for k in range(len(cohort.label_names)):
        idx = np.flatnonzero(cohort.labels == k)
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        keep.extend(sorted(idx.tolist()))
    return subset_cohort(cohort, np.array(sorted(keep), dtype=int))


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
    sub = subset_cohort(cohort, np.array(sorted(keep), dtype=int))
    log(f"subsampled to {len(sub)} cells, {len(sub.label_names)} classes {class_counts(sub)}")
    return sub
