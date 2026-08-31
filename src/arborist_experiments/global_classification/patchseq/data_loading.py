
import h5py
import numpy as np

from arborist_experiments.utils.cells import (
    TYPE_APICAL,
    TYPE_AXON,
    TYPE_BASAL,
    TYPE_SOMA,
    CellGraph,
    Cohort,
)

# Re-exported so the 8 modules that import `Cell`, `Cohort` and the TYPE_*
# constants from here keep working. `Cell` is the old dataclass's name for what
# is now a SkeletonGraph subclass; the six field accesses it exposed are
# unchanged.
Cell = CellGraph

__all__ = [
    "TYPE_APICAL",
    "TYPE_AXON",
    "TYPE_BASAL",
    "TYPE_SOMA",
    "Cell",
    "CellGraph",
    "Cohort",
    "build_cohort",
    "extract_precomputed_morphometrics",
    "load_precomputed_morphometrics",
    "load_subclass_labels",
]


def load_subclass_labels(met_path: str) -> dict[str, str]:
    with h5py.File(met_path, "r") as f:
        ids = [s.decode() for s in f["meta/specimens"][:]]
        sub = [s.decode() for s in f["meta/subclass"][:]]
    return {i: s for i, s in zip(ids, sub) if s != "nan"}


def load_precomputed_morphometrics(met_path: str) -> tuple[list[str], np.ndarray, list[str]]:
    with h5py.File(met_path, "r") as f:
        ids = [s.decode() for s in f["morphometric/specimens"][:]]
        X = f["morphometric/data"][:]
        feats = [s.decode() for s in f["morphometric/features"][:]]
    return ids, X, feats


def extract_precomputed_morphometrics(cohort: Cohort, met_path: str):
    """The MET table's own curated morphometrics, as a FeatureResult.

    Lives here rather than with the shared extractors because it reads a Patch-seq table instead
    of a reconstruction: there is no MICrONS counterpart, and since it is the strongest Patch-seq
    row (0.830) the top of the MICrONS ranking is necessarily a different representation -- a
    property of the datasets, not a result. `microns.data_loading.soma_depth_features` is the
    same shape of thing on the other cohort.
    """
    from arborist_experiments.global_classification.extract import FeatureResult

    ids, X_all, feats = load_precomputed_morphometrics(met_path)
    id_to_row = {i: k for k, i in enumerate(ids)}
    cohort_ids = [c.specimen_id for c in cohort.cells]

    rows: list[tuple[str, np.ndarray]] = []
    skipped: list[tuple[str, str]] = []
    for sid in cohort_ids:
        k = id_to_row.get(sid)
        if k is None:
            skipped.append((sid, "precomputed_morphometrics: not in MET table"))
            continue
        rows.append((sid, X_all[k]))
    if not rows:
        return FeatureResult(specimen_ids=[], X=np.zeros((0, len(feats))),
                             feature_names=feats, skipped=skipped)
    X = np.vstack([r[1] for r in rows])
    col_ok = ~np.all(np.isnan(X), axis=0)
    X = X[:, col_ok]
    kept_feats = [n for n, ok in zip(feats, col_ok) if ok]
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return FeatureResult(
        specimen_ids=[r[0] for r in rows], X=X, feature_names=kept_feats, skipped=skipped,
    )


def _per_cell(f: h5py.File, group: str, i: int) -> np.ndarray:
    shape = tuple(int(s) for s in f[group]["shape"][i])
    return f[group]["data"][i].reshape(shape)


def _reparent_to_soma(types: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reparent every orphan (parent == -1) that is not the soma to the soma node.

    Patch-seq SWC HDF5 stores many cells as several disconnected subtrees rooted at
    parent = -1. morphopy assumes a single connected tree; without this the
    topological-minor traversal walks off the graph.

    Returns (parents, synthetic) where synthetic marks the invented links. They are not
    traced cable and are long: 42% of cells are fragmented, median 74 fragments each,
    median soma-to-fragment distance 96 um against a real edge length of 1.15 um. Any
    length-weighted feature must exclude them or they contribute ~32% of total edge
    length as straight lines nobody traced.
    """
    synthetic = np.zeros(len(parents), dtype=bool)
    soma_idx = np.flatnonzero(types == 1)
    if soma_idx.size == 0:
        return parents, synthetic
    soma = int(soma_idx[0])
    new_parents = parents.copy()
    orphan_mask = (new_parents == -1)
    orphan_mask[soma] = False
    new_parents[orphan_mask] = soma
    return new_parents, orphan_mask


def _soma_origin(xyz: np.ndarray, types: np.ndarray) -> np.ndarray:
    soma = xyz[types == TYPE_SOMA]
    return soma.mean(axis=0) if len(soma) else xyz.mean(axis=0)


def _iter_swc(swc_path: str):
    """Yield soma-centered cells from the SWC HDF5.

    Two things this file does not contain, both learned the hard way:
      - There is no radius anywhere. Every swc_* group is (N, 3) coordinates; `swc_conf` is the
        conformal frame (cols 0-1 microns, col 2 normalized pia->wm depth), NOT radius. Radius
        is therefore constant, which makes morphopy's thickness/surface/volume statistics
        radius-free rather than silently depth-derived.
      - `swc_upright` is soma-centered in x and y but NOT in z, where the soma sits at the
        absolute slice coordinate (4-181 um). Centering z too makes all three axes soma-relative,
        so no feature can key off position within the slice.
    """
    with h5py.File(swc_path, "r") as f:
        specimens = [s.decode() for s in f["swc_final/specimens"][:]]
        for i, sid in enumerate(specimens):
            types = _per_cell(f, "swc_types", i).astype(np.int32)
            parents = _per_cell(f, "swc_parents", i).astype(np.int64)
            xyz = _per_cell(f, "swc_upright", i).astype(np.float64)
            xyz = xyz - _soma_origin(xyz, types)
            radius = np.ones(len(types), dtype=np.float64)
            parents, synthetic = _reparent_to_soma(types, parents)
            yield CellGraph.from_arrays(
                specimen_id=sid, types=types, parents=parents, xyz=xyz,
                radius=radius, synthetic_edge=synthetic)


def build_cohort(
    met_path: str,
    swc_path: str,
    min_class_size: int = 25,
    allowed_ids: set[str] | None = None,
) -> Cohort:
    subclass_by_id = load_subclass_labels(met_path)

    labelled: list[Cell] = []
    subclass_by_cell: list[str] = []
    for cell in _iter_swc(swc_path):
        if allowed_ids is not None and cell.specimen_id not in allowed_ids:
            continue
        sc = subclass_by_id.get(cell.specimen_id)
        if sc is None:
            continue
        labelled.append(cell)
        subclass_by_cell.append(sc)

    counts: dict[str, int] = {}
    for sc in subclass_by_cell:
        counts[sc] = counts.get(sc, 0) + 1
    surviving = sorted([s for s, n in counts.items() if n >= min_class_size])
    keep = set(surviving)

    cells: list[Cell] = []
    labels_str: list[str] = []
    for cell, sc in zip(labelled, subclass_by_cell):
        if sc in keep:
            cells.append(cell)
            labels_str.append(sc)

    label_names = surviving
    label_index = {name: i for i, name in enumerate(label_names)}
    labels = np.array([label_index[s] for s in labels_str], dtype=np.int64)

    return Cohort(cells=cells, labels=labels, label_names=label_names, subclass_by_id=subclass_by_id)
