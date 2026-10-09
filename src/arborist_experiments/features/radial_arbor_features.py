"""Radial arbor features: two families per arbor group, extracted separately.

  - `radial_hist` -- a cylindrical (depth, in-plane radius) 2D histogram of resampled cable,
    each radial bin divided by its annulus area so it reads as a density rather than a count
    that grows with circumference.
  - `radial_sholl` -- 3D segment crossings of concentric soma-centered spheres.

`features.concat_features` joins the two into one vector for a caller who wants them together.

Three design choices:

  - Both families run in the upright frame: the cortical axis is upright y in microns, so a depth bin
    is a soma-relative micron offset rather than a laminar fraction, and curved cortex is not
    straightened. The in-plane radius is sqrt(x^2 + z^2). In a slice preparation z is slice-normal and
    spans much less than x, so the radius is dominated by x; in a volume the two are comparable.
  - Arbor groups match MODALITY_TYPES and the modality argument selects which blocks are emitted.
  - Nodes are resampled to unit spacing before binning, so the histogram measures cable length
    rather than tracer node placement (see _resample_cable).

Bin edges are the only dataset-dependent thing here and they arrive as two arrays, `r_edges` and
`z_edges`. They reach the histogram only -- the Sholl profile's radii are the module constants below,
so `extract_radial_sholl` takes no dataset constant at all. Everything is soma-centered, so the
features carry no absolute position -- a group with no cable yields an all-zero block, which is a true
zero density and needs no imputation.
"""
from __future__ import annotations

import numpy as np

from arborist.skeleton_graph import SkeletonGraph

from arborist_experiments.utils.cells import (
    TYPE_APICAL,
    TYPE_AXON,
    TYPE_BASAL,
    TYPE_SOMA,
)

from .mapping import map_graphs
from .result import FeatureResult

STRUCTURE_GROUPINGS = {
    "axon":     [TYPE_AXON],
    "dendrite": [TYPE_BASAL, TYPE_APICAL],
    "basal":    [TYPE_BASAL],
    "apical":   [TYPE_APICAL],
}

# `dendrite` is ONE summed basal+apical block rather than two, so its layout is the same whether or
# not a dataset distinguishes basal from apical; request the `basal` and `apical` modalities for the
# split. `full` is two blocks in axon-then-dendrite order for the same reason.
MODALITY_GROUPS = {
    "full":     ["axon", "dendrite"],
    "axon":     ["axon"],
    "dendrite": ["dendrite"],
    "basal":    ["basal"],
    "apical":   ["apical"],
}

CORTICAL_AXIS = 1                                   # upright y (pia direction)
_INPLANE_AXES = [a for a in range(3) if a != CORTICAL_AXIS]

# What this module asks of a graph, checked in the parent before it is pickled. `type` selects the
# arbor groups; `soma_at_origin` because both families read node_xyz as a soma-centered displacement
# without re-centering it -- `sholl_profile` takes norm(node_xyz) directly and the histogram bins
# node_xyz[CORTICAL_AXIS] against soma-relative edges. Node radius is deliberately absent: `radii`
# below is distance from the origin, and nothing here reads node_feats["radius"].
REQ = {"keys": ("type",), "soma_at_origin": True}

# Fixed for every dataset, so not part of `BinningConfig`.
SHOLL_MAX_RADIUS = 600.0
SHOLL_NUM_STEPS = 20
RESAMPLE_STEP = 1.0                                 # microns, matching morphopy resample_nodes(d=1)


def _real_edges(cell: SkeletonGraph, struct_ids: list[int],
                include_soma: bool = False) -> np.ndarray:
    """Child indices of traced edges belonging to `struct_ids`.

    Excludes links marked `synthetic_edge` (e.g. invented by `reparent_to_soma`): they are straight
    lines from the soma to a disconnected fragment, and no cable was ever traced along them. The
    fragments' own nodes and edges are real and are kept. A no-op on a graph that is one connected
    tree with no synthetic edges.
    """
    types = struct_ids + [TYPE_SOMA] if include_soma else struct_ids
    feats = cell.node_feats
    return np.flatnonzero(
        np.isin(feats["type"], types) & (feats["parent"] != -1) & ~feats["synthetic_edge"]
    )


def _resample_cable(cell: SkeletonGraph, struct_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Samples spaced ~RESAMPLE_STEP um along each edge, with the cable length each carries.

    Node density in a manual tracing is an artifact of the tracer -- edge lengths can vary several-
    fold within a cell -- and morphopy's density maps call resample_nodes(d=1) for the same reason.
    Samples sit at segment midpoints and are weighted by length/n rather than counted, so a stretch
    traced with extra nodes gets no extra weight, and edges shorter than half a step are not
    over-weighted by the one-sample-per-edge floor.
    """
    child = _real_edges(cell, struct_ids)
    if len(child) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    xyz, parents = cell.node_xyz, cell.node_feats["parent"]
    p0 = xyz[parents[child]]
    p1 = xyz[child]
    lengths = np.linalg.norm(p1 - p0, axis=1)
    n = np.maximum(1, np.rint(lengths / RESAMPLE_STEP).astype(np.int64))
    seg = np.repeat(np.arange(len(child)), n)
    within = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    t = (within + 0.5) / n[seg]
    return p0[seg] + (p1[seg] - p0[seg]) * t[:, None], lengths[seg] / n[seg]


def disk_areas(r_edges: np.ndarray) -> np.ndarray:
    """Annulus areas for `r_edges`, normalized by the largest.

    Derived rather than passed alongside `r_edges`, because it is not an independent parameter: a
    caller who supplied both could pair one set of edges with another's discs, and normalizing a
    histogram by the wrong annuli is silent.
    """
    areas = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    return areas / areas.max()


def radial_histogram(cell: SkeletonGraph, struct_ids: list[int], *,
                     r_edges: np.ndarray, z_edges: np.ndarray) -> np.ndarray:
    pts, weights = _resample_cable(cell, struct_ids)
    r = np.linalg.norm(pts[:, _INPLANE_AXES], axis=1)
    hist = np.histogram2d(pts[:, CORTICAL_AXIS], r, [z_edges, r_edges], weights=weights)[0]
    return hist / disk_areas(r_edges)


def sholl_profile(cell: SkeletonGraph, struct_ids: list[int]) -> np.ndarray:
    query_radii = np.linspace(0.0, SHOLL_MAX_RADIUS, SHOLL_NUM_STEPS)
    radii = np.linalg.norm(cell.node_xyz, axis=1)
    # The soma joins the group so that arbor-to-soma segments are counted.
    child = _real_edges(cell, struct_ids, include_soma=True)
    if len(child) == 0:
        return np.zeros(SHOLL_NUM_STEPS)
    seg = np.stack([radii[cell.node_feats["parent"][child]], radii[child]], axis=-1)
    is_larger = seg[..., None] > query_radii[None, None]
    return np.sum(is_larger[:, 0] ^ is_larger[:, 1], axis=0)


def radial_hist_names(modality: str, *, r_edges: np.ndarray, z_edges: np.ndarray) -> list[str]:
    return [
        f"hist_{g}_z{zi}_r{ri}"
        for g in MODALITY_GROUPS[modality]
        for zi in range(len(z_edges) - 1)
        for ri in range(len(r_edges) - 1)
    ]


def radial_sholl_names(modality: str) -> list[str]:
    return [f"sholl_{g}_{si}" for g in MODALITY_GROUPS[modality] for si in range(SHOLL_NUM_STEPS)]


def radial_hist_vector(cell: SkeletonGraph, modality: str, *,
                       r_edges: np.ndarray, z_edges: np.ndarray) -> np.ndarray:
    return np.concatenate([
        radial_histogram(cell, STRUCTURE_GROUPINGS[g], r_edges=r_edges, z_edges=z_edges).ravel()
        for g in MODALITY_GROUPS[modality]
    ]).astype(np.float64)


def radial_sholl_vector(cell: SkeletonGraph, modality: str) -> np.ndarray:
    return np.concatenate([
        sholl_profile(cell, STRUCTURE_GROUPINGS[g]).ravel()
        for g in MODALITY_GROUPS[modality]
    ]).astype(np.float64)


# The one skip both families report, and they report it identically: the groups a modality selects are
# the same groups, so a graph that has no cable for one has none for the other. That keeps the two
# batches aligned under `concat_features`, with no row present in one and missing from the other.
NO_NODES = "no nodes for modality"


def _no_modality_nodes(cell: SkeletonGraph, modality: str) -> bool:
    struct_ids = [t for g in MODALITY_GROUPS[modality] for t in STRUCTURE_GROUPINGS[g]]
    return not np.isin(cell.node_feats["type"], struct_ids).any()


def _job_radial_hist(cell: SkeletonGraph, modality: str,
                     r_edges: np.ndarray, z_edges: np.ndarray):
    """No `except Exception` here on purpose, unlike the morphopy jobs.

    `check` rejects unusable graphs in the parent, so the only thing an exception here could mean is a
    defect, and swallowing it would turn a defect into a silently skipped cell. `run_jobs` lets it
    propagate instead. The empty-modality return below is a real per-graph skip, not a failure.

    Takes the two edge arrays positionally, unlike the public functions: `run_jobs` forwards
    `map_graphs`' `*job_args` through `ex.map`, so a job cannot have keyword-only parameters.
    """
    if _no_modality_nodes(cell, modality):
        return None, None, NO_NODES
    return radial_hist_vector(cell, modality, r_edges=r_edges, z_edges=z_edges), None, None


def _job_radial_sholl(cell: SkeletonGraph, modality: str):
    """Unnetted for the same reason as `_job_radial_hist`."""
    if _no_modality_nodes(cell, modality):
        return None, None, NO_NODES
    return radial_sholl_vector(cell, modality), None, None


def extract_radial_hist(
    graphs: list[SkeletonGraph], modality: str, *,
    r_edges: np.ndarray, z_edges: np.ndarray,
) -> list[FeatureResult]:
    """One `FeatureResult` per input graph, in order.

    `r_edges` and `z_edges` are what this feature actually depends on -- 12 in-plane radial bins and
    12 soma-relative depth bins in microns, say -- rather than a whole `BinningConfig`, four
    of whose six fields reach nothing here. Keyword-only because two float arrays of the same shape
    passed the wrong way round would bin depth against radius without raising.

    `names` comes from the edges rather than from a successful cell, so every element carries the
    width -- including the skipped ones, and including a batch in which nothing succeeded.
    """
    names = radial_hist_names(modality, r_edges=r_edges, z_edges=z_edges)
    return [
        FeatureResult(X=v, feature_names=names, skipped=err)
        for v, _local_names, err in map_graphs(
            graphs, REQ, _job_radial_hist, modality, r_edges, z_edges
        )
    ]


def extract_radial_sholl(
    graphs: list[SkeletonGraph], modality: str,
) -> list[FeatureResult]:
    """One `FeatureResult` per input graph, in order.

    No dataset constant, and that is a fact about the feature rather than an omission: the query radii
    are `SHOLL_NUM_STEPS` steps to `SHOLL_MAX_RADIUS`, module constants. `extract_sholl` is a different
    feature -- morphopy's 2D `xy` projection over caller-supplied `sholl_radii`.
    """
    names = radial_sholl_names(modality)
    return [
        FeatureResult(X=v, feature_names=names, skipped=err)
        for v, _local_names, err in map_graphs(graphs, REQ, _job_radial_sholl, modality)
    ]
