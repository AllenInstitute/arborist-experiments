"""Port of /root/capsule/radial_arbors.py into the benchmark.

Two feature families per arbor group, concatenated into one vector:

  - radial density histogram -- a cylindrical (depth, in-plane radius) 2D histogram of
    resampled cable, each radial bin divided by its annulus area so it reads as a density
    rather than a count that grows with circumference.
  - Sholl profile -- 3D segment crossings of concentric soma-centered spheres.

Three deliberate departures from the original, all for comparability with the other
benchmark rows:

  - Both families run in `swc_upright`, not the original's split frames (conformal for the
    histogram, upright for Sholl). So the cortical axis is upright y in microns rather than
    conformal normalized depth -- curved cortex is not straightened and a bin means a
    soma-relative micron offset, not a laminar fraction. The in-plane radius is
    sqrt(x^2 + z^2); z is slice-normal and spans only ~200 um against x's ~550, so the
    radius is dominated by x. No NaN mask is needed (swc_upright has 0 non-finite nodes
    across 1618 cells, unlike the conformal frame the original guarded against).
  - Arbor groups are axon and dendrite (basal + apical), matching MODALITY_TYPES, and the
    modality argument selects which blocks are emitted. The original's four groups included
    dendrite alongside basal and apical, which is redundant -- histograms are additive, so
    the dendrite block was exactly basal + apical.
  - Nodes are resampled to unit spacing before binning, so the histogram measures cable
    length rather than tracer node placement (see _resample_cable).

Bin edges come from the p99.5 envelope of soma-centered swc_upright over all 1618 cells
(in-plane radius 318 um, y -310..556). Everything is soma-centered, so the features carry no
absolute position -- a group with no cable yields an all-zero block, which is a true zero
density and needs no imputation.
"""
from __future__ import annotations

import numpy as np

from .data_loading import Cell, Cohort, TYPE_SOMA, TYPE_AXON, TYPE_BASAL, TYPE_APICAL
from .features import FeatureResult, _run_jobs


STRUCTURE_GROUPINGS = {
    "axon":     [TYPE_AXON],
    "dendrite": [TYPE_BASAL, TYPE_APICAL],
}

MODALITY_GROUPS = {
    "full":     ["axon", "dendrite"],
    "axon":     ["axon"],
    "dendrite": ["dendrite"],
}

CORTICAL_AXIS = 1                                   # upright y (pia direction)
_INPLANE_AXES = [a for a in range(3) if a != CORTICAL_AXIS]

R_EDGES = np.linspace(0.0, 320.0, 13)               # 12 in-plane radial bins (microns)
Z_EDGES = np.linspace(-350.0, 600.0, 13)            # 12 soma-relative depth bins (microns)
SHOLL_MAX_RADIUS = 600.0
SHOLL_NUM_STEPS = 20
RESAMPLE_STEP = 1.0                                 # microns, matching morphopy resample_nodes(d=1)

_DISK_AREAS = np.pi * (R_EDGES[1:] ** 2 - R_EDGES[:-1] ** 2)
_DISK_AREAS = _DISK_AREAS / _DISK_AREAS.max()


def _real_edges(cell: Cell, struct_ids: list[int], include_soma: bool = False) -> np.ndarray:
    """Child indices of traced edges belonging to `struct_ids`.

    Excludes the links _reparent_to_soma invented: they are straight lines from the soma to
    each disconnected fragment, 32% of total edge length across the cohort, and no cable was
    ever traced along them. The fragments' own nodes and edges are real and are kept.
    """
    types = struct_ids + [TYPE_SOMA] if include_soma else struct_ids
    return np.flatnonzero(
        np.isin(cell.types, types) & (cell.parents != -1) & ~cell.synthetic_edge
    )


def _resample_cable(cell: Cell, struct_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Samples spaced ~RESAMPLE_STEP um along each edge, with the cable length each carries.

    Node density is a tracer artifact -- median edge length is 1.15 um but the spread runs to
    8.8 um -- and morphopy's density maps call resample_nodes(d=1) for the same reason. Samples
    sit at segment midpoints and are weighted by length/n rather than counted, so a stretch
    traced with extra nodes gets no extra weight: 0.7% of edges are shorter than half a step
    and would otherwise be over-weighted by the one-sample-per-edge floor.
    """
    child = _real_edges(cell, struct_ids)
    if len(child) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    p0 = cell.xyz[cell.parents[child]]
    p1 = cell.xyz[child]
    lengths = np.linalg.norm(p1 - p0, axis=1)
    n = np.maximum(1, np.rint(lengths / RESAMPLE_STEP).astype(np.int64))
    seg = np.repeat(np.arange(len(child)), n)
    within = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    t = (within + 0.5) / n[seg]
    return p0[seg] + (p1[seg] - p0[seg]) * t[:, None], lengths[seg] / n[seg]


def radial_histogram(cell: Cell, struct_ids: list[int]) -> np.ndarray:
    pts, weights = _resample_cable(cell, struct_ids)
    r = np.linalg.norm(pts[:, _INPLANE_AXES], axis=1)
    hist = np.histogram2d(pts[:, CORTICAL_AXIS], r, [Z_EDGES, R_EDGES], weights=weights)[0]
    return hist / _DISK_AREAS


def sholl_profile(cell: Cell, struct_ids: list[int]) -> np.ndarray:
    query_radii = np.linspace(0.0, SHOLL_MAX_RADIUS, SHOLL_NUM_STEPS)
    radii = np.linalg.norm(cell.xyz, axis=1)
    # The soma joins the group so that arbor-to-soma segments are counted.
    child = _real_edges(cell, struct_ids, include_soma=True)
    if len(child) == 0:
        return np.zeros(SHOLL_NUM_STEPS)
    seg = np.stack([radii[cell.parents[child]], radii[child]], axis=-1)
    is_larger = seg[..., None] > query_radii[None, None]
    return np.sum(is_larger[:, 0] ^ is_larger[:, 1], axis=0)


def feature_names(modality: str) -> list[str]:
    groups = MODALITY_GROUPS[modality]
    names = [
        f"hist_{g}_z{zi}_r{ri}"
        for g in groups
        for zi in range(len(Z_EDGES) - 1)
        for ri in range(len(R_EDGES) - 1)
    ]
    names += [f"sholl_{g}_{si}" for g in groups for si in range(SHOLL_NUM_STEPS)]
    return names


def arbor_feature_vector(cell: Cell, modality: str) -> np.ndarray:
    groups = [STRUCTURE_GROUPINGS[g] for g in MODALITY_GROUPS[modality]]
    blocks = [radial_histogram(cell, ids).ravel() for ids in groups]
    blocks += [sholl_profile(cell, ids).ravel() for ids in groups]
    return np.concatenate(blocks).astype(np.float64)


def _job_radial_arbors(cell: Cell, modality: str):
    struct_ids = [t for g in MODALITY_GROUPS[modality] for t in STRUCTURE_GROUPINGS[g]]
    if not np.isin(cell.types, struct_ids).any():
        return cell.specimen_id, None, None, "no nodes for modality"
    try:
        return cell.specimen_id, arbor_feature_vector(cell, modality), None, None
    except Exception as e:
        return cell.specimen_id, None, None, f"radial_arbors: {type(e).__name__}: {e}"


def extract_radial_arbors(cohort: Cohort, modality: str) -> FeatureResult:
    names = feature_names(modality)
    rows: list[tuple[str, np.ndarray]] = []
    skipped: list[tuple[str, str]] = []
    for sid, v, _names, err in _run_jobs(cohort, _job_radial_arbors, modality):
        if err is not None:
            skipped.append((sid, err)); continue
        rows.append((sid, v))
    X = np.vstack([r[1] for r in rows]) if rows else np.zeros((0, len(names)))
    return FeatureResult(specimen_ids=[r[0] for r in rows], X=X, feature_names=names, skipped=skipped)
