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
    sqrt(x^2 + z^2). On Patch-seq z is slice-normal and spans only ~200 um against x's ~550, so
    the radius is dominated by x; on MICrONS the two are comparable (~320 vs ~280 um) because
    that is a volume rather than a slice. No NaN mask is needed on either -- swc_upright has 0
    non-finite nodes in both cohorts, unlike the conformal frame the original guarded against.
  - Arbor groups match MODALITY_TYPES and the modality argument selects which blocks are
    emitted. The original's four groups included dendrite alongside basal and apical, which is
    redundant -- histograms are additive, so the dendrite block was exactly basal + apical.
  - Nodes are resampled to unit spacing before binning, so the histogram measures cable
    length rather than tracer node placement (see _resample_cable).

Bin edges arrive in the `BinningConfig`; the three cohorts' envelopes differ by more than a
factor of two in radius. Everything is soma-centered, so the features carry no absolute position
-- a group with no cable yields an all-zero block, which is a true zero density and needs no
imputation.
"""
from __future__ import annotations

import numpy as np

from arborist_experiments.utils.cells import (
    CellGraph, Cohort, TYPE_APICAL, TYPE_AXON, TYPE_BASAL, TYPE_SOMA, run_jobs,
)
from .binning import BinningConfig
from .features import FeatureResult


STRUCTURE_GROUPINGS = {
    "axon":     [TYPE_AXON],
    "dendrite": [TYPE_BASAL, TYPE_APICAL],
    "basal":    [TYPE_BASAL],
    "apical":   [TYPE_APICAL],
}

# `dendrite` stays ONE summed basal+apical block rather than two separate blocks, so the
# Patch-seq and MICrONS layouts match and the headline rows stay comparable. Splitting it would
# look like an improvement and would quietly invalidate the comparison; get the split via
# --modalities basal apical instead.
#
# `full` is two blocks in axon-then-dendrite order for the same reason. On the MICrONS excitatory
# cohort it is unreachable -- `microns.modality.check_modality` rejects it there, where 0.43%
# axon prevalence would make it a duplicate of `dendrite`.
MODALITY_GROUPS = {
    "full":     ["axon", "dendrite"],
    "axon":     ["axon"],
    "dendrite": ["dendrite"],
    "basal":    ["basal"],
    "apical":   ["apical"],
}

# Confirmed on MICrONS two independent ways: pia_upright y > 0 > wm_upright y for 100% of cells,
# and mean apical y > mean basal y for 97% of apical-bearing cells. Soma depth also orders the
# laminar labels monotonically (L2IT 131 um -> L6CT 606 um from pia).
CORTICAL_AXIS = 1                                   # upright y (pia direction)
_INPLANE_AXES = [a for a in range(3) if a != CORTICAL_AXIS]

# Cohort-invariant, and outside the set feature_cache.constants_digest hashes -- see binning.py.
SHOLL_MAX_RADIUS = 600.0
SHOLL_NUM_STEPS = 20
RESAMPLE_STEP = 1.0                                 # microns, matching morphopy resample_nodes(d=1)


def _real_edges(cell: CellGraph, struct_ids: list[int],
                include_soma: bool = False) -> np.ndarray:
    """Child indices of traced edges belonging to `struct_ids`.

    Excludes the links _reparent_to_soma invented: they are straight lines from the soma to
    each disconnected fragment, 32% of total Patch-seq edge length, and no cable was ever traced
    along them. The fragments' own nodes and edges are real and are kept. A no-op on MICrONS,
    where every cell arrives as one connected tree with exactly one root and synthetic_edge is
    all-False -- kept unconditional so the two cohorts run the same code.
    """
    types = struct_ids + [TYPE_SOMA] if include_soma else struct_ids
    return np.flatnonzero(
        np.isin(cell.types, types) & (cell.parents != -1) & ~cell.synthetic_edge
    )


def _resample_cable(cell: CellGraph, struct_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Samples spaced ~RESAMPLE_STEP um along each edge, with the cable length each carries.

    On Patch-seq node density is a tracer artifact -- median edge length is 1.15 um but the
    spread runs to 8.8 um -- and morphopy's density maps call resample_nodes(d=1) for the same
    reason. MICrONS skeletons are already near-uniform (median 1.506 um, p99 1.511) so it matters
    far less there, but dropping it would make the row incomparable. Samples sit at segment
    midpoints and are weighted by length/n rather than counted, so a stretch traced with extra
    nodes gets no extra weight: the 0.7% (Patch-seq) / 1.0% (MICrONS) of edges shorter than half
    a step would otherwise be over-weighted by the one-sample-per-edge floor.
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


def radial_histogram(cell: CellGraph, struct_ids: list[int], cfg: BinningConfig) -> np.ndarray:
    pts, weights = _resample_cable(cell, struct_ids)
    r = np.linalg.norm(pts[:, _INPLANE_AXES], axis=1)
    hist = np.histogram2d(pts[:, CORTICAL_AXIS], r, [cfg.z_edges, cfg.r_edges],
                          weights=weights)[0]
    return hist / cfg.disk_areas


def sholl_profile(cell: CellGraph, struct_ids: list[int]) -> np.ndarray:
    query_radii = np.linspace(0.0, SHOLL_MAX_RADIUS, SHOLL_NUM_STEPS)
    radii = np.linalg.norm(cell.xyz, axis=1)
    # The soma joins the group so that arbor-to-soma segments are counted.
    child = _real_edges(cell, struct_ids, include_soma=True)
    if len(child) == 0:
        return np.zeros(SHOLL_NUM_STEPS)
    seg = np.stack([radii[cell.parents[child]], radii[child]], axis=-1)
    is_larger = seg[..., None] > query_radii[None, None]
    return np.sum(is_larger[:, 0] ^ is_larger[:, 1], axis=0)


def feature_names(modality: str, cfg: BinningConfig) -> list[str]:
    groups = MODALITY_GROUPS[modality]
    names = [
        f"hist_{g}_z{zi}_r{ri}"
        for g in groups
        for zi in range(len(cfg.z_edges) - 1)
        for ri in range(len(cfg.r_edges) - 1)
    ]
    names += [f"sholl_{g}_{si}" for g in groups for si in range(SHOLL_NUM_STEPS)]
    return names


def arbor_feature_vector(cell: CellGraph, modality: str, cfg: BinningConfig) -> np.ndarray:
    groups = [STRUCTURE_GROUPINGS[g] for g in MODALITY_GROUPS[modality]]
    blocks = [radial_histogram(cell, ids, cfg).ravel() for ids in groups]
    blocks += [sholl_profile(cell, ids).ravel() for ids in groups]
    return np.concatenate(blocks).astype(np.float64)


def _job_radial_arbors(cell: CellGraph, modality: str, cfg: BinningConfig):
    struct_ids = [t for g in MODALITY_GROUPS[modality] for t in STRUCTURE_GROUPINGS[g]]
    if not np.isin(cell.types, struct_ids).any():
        return cell.specimen_id, None, None, "no nodes for modality"
    try:
        return cell.specimen_id, arbor_feature_vector(cell, modality, cfg), None, None
    except Exception as e:
        return cell.specimen_id, None, None, f"radial_arbors: {type(e).__name__}: {e}"


def extract_radial_arbors(cohort: Cohort, modality: str, cfg: BinningConfig) -> FeatureResult:
    names = feature_names(modality, cfg)
    rows: list[tuple[str, np.ndarray]] = []
    skipped: list[tuple[str, str]] = []
    for sid, v, _names, err in run_jobs(cohort, _job_radial_arbors, modality, cfg):
        if err is not None:
            skipped.append((sid, err)); continue
        rows.append((sid, v))
    X = np.vstack([r[1] for r in rows]) if rows else np.zeros((0, len(names)))
    return FeatureResult(specimen_ids=[r[0] for r in rows], X=X, feature_names=names, skipped=skipped)
