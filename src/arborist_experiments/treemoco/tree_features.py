"""TreeMoCo tree preprocessing -- Chen et al. 2022, section 3.1.

Collapses a reconstruction to its topological minor (root, branch points, tips) and attaches
the geometry of each dropped degree-1 chain to the chain's downstream node as branch features.
The result is a small tree -- median 163 nodes against a median 5955 raw nodes on this cohort --
which is what makes a Tree-LSTM over whole neurons tractable.

Node feature layout, 30 columns:

    0:3    x, y, z         soma-centered swc_upright coordinates
    3      L               branch arc length (um)
    4      C = D / L       contraction, 1.0 for a straight branch
    5:29   F_E             8 edge vectors of the branch resampled to 9 points, in a canonical
                           per-branch frame
    29     is_synthetic    1.0 if this node's parent link was invented by _reparent_to_soma

Two departures from the paper, both forced by this cohort:

  - `is_synthetic` is a 30th feature the paper does not have. 42% of Patch-seq cells are stored as
    disconnected subtrees and `_reparent_to_soma` invents soma-to-fragment links that are ~32% of
    total edge length and were never traced. `radial_arbor_features` simply excludes them, but a
    Tree-LSTM needs the tree connected, so instead the link is kept as topology while its geometry
    is zeroed (L = C = 0, F_E = 0) and flagged. The root gets the same treatment: it has no parent
    branch.
  - The paper says each branch is resampled to "8 vertices" and that F_E is in R^24. 8 vertices give
    7 edge vectors, i.e. 21 dims. 24 = 8 x 3 requires 8 edge vectors, so F_E is built from 9 points
    here.
  - Branches are *stored* at 33 points and F_E takes every 4th, because L and C must not be measured
    on the 9-point polyline. Across 11019 real branches from this cohort a 9-point resample loses a
    median 2.2% of arc length and >5% on 21% of branches (worst case 33%), which would inflate
    C = D / L exactly on the tortuous branches that distinguish interneurons. At 33 points the median
    loss is 0.5%.

The canonical frame follows the paper's "start and end points are on the x-axis and the principle
direction is aligned with y-axis", which makes F_E invariant to global rotation. The sign of the y
and z axes is fixed independently, so F_E is also invariant to mirroring -- consistent with mirroring
being one of the augmentations, and it means the flip augmentation acts on the coordinate block only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arborist_experiments.utils.cells import CellGraph


N_POLYLINE = 33                     # stored points per branch; L and C are measured on these
N_RESAMPLE = 9                      # points F_E is built from; 8 edge vectors -> F_E in R^24
FE_STRIDE = (N_POLYLINE - 1) // (N_RESAMPLE - 1)
N_EDGES = N_RESAMPLE - 1
N_FEATURES = 3 + 1 + 1 + 3 * N_EDGES + 1

# Soma / axon / dendrite. Basal (3) and apical (4) dendrite are deliberately merged: an annotator
# marks a dendrite apical only after judging the cell pyramidal, so that split partly encodes the
# subclass label rather than the morphology -- 189 of 400 reconstructed cells have no type-4 node at
# all. Axon-vs-dendrite is a direct observation and is what the other representations get from being
# extracted per modality.
TYPE_GROUPS = ((1,), (2,), (3, 4))
TYPE_GROUP_NAMES = ("soma", "axon", "dendrite")
N_TYPE = len(TYPE_GROUPS)
N_FEATURES_TYPED = N_FEATURES + N_TYPE

COL_XYZ = slice(0, 3)
COL_L = 3
COL_C = 4
COL_FE = slice(5, 5 + 3 * N_EDGES)
COL_SYNTHETIC = 5 + 3 * N_EDGES
COL_TYPE = slice(N_FEATURES, N_FEATURES_TYPED)


@dataclass
class CollapsedTree:
    """A reconstruction reduced to its topological minor.

    Attributes
    ----------
    specimen_id : str
    parents : np.ndarray
        (M,) int64 -- collapsed parent index, -1 at the root.
    types : np.ndarray
        (M,) int32 -- SWC compartment type, carried over from the surviving nodes.
    xyz : np.ndarray
        (M, 3) float64 -- surviving node coordinates.
    polylines : np.ndarray
        (M, N_POLYLINE, 3) float64 -- each node's parent branch, arc-length resampled, ordered
        parent -> node. All zeros where `no_branch` is True.
    is_synthetic : np.ndarray
        (M,) bool -- the parent link crosses an edge invented by _reparent_to_soma.
    no_branch : np.ndarray
        (M,) bool -- the node has no usable parent branch: the root, or a synthetic link.
    """

    specimen_id: str
    parents: np.ndarray
    types: np.ndarray
    xyz: np.ndarray
    polylines: np.ndarray
    is_synthetic: np.ndarray
    no_branch: np.ndarray

    def __len__(self):
        return len(self.parents)


def resample_fixed(xyz: np.ndarray, n_points: int) -> np.ndarray:
    """`n_points` arc-length-equispaced samples along a polyline, endpoints included.

    A fixed *count* rather than the fixed *step* of `path_encodings.resample_path`: F_E has a fixed
    width, so every branch must yield the same number of samples regardless of its length. Repeated
    coordinates are dropped first because np.interp needs strictly increasing arc length.
    """
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    pts = xyz[np.concatenate([[True], seg > 0])]
    if len(pts) < 2:
        return np.repeat(xyz[:1], n_points, axis=0)
    arc = np.concatenate([[0.0], np.cumsum(seg[seg > 0])])
    t = np.linspace(0.0, arc[-1], n_points)
    return np.stack([np.interp(t, arc, pts[:, k]) for k in range(3)], axis=1)


def collapse(cell: CellGraph) -> CollapsedTree:
    """Reduce `cell` to root, branch points and tips, keeping each dropped chain as a polyline.

    Degree-1 nodes are exactly the nodes with one child that are not the root. Working from the
    child counts of `cell.parents` is O(N) and keeps the parent -> child direction, unlike
    `path_encodings.extract_paths`, which builds an undirected graph because the CurveEncoder's
    unit is an unoriented path.

    `is_synthetic` is OR-ed along the collapsed chain, the same propagation `modality.filter_cell`
    applies when it rewires across dropped nodes.
    """
    parents = cell.parents
    n = len(parents)
    n_children = np.bincount(parents[parents >= 0], minlength=n)
    keep = (parents == -1) | (n_children != 1)
    kept = np.flatnonzero(keep)

    new_index = np.full(n, -1, dtype=np.int64)
    new_index[kept] = np.arange(len(kept))

    m = len(kept)
    new_parents = np.full(m, -1, dtype=np.int64)
    is_synthetic = np.zeros(m, dtype=bool)
    polylines = np.zeros((m, N_POLYLINE, 3), dtype=np.float64)
    no_branch = np.zeros(m, dtype=bool)

    for i, node in enumerate(kept):
        chain = [int(node)]
        syn = bool(cell.synthetic_edge[node])
        p = int(parents[node])
        while p != -1 and not keep[p]:
            chain.append(p)
            syn = syn or bool(cell.synthetic_edge[p])
            p = int(parents[p])

        is_synthetic[i] = syn
        if p == -1:
            no_branch[i] = True                     # root: no parent branch
            continue
        new_parents[i] = new_index[p]
        if syn:
            no_branch[i] = True                     # invented cable, geometry is meaningless
            continue
        chain.append(p)
        polylines[i] = resample_fixed(cell.xyz[chain[::-1]], N_POLYLINE)

    return CollapsedTree(
        specimen_id=cell.specimen_id,
        parents=new_parents,
        types=cell.types[kept].astype(np.int32),
        xyz=cell.xyz[kept].copy(),
        polylines=polylines,
        is_synthetic=is_synthetic,
        no_branch=no_branch,
    )


def _canonical_frames(polylines: np.ndarray) -> np.ndarray:
    """Per-branch rotation matrices, shape (M, 3, 3), rows = (x, y, z) axes of the canonical frame.

    x runs start -> end, so the parent -> child direction survives. y is the in-plane principal
    direction of the residual after removing x, obtained in closed form from the 2x2 residual
    covariance rather than a 3x3 eigendecomposition per branch. y and z signs are fixed from the
    largest-magnitude coordinate, which also makes the frame mirror-invariant.
    """
    m = len(polylines)
    rel = polylines - polylines[:, :1]
    v = rel[:, -1]
    v_norm = np.linalg.norm(v, axis=1)

    # A branch whose endpoints coincide has no start->end direction; fall back to the direction of
    # the point furthest from the start, which is well defined unless the branch is a single point.
    far = rel[np.arange(m), np.argmax(np.linalg.norm(rel, axis=2), axis=1)]
    axis_x = np.where(v_norm[:, None] > 1e-9, v, far)
    x_norm = np.linalg.norm(axis_x, axis=1)
    degenerate = x_norm < 1e-9
    axis_x = np.where(degenerate[:, None], np.array([1.0, 0.0, 0.0]), axis_x)
    axis_x = axis_x / np.where(degenerate, 1.0, x_norm)[:, None]

    # Any orthonormal pair spanning the plane perpendicular to x. Choosing the world axis least
    # aligned with x keeps the cross products away from zero.
    world = np.eye(3)[np.argmin(np.abs(axis_x), axis=1)]
    e1 = np.cross(axis_x, world)
    e1 = e1 / np.clip(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12, None)
    e2 = np.cross(axis_x, e1)

    p = np.einsum("mpk,mk->mp", rel, e1)
    q = np.einsum("mpk,mk->mp", rel, e2)
    a = (p * p).sum(axis=1)
    b = (p * q).sum(axis=1)
    c = (q * q).sum(axis=1)
    theta = 0.5 * np.arctan2(2.0 * b, a - c)
    axis_y = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2
    axis_z = np.cross(axis_x, axis_y)

    for axis in (axis_y, axis_z):
        proj = np.einsum("mpk,mk->mp", rel, axis)
        extreme = proj[np.arange(m), np.argmax(np.abs(proj), axis=1)]
        axis *= np.where(extreme < 0.0, -1.0, 1.0)[:, None]

    return np.stack([axis_x, axis_y, axis_z], axis=1)


def branch_features(polylines: np.ndarray, no_branch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(L, C, F_E) for a stack of branch polylines.

    Recomputed from the polylines on every augmented view rather than cached, since jittering,
    deforming and scaling all change them. Rows flagged by `no_branch` are zeroed: a zero is a
    truthful "this node has no traced parent cable", and `is_synthetic` tells the encoder which
    kind of zero it is.
    """
    L = np.linalg.norm(np.diff(polylines, axis=1), axis=2).sum(axis=1)
    D = np.linalg.norm(polylines[:, -1] - polylines[:, 0], axis=1)
    C = np.divide(D, L, out=np.zeros_like(D), where=L > 1e-12)

    # The frame comes from all stored points -- a better principal direction than the 9 F_E samples
    # give -- and is then applied to the subsampled edge vectors.
    frames = _canonical_frames(polylines)
    edges = np.diff(polylines[:, ::FE_STRIDE], axis=1)
    fe = np.einsum("mij,mpj->mpi", frames, edges).reshape(len(polylines), -1)

    L = np.where(no_branch, 0.0, L)
    C = np.where(no_branch, 0.0, C)
    fe = np.where(no_branch[:, None], 0.0, fe)
    return L, C, fe


def node_features(xyz: np.ndarray, polylines: np.ndarray, is_synthetic: np.ndarray,
                  no_branch: np.ndarray, types: np.ndarray | None = None) -> np.ndarray:
    """Assemble the (M, 30) node feature matrix for one tree, or (M, 33) when `types` is given.

    The compartment one-hot is opt-in, and appended after the paper's block rather than inserted, so
    a checkpoint pretrained without it still lines up column-for-column. Without it the encoder
    cannot tell an axonal branch from a dendritic one; every other representation in the benchmark
    gets that distinction for free, because run_benchmark extracts each modality separately and
    concatenates. An unrecognized type code yields an all-zero block.

    The soma column is redundant with being the root in this cohort (exactly one type-1 node per
    cell, and _reparent_to_soma makes it the root), but it costs one dimension and keeps a soma
    distinguishable from an unrecognized code.
    """
    L, C, fe = branch_features(polylines, no_branch)
    width = N_FEATURES if types is None else N_FEATURES_TYPED
    out = np.empty((len(xyz), width), dtype=np.float32)
    out[:, COL_XYZ] = xyz
    out[:, COL_L] = L
    out[:, COL_C] = C
    out[:, COL_FE] = fe
    out[:, COL_SYNTHETIC] = is_synthetic
    if types is not None:
        for k, codes in enumerate(TYPE_GROUPS):
            out[:, N_FEATURES + k] = np.isin(types, codes)
    return out


def n_features(include_type: bool) -> int:
    return N_FEATURES_TYPED if include_type else N_FEATURES


def feature_names(include_type: bool = False) -> list[str]:
    names = ["x", "y", "z", "length", "contraction"]
    names += [f"fe_{e}_{ax}" for e in range(N_EDGES) for ax in "xyz"]
    names += ["is_synthetic"]
    if include_type:
        names += [f"type_{n}" for n in TYPE_GROUP_NAMES]
    return names
