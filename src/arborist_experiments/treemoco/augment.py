"""TreeMoCo data augmentations -- Chen et al. 2022, Table 1.

All eleven augmentations, grouped as the paper groups them. Table 1's "Target" column is load
bearing: category (i) targets Coordinates and category (ii)/(iii) target Branch feats, so
"deforming branches" rescales a branch's own geometry without moving any node coordinate.

`arborist.data.augmentation.RandomRotation3D` / `RandomMirror3D` / `RandomJitter3D` are deliberately
not reused. Each draws from the global `random`/`np.random` state and centers on the centroid of the
single array it is handed. Here one transform has to be shared by the node coordinates *and* every
branch polyline -- two calls would rotate them by different angles about different centers, silently
decoupling a node from its own branch -- and rotation/scaling must be about the soma at the origin,
not the arbor centroid, or the augmentation translates the soma as a side effect.

Two presets:

  upright  Primary. Keeps the upright frame, in which y is the pia direction. Rotation is limited
           to +/-15 degrees about y and mirroring to x and z. A free rotation about y would blend the
           ~550 um tangential x extent into the ~200 um slice-normal z extent, and mirroring y would
           exchange pia for white matter. On Patch-seq the pia-relative frame is where the signal
           is: density_xy scores 0.58 accuracy against density_xz's 0.32.
  paper    Faithful to section 4.1 plus Table 1: PCA-normalize each tree's orientation, then rotate
           freely and flip on all three axes. Deliberately rotation-invariant, i.e. it throws the
           laminar frame away.

Topology augmentations break the consistency between a node's coordinate and its parent branch --
skipping a parent attaches a branch to a node it never touched. The paper accepts this for trees
(section 3.3 argues only that general-graph edge perturbation is worse), and the branch keeps its
own intrinsic shape, so L, C and F_E stay meaningful even when the link is new.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tree_features import COL_L, COL_SYNTHETIC, CollapsedTree, node_features

CORTICAL_AXIS = 1                   # upright y, the pia direction


@dataclass
class AugmentConfig:
    name: str
    pca_normalize: bool = False

    # (i) point transformations, applied to node coordinates and branch polylines together
    p_scale: float = 0.5
    scale_range: tuple[float, float] = (0.8, 1.25)
    p_rotate: float = 0.5
    rotate_axis: int | None = None                  # None = free 3D rotation
    rotate_max_angle: float = np.pi
    p_shift: float = 0.5
    shift_sigma: float = 10.0                       # um
    p_jitter: float = 0.5
    jitter_sigma: float = 1.0                        # um, ~ the 1.15 um median node spacing
    p_flip: float = 0.5
    flip_axes: tuple[int, ...] = (0, 1, 2)

    # (ii) morphology, branch geometry only
    p_branch_jitter: float = 0.5
    branch_jitter_sigma: float = 0.5                 # um
    p_branch_deform: float = 0.5
    branch_deform_range: tuple[float, float] = (0.8, 1.25)

    # (iii) attribute masking on the branch-feature block
    p_mask: float = 0.5
    mask_rate: float = 0.15

    # (iv) topology
    p_drop_subtree: float = 0.5
    p_skip_parent: float = 0.5
    p_swap_sibling: float = 0.5
    topo_node_prob: float = 0.05                     # per-node edit probability, paper's 0.05
    topo_max_edits: int = 10                         # paper's m = 10
    min_nodes: int = 4                               # never edit a tree below this

    # (v) node identity, not in the paper. Appends a one-hot of the SWC compartment type.
    include_type: bool = False


def upright_preset(**overrides) -> AugmentConfig:
    cfg = AugmentConfig(
        name="upright",
        pca_normalize=False,
        rotate_axis=CORTICAL_AXIS,
        rotate_max_angle=float(np.deg2rad(15.0)),
        flip_axes=(0, 2),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def paper_preset(**overrides) -> AugmentConfig:
    cfg = AugmentConfig(
        name="paper",
        pca_normalize=True,
        rotate_axis=None,
        rotate_max_angle=np.pi,
        flip_axes=(0, 1, 2),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


PRESETS = {"upright": upright_preset, "paper": paper_preset}


@dataclass
class TreeArrays:
    """Mutable working copy of a CollapsedTree, the unit augmentations act on."""

    parents: np.ndarray
    xyz: np.ndarray
    polylines: np.ndarray
    is_synthetic: np.ndarray
    no_branch: np.ndarray
    types: np.ndarray
    children: list[list[int]] = field(default_factory=list)

    @classmethod
    def from_tree(cls, tree: CollapsedTree) -> TreeArrays:
        return cls(
            parents=tree.parents.copy(),
            xyz=tree.xyz.copy(),
            polylines=tree.polylines.copy(),
            is_synthetic=tree.is_synthetic.copy(),
            no_branch=tree.no_branch.copy(),
            types=tree.types.copy(),
        )

    def __len__(self):
        return len(self.parents)


def pca_normalize_frame(xyz: np.ndarray, polylines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate into the tree's own principal axes, section 4.1.

    The paper does this to stop the registered CCF orientation from leaking anatomy into the
    representation. Here it is the opposite trade: it discards the pia frame, which is why it is
    confined to the `paper` preset. Axis signs are fixed from the largest-magnitude coordinate so
    the result is deterministic.
    """
    centered = xyz - xyz.mean(axis=0)
    # full_matrices=True so vt is always 3x3: a tree with fewer than 3 nodes spans a lower-rank
    # subspace, and the economy SVD would return only as many axes as there are nodes. The surplus
    # rows complete an orthonormal basis, and the sign loop below leaves them alone because their
    # projections are all zero.
    _, _, vt = np.linalg.svd(centered, full_matrices=True)
    axes = vt.copy()
    for k in range(3):
        proj = centered @ axes[k]
        if proj[np.argmax(np.abs(proj))] < 0.0:
            axes[k] *= -1.0
    return xyz @ axes.T, polylines @ axes.T


# --- (i) point transformations ---
def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _apply_point_transforms(t: TreeArrays, cfg: AugmentConfig, rng: np.random.Generator) -> None:
    if rng.random() < cfg.p_scale:
        s = rng.uniform(*cfg.scale_range)
        t.xyz *= s
        t.polylines *= s

    if rng.random() < cfg.p_rotate:
        if cfg.rotate_axis is None:
            axis = rng.normal(size=3)
            if np.linalg.norm(axis) < 1e-9:
                axis = np.array([0.0, 1.0, 0.0])
        else:
            axis = np.eye(3)[cfg.rotate_axis]
        r = _rotation_matrix(axis, rng.uniform(-cfg.rotate_max_angle, cfg.rotate_max_angle))
        t.xyz = t.xyz @ r.T
        t.polylines = t.polylines @ r.T

    if rng.random() < cfg.p_shift:
        d = rng.normal(0.0, cfg.shift_sigma, size=3)
        t.xyz += d
        t.polylines += d

    if rng.random() < cfg.p_flip:
        sign = np.ones(3)
        for ax in cfg.flip_axes:
            if rng.random() < 0.5:
                sign[ax] = -1.0
        t.xyz *= sign
        t.polylines *= sign

    if rng.random() < cfg.p_jitter:
        # Polylines are jittered and node coordinates re-derived from the branch endpoint, so a node
        # never drifts away from the end of its own branch. Nodes without a branch (root, synthetic
        # link) get their own noise.
        t.polylines += rng.normal(0.0, cfg.jitter_sigma, size=t.polylines.shape)
        has_branch = ~t.no_branch
        t.xyz[has_branch] = t.polylines[has_branch, -1]
        t.xyz[t.no_branch] += rng.normal(
            0.0, cfg.jitter_sigma, size=(int(t.no_branch.sum()), 3))


# --- (ii) morphology ---
def _apply_morphology(t: TreeArrays, cfg: AugmentConfig, rng: np.random.Generator) -> None:
    if rng.random() < cfg.p_branch_jitter:
        # Interior points only: the endpoints are the parent and child node positions, so perturbing
        # them would be a coordinate transform, which is category (i)'s job.
        noise = rng.normal(0.0, cfg.branch_jitter_sigma, size=t.polylines.shape)
        noise[:, 0] = 0.0
        noise[:, -1] = 0.0
        t.polylines = t.polylines + noise

    if rng.random() < cfg.p_branch_deform:
        s = rng.uniform(*cfg.branch_deform_range, size=(len(t), 1, 1))
        start = t.polylines[:, :1]
        t.polylines = start + (t.polylines - start) * s


# --- (iv) topology ---
def _build_children(parents: np.ndarray) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(len(parents))]
    for i, p in enumerate(parents):
        if p >= 0:
            children[int(p)].append(i)
    return children


def _subtree_nodes(children: list[list[int]], root: int) -> list[int]:
    out, stack = [], [root]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(children[n])
    return out


def _bfs_order(children: list[list[int]], root: int) -> list[int]:
    order, queue = [], [root]
    while queue:
        n = queue.pop(0)
        order.append(n)
        queue.extend(children[n])
    return order


def _drop_subtrees(children, root, cfg, rng) -> None:
    """(9) Truncated dropping subtrees: root-to-leaf visit, drop a child subtree with small prob."""
    edits = 0
    for node in _bfs_order(children, root):
        if edits >= cfg.topo_max_edits:
            break
        if not children[node] or rng.random() >= cfg.topo_node_prob:
            continue
        victim = children[node][rng.integers(len(children[node]))]
        children[node].remove(victim)
        edits += 1


def _skip_parents(children, parents, root, cfg, rng) -> None:
    """(10) Truncated skipping parent node: replace a node by one of its subtrees."""
    edits = 0
    for node in _bfs_order(children, root):
        if edits >= cfg.topo_max_edits:
            break
        p = int(parents[node])
        if p < 0 or not children[node] or node not in children[p]:
            continue
        if rng.random() >= cfg.topo_node_prob:
            continue
        keeper = children[node][rng.integers(len(children[node]))]
        children[p].remove(node)
        children[p].append(keeper)
        parents[keeper] = p
        children[node] = []
        edits += 1


def _swap_siblings(children, parents, root, cfg, rng) -> None:
    """(11) Truncated swapping sibling child: exchange a subtree with a sibling's subtree."""
    edits = 0
    for node in _bfs_order(children, root):
        if edits >= cfg.topo_max_edits:
            break
        kids = children[node]
        if len(kids) < 2 or rng.random() >= cfg.topo_node_prob:
            continue
        a, b = rng.choice(len(kids), size=2, replace=False)
        u, v = kids[int(a)], kids[int(b)]
        if not children[u] or not children[v]:
            continue
        su = children[u][rng.integers(len(children[u]))]
        sv = children[v][rng.integers(len(children[v]))]
        children[u].remove(su)
        children[v].remove(sv)
        children[u].append(sv)
        children[v].append(su)
        parents[sv] = u
        parents[su] = v
        edits += 1


def _apply_topology(t: TreeArrays, cfg: AugmentConfig, rng: np.random.Generator) -> TreeArrays:
    roots = np.flatnonzero(t.parents < 0)
    if len(roots) != 1 or len(t) < cfg.min_nodes:
        return t
    root = int(roots[0])

    children = _build_children(t.parents)
    parents = t.parents.copy()

    if rng.random() < cfg.p_drop_subtree:
        _drop_subtrees(children, root, cfg, rng)
    if rng.random() < cfg.p_skip_parent:
        _skip_parents(children, parents, root, cfg, rng)
    if rng.random() < cfg.p_swap_sibling:
        _swap_siblings(children, parents, root, cfg, rng)

    keep = np.array(sorted(_subtree_nodes(children, root)), dtype=np.int64)
    if len(keep) < cfg.min_nodes:
        return t                                    # edits gutted a small tree; keep the original
    if len(keep) == len(t):
        return TreeArrays(parents=parents, xyz=t.xyz, polylines=t.polylines,
                          is_synthetic=t.is_synthetic, no_branch=t.no_branch, types=t.types)

    remap = np.full(len(t), -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    new_parents = np.where(parents[keep] >= 0, remap[parents[keep]], -1)
    return TreeArrays(
        parents=new_parents,
        xyz=t.xyz[keep],
        polylines=t.polylines[keep],
        is_synthetic=t.is_synthetic[keep],
        no_branch=t.no_branch[keep],
        types=t.types[keep],
    )


# --- Driver ---
def augment(tree: CollapsedTree, cfg: AugmentConfig,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One augmented view.

    Returns
    -------
    parents : np.ndarray
        (M',) int64 collapsed parent index, -1 at the root.
    X : np.ndarray
        (M', 30) float32 node features, or (M', 34) when `cfg.include_type`.
    """
    t = TreeArrays.from_tree(tree)
    if cfg.pca_normalize:
        t.xyz, t.polylines = pca_normalize_frame(t.xyz, t.polylines)

    t = _apply_topology(t, cfg, rng)
    _apply_point_transforms(t, cfg, rng)
    _apply_morphology(t, cfg, rng)

    X = node_features(t.xyz, t.polylines, t.is_synthetic, t.no_branch,
                      t.types if cfg.include_type else None)

    # (iii) attribute masking, dropout-style on the branch-feature block. is_synthetic and the
    # compartment one-hot are exempt: masking them would tell the encoder an invented link is real
    # cable, or an axon a dendrite. Columns are named rather than sliced from the end so appending
    # the one-hot does not silently move the block.
    if rng.random() < cfg.p_mask:
        block = X[:, COL_L:COL_SYNTHETIC]
        block *= (rng.random(block.shape) >= cfg.mask_rate)
    return t.parents, X


def identity_view(tree: CollapsedTree, cfg: AugmentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Un-augmented features, for inference. Honours the preset's frame normalization only."""
    xyz, polylines = tree.xyz, tree.polylines
    if cfg.pca_normalize:
        xyz, polylines = pca_normalize_frame(xyz, polylines)
    return tree.parents, node_features(xyz, polylines, tree.is_synthetic, tree.no_branch,
                                       tree.types if cfg.include_type else None)
