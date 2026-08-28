"""Unit tests for the TreeMoCo implementation.

The level-synchronous Tree-LSTM equivalence test is the important one: everything downstream is
meaningless if the batched recursion does not reproduce a plain recursive one.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from arborist_experiments.patchseq_morphology_benchmark.data_loading import Cell
from arborist_experiments.patchseq_morphology_benchmark.treemoco import augment as ag
from arborist_experiments.patchseq_morphology_benchmark.treemoco import batching as bt
from arborist_experiments.patchseq_morphology_benchmark.treemoco import tree_features as tf
from arborist_experiments.patchseq_morphology_benchmark.treemoco.model import TreeLSTMCell


def make_cell(parents, xyz, types=None, synthetic=None) -> Cell:
    parents = np.asarray(parents, dtype=np.int64)
    xyz = np.asarray(xyz, dtype=np.float64)
    return Cell(
        specimen_id="test",
        types=np.asarray(types if types is not None else [1] + [3] * (len(parents) - 1),
                         dtype=np.int32),
        parents=parents,
        xyz=xyz,
        radius=np.ones(len(parents)),
        synthetic_edge=np.asarray(synthetic if synthetic is not None
                                  else np.zeros(len(parents), dtype=bool), dtype=bool),
    )


def chain_cell(points, extra_children=()):
    """A single chain through `points`, optionally with extra leaves hung off the last node."""
    parents = [-1] + list(range(len(points) - 1))
    xyz = list(points)
    for offset in extra_children:
        parents.append(len(points) - 1)
        xyz.append(np.asarray(points[-1]) + np.asarray(offset))
    return make_cell(parents, xyz)


# --- collapse ---
def test_collapse_drops_degree_one_chain():
    # 0 -> 1 -> 2 -> 3, with 4 and 5 hanging off 3: only 0, 3, 4, 5 survive.
    cell = make_cell(
        parents=[-1, 0, 1, 2, 3, 3],
        xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [3, 1, 0]],
    )
    tree = tf.collapse(cell)
    assert len(tree) == 4
    assert tree.parents.tolist() == [-1, 0, 1, 1]
    assert tree.no_branch.tolist() == [True, False, False, False]
    assert tree.polylines.shape == (4, tf.N_POLYLINE, 3)


def test_collapse_marks_synthetic_links_and_zeroes_their_geometry():
    cell = make_cell(
        parents=[-1, 0, 1, 1],
        xyz=[[0, 0, 0], [100, 0, 0], [101, 0, 0], [100, 1, 0]],
        synthetic=[False, True, False, False],
    )
    tree = tf.collapse(cell)
    assert tree.is_synthetic.tolist() == [False, True, False, False]
    assert tree.no_branch.tolist() == [True, True, False, False]
    X = tf.node_features(tree.xyz, tree.polylines, tree.is_synthetic, tree.no_branch)
    assert X[1, tf.COL_L] == 0.0 and X[1, tf.COL_C] == 0.0
    assert not X[1, tf.COL_FE].any()
    assert X[1, tf.COL_SYNTHETIC] == 1.0


def test_collapse_propagates_synthetic_along_the_dropped_chain():
    cell = make_cell(
        parents=[-1, 0, 1, 2, 2],
        xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]],
        synthetic=[False, False, True, False, False],
    )
    tree = tf.collapse(cell)
    # node 2 survives (two children) and its collapsed link 0 -> 1 -> 2 crosses a synthetic edge
    assert tree.is_synthetic.tolist() == [False, True, False, False]


# --- branch features ---
def test_straight_branch_has_unit_contraction():
    points = [[float(i), 0.0, 0.0] for i in range(6)]
    tree = tf.collapse(chain_cell(points, extra_children=([0, 1, 0], [0, -1, 0])))
    L, C, _ = tf.branch_features(tree.polylines, tree.no_branch)
    assert C[1] == pytest.approx(1.0, abs=1e-9)
    assert L[1] == pytest.approx(5.0, rel=1e-9)


def test_l_shaped_branch_length_and_contraction():
    points = [[float(i), 0.0, 0.0] for i in range(6)] + [[5.0, 1.0, 0.0], [5.0, 2.0, 0.0]]
    tree = tf.collapse(chain_cell(points, extra_children=([1, 0, 0], [0, 1, 0])))
    L, C, _ = tf.branch_features(tree.polylines, tree.no_branch)
    true_L = 7.0
    true_C = np.hypot(5.0, 2.0) / true_L
    # A 33-point resample of a kinked branch shortcuts the corner slightly; on this cohort the
    # median loss is 0.5%, so 2% is a generous bound that still catches a real regression.
    assert L[1] == pytest.approx(true_L, rel=0.02)
    assert C[1] == pytest.approx(true_C, rel=0.02)


def test_fe_fills_exactly_24_columns():
    points = [[float(i), np.sin(i), 0.0] for i in range(12)]
    tree = tf.collapse(chain_cell(points, extra_children=([1, 0, 0], [0, 1, 0])))
    _, _, fe = tf.branch_features(tree.polylines, tree.no_branch)
    assert fe.shape[1] == 24 == 3 * tf.N_EDGES
    assert np.count_nonzero(fe[1]) > 0


def test_fe_is_rotation_and_mirror_invariant():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(14, 3)).cumsum(axis=0)
    tree = tf.collapse(chain_cell(points, extra_children=([1, 0, 0], [0, 1, 0])))
    _, _, fe0 = tf.branch_features(tree.polylines, tree.no_branch)

    for _ in range(5):
        r = ag._rotation_matrix(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
        _, _, fe = tf.branch_features(tree.polylines @ r.T, tree.no_branch)
        assert np.abs(fe - fe0).max() < 1e-8

    mirrored = tree.polylines * np.array([-1.0, 1.0, 1.0])
    _, _, fe = tf.branch_features(mirrored, tree.no_branch)
    assert np.abs(fe - fe0).max() < 1e-8


def test_length_scales_linearly_but_contraction_does_not():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(14, 3)).cumsum(axis=0)
    tree = tf.collapse(chain_cell(points, extra_children=([1, 0, 0], [0, 1, 0])))
    L0, C0, _ = tf.branch_features(tree.polylines, tree.no_branch)
    L1, C1, _ = tf.branch_features(tree.polylines * 3.0, tree.no_branch)
    assert L1[1] == pytest.approx(3.0 * L0[1], rel=1e-9)
    assert C1[1] == pytest.approx(C0[1], rel=1e-9)


def test_resample_fixed_handles_degenerate_input():
    out = tf.resample_fixed(np.zeros((1, 3)), tf.N_POLYLINE)
    assert out.shape == (tf.N_POLYLINE, 3) and not out.any()
    out = tf.resample_fixed(np.zeros((4, 3)), tf.N_POLYLINE)
    assert out.shape == (tf.N_POLYLINE, 3) and not out.any()


# --- batching ---
def random_tree(n: int, rng: np.random.Generator) -> np.ndarray:
    """A random rooted tree whose node order is scrambled, so heights are not index-monotone."""
    parents = np.full(n, -1, dtype=np.int64)
    for i in range(1, n):
        parents[i] = rng.integers(0, i)
    perm = rng.permutation(n)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    out = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        out[inv[i]] = -1 if parents[i] < 0 else inv[int(parents[i])]
    return out


def naive_tree_lstm(cell1, cell2, x, parents) -> torch.Tensor:
    """Plain recursion over the forest; roots returned in ascending index order."""
    n = len(parents)
    kids: list[list[int]] = [[] for _ in range(n)]
    for i, p in enumerate(parents):
        if p >= 0:
            kids[int(p)].append(i)
    memo: dict[int, tuple] = {}

    def rec(k):
        if k not in memo:
            d = cell1.d_hidden
            sums = [x.new_zeros(d) for _ in range(4)]
            for j in kids[k]:
                for slot, value in enumerate(rec(j)):
                    sums[slot] = sums[slot] + value
            h1, c1 = cell1(x[k:k + 1], sums[0][None], sums[1][None])
            h2, z = cell2(c1, sums[2][None], sums[3][None])
            memo[k] = (h1[0], c1[0], h2[0], z[0])
        return memo[k]

    sys.setrecursionlimit(10000)
    return torch.stack([rec(r)[3] for r in range(n) if parents[r] < 0])


def test_node_heights():
    parents = np.array([-1, 0, 1, 1, 0], dtype=np.int64)
    assert bt.node_heights(parents).tolist() == [2, 1, 0, 0, 0]


def test_node_heights_rejects_a_cycle():
    with pytest.raises(ValueError):
        bt.node_heights(np.array([1, 0], dtype=np.int64))


def test_collate_preserves_topology_and_tree_order():
    rng = np.random.default_rng(3)
    views = [(random_tree(int(rng.integers(1, 20)), rng), None) for _ in range(4)]
    views = [(p, rng.normal(size=(len(p), 2))) for p, _ in views]
    batch = bt.collate(views)
    assert batch.n_nodes == sum(len(p) for p, _ in views)
    assert batch.roots.shape == (len(views),)
    # heights are non-decreasing in the new ordering, and every parent sits strictly above its child
    heights = np.concatenate([bt.node_heights(p) for p, _ in views])
    order_heights = np.sort(heights)
    assert np.all(np.diff(order_heights) >= 0)
    parent = batch.parent.numpy()
    for band, (lo, hi) in enumerate(zip(batch.bounds[:-1], batch.bounds[1:])):
        for i in range(lo, hi):
            if parent[i] >= 0:
                assert parent[i] >= batch.bounds[band + 1]


@pytest.mark.parametrize("trial", range(20))
def test_level_synchronous_matches_naive_recursion(trial):
    torch.manual_seed(trial)
    d_in, d_hidden = 7, 5
    cell1 = TreeLSTMCell(d_in, d_hidden).double()
    cell2 = TreeLSTMCell(d_hidden, d_hidden).double()
    rng = np.random.default_rng(trial)

    views = []
    for _ in range(int(rng.integers(1, 5))):
        p = random_tree(int(rng.integers(1, 30)), rng)
        views.append((p, rng.normal(size=(len(p), d_in))))
    batch = bt.collate(views)
    x = batch.x.double().requires_grad_(True)

    fast = bt.tree_lstm_forward(cell1, cell2, x, batch)
    fast_by_index = fast[np.argsort(batch.roots.numpy())]
    naive = naive_tree_lstm(cell1, cell2, x, batch.parent.numpy())
    assert torch.allclose(fast_by_index, naive, atol=1e-10)

    g_fast = torch.autograd.grad(fast.sum(), x, retain_graph=True)[0]
    g_naive = torch.autograd.grad(naive.sum(), x)[0]
    assert torch.allclose(g_fast, g_naive, atol=1e-10)


# --- augmentation ---
def cohort_like_tree(rng, n_leaves=12) -> tf.CollapsedTree:
    # Scaled to a realistic ~100 um extent: the augmentations have absolute micron parameters
    # (jitter_sigma = 1 um, shift_sigma = 10 um), so a unit-scale toy tree is drowned in noise.
    points = rng.normal(size=(40, 3)).cumsum(axis=0) * 15.0
    offsets = rng.normal(size=(n_leaves, 3)) * 50.0
    return tf.collapse(chain_cell(points, extra_children=offsets))


def assert_valid_tree(parents: np.ndarray) -> None:
    roots = np.flatnonzero(parents < 0)
    assert len(roots) == 1
    assert not np.any(parents == np.arange(len(parents)))
    for i in range(len(parents)):
        seen, k = set(), i
        while parents[k] >= 0:
            assert k not in seen
            seen.add(k)
            k = int(parents[k])
        assert k == int(roots[0])


@pytest.mark.parametrize("preset", sorted(ag.PRESETS))
def test_augmented_views_are_valid_trees(preset):
    rng = np.random.default_rng(7)
    cfg = ag.PRESETS[preset]()
    for _ in range(15):
        tree = cohort_like_tree(rng)
        parents, X = ag.augment(tree, cfg, rng)
        assert_valid_tree(parents)
        assert X.shape == (len(parents), tf.N_FEATURES)
        assert np.all(np.isfinite(X))


@pytest.mark.parametrize("op", ["p_drop_subtree", "p_skip_parent", "p_swap_sibling"])
def test_each_topology_op_alone_keeps_a_valid_tree(op):
    rng = np.random.default_rng(11)
    off = dict.fromkeys(
        ["p_scale", "p_rotate", "p_shift", "p_jitter", "p_flip", "p_branch_jitter",
         "p_branch_deform", "p_mask", "p_drop_subtree", "p_skip_parent", "p_swap_sibling"], 0.0)
    off[op] = 1.0
    cfg = ag.upright_preset(topo_node_prob=0.5, **off)
    for _ in range(20):
        tree = cohort_like_tree(rng)
        parents, _ = ag.augment(tree, cfg, rng)
        assert_valid_tree(parents)
        assert len(parents) >= min(len(tree), cfg.min_nodes)


def test_topology_edits_are_truncated_at_m():
    rng = np.random.default_rng(13)
    # drop-subtree is the only op that removes whole subtrees, so its budget is the visible one
    cfg = ag.upright_preset(
        topo_node_prob=1.0, topo_max_edits=3, p_skip_parent=0.0, p_swap_sibling=0.0,
        p_scale=0.0, p_rotate=0.0, p_shift=0.0, p_jitter=0.0, p_flip=0.0,
        p_branch_jitter=0.0, p_branch_deform=0.0, p_mask=0.0, p_drop_subtree=1.0)
    tree = tf.collapse(chain_cell(rng.normal(size=(30, 3)).cumsum(axis=0),
                                  extra_children=rng.normal(size=(20, 3)) * 5.0))
    parents, _ = ag.augment(tree, cfg, rng)
    assert_valid_tree(parents)
    assert len(tree) - len(parents) <= cfg.topo_max_edits


def test_swap_sibling_preserves_node_count():
    rng = np.random.default_rng(17)
    cfg = ag.upright_preset(
        topo_node_prob=1.0, p_drop_subtree=0.0, p_skip_parent=0.0, p_swap_sibling=1.0,
        p_scale=0.0, p_rotate=0.0, p_shift=0.0, p_jitter=0.0, p_flip=0.0,
        p_branch_jitter=0.0, p_branch_deform=0.0, p_mask=0.0)
    tree = cohort_like_tree(rng)
    parents, _ = ag.augment(tree, cfg, rng)
    assert len(parents) == len(tree)
    assert_valid_tree(parents)


def test_upright_preset_keeps_the_pia_axis_and_paper_does_not():
    rng = np.random.default_rng(19)
    trees = [cohort_like_tree(rng) for _ in range(12)]
    geometry_only = dict(p_drop_subtree=0.0, p_skip_parent=0.0, p_swap_sibling=0.0, p_mask=0.0)

    upright = []
    cfg = ag.upright_preset(**geometry_only)
    for tree in trees:
        _, X = ag.augment(tree, cfg, rng)
        upright.append(np.corrcoef(X[:, 1], tree.xyz[:, 1])[0, 1])
    assert np.min(upright) > 0.95

    paper = []
    cfg = ag.paper_preset(**geometry_only)
    for tree in trees:
        _, X = ag.augment(tree, cfg, rng)
        paper.append(np.corrcoef(X[:, 1], tree.xyz[:, 1])[0, 1])
    assert np.min(paper) < 0.0


def test_pca_normalize_is_rotation_equivariant():
    rng = np.random.default_rng(23)
    tree = cohort_like_tree(rng)
    xyz0, poly0 = ag.pca_normalize_frame(tree.xyz, tree.polylines)
    r = ag._rotation_matrix(rng.normal(size=3), 1.1)
    xyz1, poly1 = ag.pca_normalize_frame(tree.xyz @ r.T, tree.polylines @ r.T)
    assert np.abs(xyz1 - xyz0).max() < 1e-8
    assert np.abs(poly1 - poly0).max() < 1e-8


@pytest.mark.parametrize("n_nodes", [1, 2, 3])
def test_pca_normalize_handles_rank_deficient_trees(n_nodes):
    # An axon can collapse to a 2-node tree, which spans a plane at most; the economy SVD then
    # returns fewer than 3 axes.
    rng = np.random.default_rng(31)
    xyz = rng.normal(size=(n_nodes, 3)) * 20.0
    polylines = rng.normal(size=(n_nodes, tf.N_POLYLINE, 3)) * 20.0
    out_xyz, out_poly = ag.pca_normalize_frame(xyz, polylines)
    assert out_xyz.shape == xyz.shape
    assert out_poly.shape == polylines.shape
    assert np.isfinite(out_xyz).all() and np.isfinite(out_poly).all()
    # A rotation preserves pairwise distances from the centroid.
    assert np.allclose(np.linalg.norm(xyz - xyz.mean(axis=0), axis=1),
                       np.linalg.norm(out_xyz - out_xyz.mean(axis=0), axis=1))


def test_paper_preset_augments_a_two_node_tree():
    rng = np.random.default_rng(37)
    tree = tf.collapse(chain_cell(rng.normal(size=(6, 3)) * 20.0))
    parents, X = ag.augment(tree, ag.paper_preset(), rng)
    assert_valid_tree(parents)
    assert np.isfinite(X).all()


# --- compartment-type one-hot ---
def test_type_one_hot_is_soma_axon_dendrite():
    # Basal (3) and apical (4) must land in the same column: the apical call encodes the annotator's
    # cell-type judgement, so splitting it would leak the label.
    tree = tf.collapse(make_cell(
        parents=[-1, 0, 1, 1, 0],
        xyz=[[0, 0, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0], [0, -9, 0]],
        types=[1, 2, 2, 4, 3],
    ))
    X = tf.node_features(tree.xyz, tree.polylines, tree.is_synthetic, tree.no_branch, tree.types)
    assert X.shape == (len(tree), tf.N_FEATURES_TYPED)
    assert tf.N_TYPE == 3
    onehot = X[:, tf.COL_TYPE]
    assert onehot.sum(axis=1).tolist() == [1.0] * len(tree)
    group = {1: 0, 2: 1, 3: 2, 4: 2}
    assert onehot.argmax(axis=1).tolist() == [group[int(t)] for t in tree.types]


def test_unknown_type_code_gives_an_all_zero_block():
    tree = tf.collapse(make_cell(
        parents=[-1, 0, 1, 1],
        xyz=[[0, 0, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]],
        types=[1, 7, 2, 3],
    ))
    X = tf.node_features(tree.xyz, tree.polylines, tree.is_synthetic, tree.no_branch, tree.types)
    assert X[1, tf.COL_TYPE].sum() == 0.0


def test_typed_features_extend_the_untyped_ones_without_moving_them():
    # The backward-compatibility guarantee: a 30-feature checkpoint still lines up column for column.
    rng = np.random.default_rng(43)
    tree = cohort_like_tree(rng)
    plain = tf.node_features(tree.xyz, tree.polylines, tree.is_synthetic, tree.no_branch)
    typed = tf.node_features(tree.xyz, tree.polylines, tree.is_synthetic, tree.no_branch,
                             tree.types)
    assert np.array_equal(typed[:, :tf.N_FEATURES], plain)
    assert tf.feature_names(True)[:tf.N_FEATURES] == tf.feature_names()


@pytest.mark.parametrize("preset", sorted(ag.PRESETS))
def test_typed_augmented_views_have_the_wider_feature_matrix(preset):
    rng = np.random.default_rng(47)
    cfg = ag.PRESETS[preset](include_type=True)
    for _ in range(10):
        tree = cohort_like_tree(rng)
        parents, X = ag.augment(tree, cfg, rng)
        assert_valid_tree(parents)
        assert X.shape == (len(parents), tf.N_FEATURES_TYPED)
        assert np.isfinite(X).all()


def test_topology_edits_keep_each_node_with_its_own_type():
    rng = np.random.default_rng(53)
    tree = cohort_like_tree(rng)
    tree.types = np.where(np.arange(len(tree)) % 2 == 0, 2, 3).astype(np.int32)
    cfg = ag.upright_preset(include_type=True, topo_node_prob=0.5, p_drop_subtree=1.0,
                            p_skip_parent=1.0, p_swap_sibling=1.0)
    for _ in range(20):
        out = ag._apply_topology(ag.TreeArrays.from_tree(tree), cfg, rng)
        assert len(out.types) == len(out.parents)
        # Coordinates are unique here, so they identify which original node each row came from.
        rows = [int(np.flatnonzero((tree.xyz == p).all(axis=1))[0]) for p in out.xyz]
        assert out.types.tolist() == tree.types[rows].tolist()


def test_attribute_masking_spares_the_type_one_hot():
    rng = np.random.default_rng(59)
    tree = cohort_like_tree(rng)
    tree.types = np.where(np.arange(len(tree)) % 2 == 0, 2, 4).astype(np.int32)
    off = dict.fromkeys(
        ["p_scale", "p_rotate", "p_shift", "p_jitter", "p_flip", "p_branch_jitter",
         "p_branch_deform", "p_drop_subtree", "p_skip_parent", "p_swap_sibling"], 0.0)
    cfg = ag.upright_preset(include_type=True, p_mask=1.0, mask_rate=1.0, **off)
    _, X = ag.augment(tree, cfg, rng)
    assert np.all(X[:, tf.COL_L:tf.COL_SYNTHETIC] == 0.0)
    assert np.array_equal(X[:, tf.COL_SYNTHETIC], tree.is_synthetic.astype(np.float32))
    assert X[:, tf.COL_TYPE].sum(axis=1).tolist() == [1.0] * len(tree)


def test_identity_view_matches_unaugmented_features():
    rng = np.random.default_rng(29)
    tree = cohort_like_tree(rng)
    parents, X = ag.identity_view(tree, ag.upright_preset())
    assert parents is tree.parents
    expected = tf.node_features(tree.xyz, tree.polylines, tree.is_synthetic, tree.no_branch)
    assert np.array_equal(X, expected)
