import copy

import numpy as np
import torch

from arborist.skeleton_graph import SkeletonGraph
from arborist.utils.graph_utils import topological_decomposition
from neuron_proofreader.split_proofreading.split_feature_extraction import (
    get_feature_dict,
)

from arborist_experiments.split_correction import (
    ArboristVisionHGAT,
    ProposalTreeDataset,
    ProposalTreeEncoder,
    ProposalTreeFeatureExtractor,
    SplitDetector,
    bridge_points,
    proposal_length_feature,
    proposal_rooted_subgraph,
    spline_bridge,
    tail_path,
)


class _StubProposalGraph(SkeletonGraph):
    """Minimal stand-in for neuron_proofreader.ProposalGraph."""

    def __init__(self, xyz, edges, node_spacing=1.0):
        super().__init__(node_spacing=node_spacing)
        for _ in xyz:
            self.add_node(None)
        for u, v in edges:
            self.add_edge(u, v, None)
        self.node_xyz = np.asarray(xyz, dtype=float)
        self.node_feats = {"radius": np.ones(len(xyz))}
        self.proposals = set()
        self.gt_accepts = set()

    def list_proposals(self):
        return list(self.proposals)


def _two_fragments(gap=6.0, n=11):
    """Two collinear fragments along x separated by ``gap`` microns."""
    xyz_a = [(x, 0.0, 0.0) for x in range(n)]
    xyz_b = [(n - 1 + gap + k, 0.0, 0.0) for k in range(n)]
    edges = [(k, k + 1) for k in range(n - 1)]
    edges += [(n + k, n + k + 1) for k in range(n - 1)]
    graph = _StubProposalGraph(xyz_a + xyz_b, edges)
    proposal = frozenset((n - 1, n))  # facing leaves
    graph.proposals.add(proposal)
    return graph, proposal


def _two_arcs(radius=20.0, gap_deg=20.0, n=12, step_deg=3.0):
    """Two fragments lying on one circle in the xy-plane, separated by a gap."""
    def arc(theta0):
        th = np.deg2rad(theta0 + step_deg * np.arange(n))
        return [(radius * np.cos(t), radius * np.sin(t), 0.0) for t in th]
    A = arc(0.0)
    B = arc(step_deg * (n - 1) + gap_deg)
    edges = [(k, k + 1) for k in range(n - 1)] + [(n + k, n + k + 1) for k in range(n - 1)]
    graph = _StubProposalGraph(A + B, edges)
    proposal = frozenset((n - 1, n))
    graph.proposals.add(proposal)
    return graph, proposal, radius


def _path_to_root(subgraph, node):
    """Walks from ``node`` back to the root (0) along the unbranched chain."""
    path = [node]
    while path[-1] != 0:
        nxt = [w for w in subgraph.neighbors(path[-1]) if w not in path]
        nxt.sort(key=lambda w: np.linalg.norm(subgraph.node_xyz[w] - subgraph.node_xyz[0]))
        path.append(nxt[0])
    return path


def _hetero_inputs(samples, patch_shape=(16, 16, 16)):
    dims = get_feature_dict()
    return {
        "x_dict": {"branch": torch.randn(3, dims["branch"]), "proposal": torch.randn(2, dims["proposal"])},
        "img": torch.randn(2, 2, *patch_shape),
        "edge_index_dict": {
            ("branch", "to", "branch"): torch.tensor([[0, 1], [1, 2]]),
            ("proposal", "to", "proposal"): torch.tensor([[0, 1], [1, 0]]),
            ("branch", "to", "proposal"): torch.tensor([[0, 1, 2], [0, 0, 1]]),
            ("proposal", "to", "branch"): torch.tensor([[0, 0, 1], [0, 1, 2]]),
        },
        "tree_samples": samples,
    }


# --- Bridges and rooted subgraph ---
def test_bridge_points_spacing():
    pts = bridge_points(np.zeros(3), np.array([6.0, 0, 0]), spacing=1.0)
    assert pts.shape == (6, 3)
    steps = np.linalg.norm(np.diff(np.vstack([np.zeros(3), pts]), axis=0), axis=1)
    assert np.allclose(steps, 1.0)
    assert np.allclose(pts[-1], [6.0, 0, 0])
    assert bridge_points(np.zeros(3), np.zeros(3), 1.0).shape == (1, 3)


def test_rooted_subgraph_geometry():
    graph, proposal = _two_fragments(gap=6.0)
    depth = 8.0
    subgraph, (a, b) = proposal_rooted_subgraph(graph, proposal, depth)

    # Root sits at the proposal center and has one bridge per side
    assert np.allclose(subgraph.node_xyz[0], [13.0, 0, 0])
    assert subgraph.degree(0) == 2

    # Bridges land on the endpoints and keep unit spacing
    assert np.allclose(subgraph.node_xyz[a], graph.node_xyz[10])
    assert np.allclose(subgraph.node_xyz[b], graph.node_xyz[11])
    for endpoint in (a, b):
        xyz = subgraph.node_xyz[_path_to_root(subgraph, endpoint)]
        assert np.allclose(np.linalg.norm(np.diff(xyz, axis=0), axis=1), 1.0)

    # Every node lies strictly within the depth measured from the root
    dists = np.linalg.norm(subgraph.node_xyz - subgraph.node_xyz[0], axis=1)
    assert dists.max() < depth
    # Each side: 3 bridge nodes (the last is the endpoint) + 4 skeleton nodes
    # at path distance 4..7 from the root, plus the root itself
    assert subgraph.num_nodes() == 1 + 2 * (3 + 4)

    # Decomposition yields exactly the two proposal-side curves at the root
    _, paths, topo_edge_index = topological_decomposition(subgraph)
    assert len(paths) == 2
    assert all(src == 0 for src, _ in topo_edge_index)


def test_small_depth_keeps_only_bridge():
    graph, proposal = _two_fragments(gap=6.0)
    subgraph, (a, b) = proposal_rooted_subgraph(graph, proposal, depth=3.5)
    assert subgraph.num_nodes() == 7
    assert subgraph.degree(a) == 1 and subgraph.degree(b) == 1


def test_spline_bridge_follows_curvature():
    graph, proposal, R = _two_arcs()
    i, j = tuple(proposal)
    assert tail_path(graph, i, max_length=6.0)[-1] == i
    assert len(tail_path(graph, i, max_length=6.0)) > 1

    root, pts_i, pts_j = spline_bridge(graph, i, j, spacing=1.0, context=8.0)
    # Bridge points stay close to the circle, whereas the chord sags inward
    bridge = np.vstack([pts_i, root[None], pts_j])
    radial_err = np.abs(np.linalg.norm(bridge[:, :2], axis=1) - R)
    chord_mid = 0.5 * (graph.node_xyz[i] + graph.node_xyz[j])
    chord_err = abs(np.linalg.norm(chord_mid[:2]) - R)
    assert radial_err.max() < 0.25 * chord_err
    assert np.allclose(pts_i[-1], graph.node_xyz[i]) and np.allclose(pts_j[-1], graph.node_xyz[j])
    for pts in (pts_i, pts_j):
        steps = np.linalg.norm(np.diff(np.vstack([root, pts]), axis=0), axis=1)
        assert np.all(np.abs(steps - 1.0) < 0.35)

    sub, _ = proposal_rooted_subgraph(graph, proposal, depth=15.0, bridge_mode="spline")
    assert sub.degree(0) == 2
    sample = ProposalTreeFeatureExtractor(max_depth=15.0, bridge_mode="spline")(graph, proposal)
    assert sample.root_curve_indices == [0, 1]


def test_spline_falls_back_for_nonleaf_endpoints():
    graph, _ = _two_fragments()
    interior = frozenset((5, 11))  # no unbranched tail on the first side
    for mode in ("linear", "spline"):
        sub, _ = proposal_rooted_subgraph(graph, interior, depth=6.0, bridge_mode=mode)
        assert sub.num_nodes() > 0


# --- Extraction and datasets ---
def test_extractor_and_dataset():
    graph, proposal = _two_fragments()
    graph.gt_accepts.add(proposal)
    graph.proposals.add(frozenset((0, 21)))  # far leaves, never accepted

    extractor = ProposalTreeFeatureExtractor(max_depth=8.0, min_curve_len=10)
    sample = extractor(graph, proposal)
    assert len(sample.curves) == 2
    assert sample.root_curve_indices == [0, 1]
    assert all(np.allclose(c[0], 0) for c in sample.curves)
    assert all(len(c) >= 10 for c in sample.curves)
    assert sample.proposal_length == 6.0  # gap between the facing leaves
    assert extractor.config["max_depth"] == 8.0

    ds = ProposalTreeDataset(graph, extractor, rebalance_classes=False)
    assert len(ds) == 2 and ds.n_proposals() == 2
    labels = sorted(label for _, label in (ds[i] for i in range(len(ds))))
    assert labels == [0, 1]


# --- Models ---
def test_encoder_baseline_and_hgat_forward():
    graph, _ = _two_fragments()
    graph.proposals.add(frozenset((0, 21)))
    extractor = ProposalTreeFeatureExtractor(max_depth=8.0)
    samples = [extractor(graph, p) for p in graph.list_proposals()]

    length = proposal_length_feature(samples, device="cpu")
    assert length.shape == (2, 1)
    expected = torch.log1p(torch.tensor([6.0, 26.0]))  # gap and far-leaf proposals
    assert torch.allclose(length[:, 0].sort().values, expected, atol=1e-6)

    tree_kwargs = dict(curve_d_token=16, curve_n_layers=1, graph_n_layers=1)
    encoder = ProposalTreeEncoder(latent_dim=16, **tree_kwargs)
    assert encoder(samples).shape == (2, 16)

    baseline = SplitDetector(latent_dim=16, hidden_dim=32, **tree_kwargs).eval()
    with torch.no_grad():
        assert baseline(samples).shape == (2, 1)
        assert baseline({"tree_sample": samples}).shape == (2, 1)

    patch_shape = (16, 16, 16)
    inputs = _hetero_inputs(samples, patch_shape)
    model = ArboristVisionHGAT(
        patch_shape, tree_embed_dim=16, tree_encoder_kwargs=tree_kwargs,
        img_embed_dim=16, hidden_dim=32,
    ).eval()
    assert model.proposal_encoder is None and model.branch_encoder is None
    assert model.branch_relay.shape == (32,)
    assert model.fusion.layers[0].in_features == 16 + 16 + 1 if hasattr(model.fusion, "layers") else True
    assert model.config["model_type"] == "ArboristVisionHGAT"
    with torch.no_grad():
        logits = model(copy.deepcopy(inputs))
    assert logits.shape == (2, 1)
