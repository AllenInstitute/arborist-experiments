"""
Created on Fri Sep 19 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Extraction of proposal-rooted subgraphs for split correction.

A split-correction proposal is a candidate edge between two fragments. To
encode it with Arborist we build a rooted subgraph whose root is a virtual
node at the center of the proposal. Two "bridge" paths, resampled at the
graph's node spacing, connect the root to the proposal endpoints, and the
skeleton is then traversed outward from each endpoint up to a path-length
depth measured from the root. The result is an ordinary arborist
SkeletonGraph, so ``topological_decomposition`` and ``TreeSample`` work
unchanged.

Two bridge geometries are supported. ``"linear"`` joins the endpoints with a
straight segment. ``"spline"`` fits a cubic spline through the tail of each
fragment leading into the proposal and samples the bridge from it, so the
bridge continues each fragment smoothly and any misalignment between the two
fragments shows up as curvature rather than as a kink.

"""

import numpy as np
from scipy.interpolate import splev, splprep

from arborist.data.datasets import TreeSample, _build_line_graph_edge_index
from arborist.skeleton_graph import SkeletonGraph
from arborist.utils.graph_utils import topological_decomposition


# --- Bridges ---
def bridge_points(start, end, spacing):
    """
    Samples points along the straight segment from start to end so that
    consecutive points are approximately ``spacing`` apart.

    Parameters
    ----------
    start : numpy.ndarray
        Start coordinate, shape (3,). Excluded from the output.
    end : numpy.ndarray
        End coordinate, shape (3,). Always the last output point.
    spacing : float
        Target distance between consecutive points (microns).

    Returns
    -------
    numpy.ndarray
        Shape (n, 3) with n >= 1, ordered from start toward end.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    length = np.linalg.norm(end - start)
    n = max(1, int(round(length / spacing))) if spacing > 0 else 1
    t = np.linspace(0.0, 1.0, n + 1)[1:]
    return start + t[:, None] * (end - start)


def tail_path(graph, node, max_length):
    """
    Walks from a leaf into its fragment along the unbranched chain.

    Parameters
    ----------
    graph : SkeletonGraph
        Fragments graph.
    node : int
        Leaf node to start from.
    max_length : float
        Maximum path length (microns) to walk.

    Returns
    -------
    List[int]
        Node IDs ordered from the far end of the chain to ``node``. If
        ``node`` is not a leaf the list contains only ``node``.
    """
    if graph.degree(node) != 1:
        return [node]
    path, length = [node], 0.0
    prev, curr = None, node
    while True:
        nbrs = [v for v in graph.neighbors(curr) if v != prev]
        if len(nbrs) != 1 and curr != node:
            break
        if not nbrs:
            break
        nxt = nbrs[0]
        step = graph.dist(curr, nxt)
        if length + step > max_length:
            break
        length += step
        path.append(nxt)
        prev, curr = curr, nxt
    return path[::-1]


def spline_bridge(graph, i, j, spacing, context=16.0):
    """
    Samples a bridge between two endpoints from a cubic spline fitted through
    the tails of both fragments.

    Parameters
    ----------
    graph : SkeletonGraph
        Fragments graph.
    i, j : int
        Proposal endpoints.
    spacing : float
        Target distance between consecutive bridge points (microns).
    context : float, optional
        Path length (microns) of each fragment tail used for the fit.
        Default is 16.

    Returns
    -------
    root : numpy.ndarray
        Bridge center, shape (3,).
    pts_i : numpy.ndarray
        Points from the root toward ``i`` (exclusive of root, inclusive of
        ``i``), shape (n_i, 3).
    pts_j : numpy.ndarray
        Points from the root toward ``j``, shape (n_j, 3).
    """
    tail_i = tail_path(graph, i, context)
    tail_j = tail_path(graph, j, context)[::-1]
    xyz = np.asarray(graph.node_xyz[np.array(tail_i + tail_j)], dtype=float)

    # Drop consecutive duplicates, which splprep rejects
    keep = np.r_[True, np.linalg.norm(np.diff(xyz, axis=0), axis=1) > 1e-9]
    xyz = xyz[keep]
    idx_i = int(np.sum(keep[: len(tail_i)])) - 1
    idx_j = idx_i + 1
    if len(xyz) < 3 or idx_j >= len(xyz):
        return _linear_bridge(graph.node_xyz[i], graph.node_xyz[j], spacing)

    # Chord-length parameterization so u tracks arc length
    u = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    k = min(3, len(xyz) - 1)
    try:
        tck, _ = splprep(xyz.T, u=u, k=k, s=0)
    except (ValueError, TypeError):
        return _linear_bridge(graph.node_xyz[i], graph.node_xyz[j], spacing)

    u_i, u_j = u[idx_i], u[idx_j]
    u_mid = 0.5 * (u_i + u_j)
    n_half = max(1, int(round(0.5 * (u_j - u_i) / spacing))) if spacing > 0 else 1

    def sample(u_from, u_to, endpoint_xyz):
        us = np.linspace(u_from, u_to, n_half + 1)[1:]
        pts = np.column_stack(splev(us, tck))
        pts[-1] = endpoint_xyz  # land exactly on the endpoint
        return pts

    root = np.array(splev(u_mid, tck), dtype=float)
    pts_i = sample(u_mid, u_i, graph.node_xyz[i])
    pts_j = sample(u_mid, u_j, graph.node_xyz[j])
    return root, pts_i, pts_j


def _linear_bridge(xyz_i, xyz_j, spacing):
    xyz_i = np.asarray(xyz_i, dtype=float)
    xyz_j = np.asarray(xyz_j, dtype=float)
    root = 0.5 * (xyz_i + xyz_j)
    return root, bridge_points(root, xyz_i, spacing), bridge_points(root, xyz_j, spacing)


# --- Rooted Subgraph ---
def proposal_rooted_subgraph(
    graph,
    proposal,
    depth,
    node_spacing=None,
    bridge_mode="linear",
    spline_context=16.0,
):
    """
    Builds a rooted subgraph centered on a proposal.

    Node 0 of the returned subgraph is a virtual root at the center of the
    proposal bridge. Bridge nodes lead from the root to each proposal
    endpoint, and the skeleton is expanded outward from the endpoints until
    the path length from the root reaches ``depth``. Proposals are never
    traversed; only skeleton edges are.

    Parameters
    ----------
    graph : SkeletonGraph
        Fragments graph (e.g. a neuron_proofreader ProposalGraph) exposing
        ``node_xyz``, ``node_feats``, ``neighbors``, ``degree``, ``dist``
        and ``node_spacing``.
    proposal : Frozenset[int]
        Pair of node IDs forming the proposal.
    depth : float
        Subgraph depth: maximum path length (microns) from the root to any
        subgraph node.
    node_spacing : float, optional
        Spacing of bridge nodes. Default is ``graph.node_spacing``.
    bridge_mode : {"linear", "spline"}, optional
        Bridge geometry. "linear" joins the endpoints with a straight
        segment whose midpoint is the root. "spline" fits a cubic spline
        through the tails of both fragments and samples the bridge from it,
        with the root at the bridge's arc-length midpoint. Default is
        "linear".
    spline_context : float, optional
        Path length (microns) of each fragment tail used for the spline fit.
        Default is 16.

    Returns
    -------
    subgraph : SkeletonGraph
        Rooted subgraph with ``node_xyz`` and ``node_feats`` populated.
    endpoints : Tuple[int, int]
        Subgraph node IDs of the two proposal endpoints, in the order the
        proposal iterates.
    """
    if bridge_mode not in ("linear", "spline"):
        raise ValueError(f"Unknown bridge_mode: {bridge_mode!r}")
    i, j = tuple(proposal)
    spacing = node_spacing or getattr(graph, "node_spacing", None) or 1.0
    radius_feats = getattr(graph, "node_feats", {}).get("radius")

    def radius_of(node):
        return float(radius_feats[node]) if radius_feats is not None else 1.0

    # Bridge geometry
    if bridge_mode == "spline":
        center, pts_i, pts_j = spline_bridge(graph, i, j, spacing, spline_context)
    else:
        center, pts_i, pts_j = _linear_bridge(
            graph.node_xyz[i], graph.node_xyz[j], spacing
        )

    # Root
    subgraph = SkeletonGraph(anisotropy=graph.anisotropy, node_spacing=spacing)
    subgraph.add_node(None)
    xyz_list = [center]
    radius_list = [(radius_of(i) + radius_of(j)) / 2.0]

    # Bridges from root to each endpoint
    node_mapping = dict()
    start_dist = dict()
    for endpoint, pts in ((i, pts_i), (j, pts_j)):
        r_center, r_end = radius_list[0], radius_of(endpoint)
        prev, prev_xyz, walked = 0, center, 0.0
        for k, xyz in enumerate(pts):
            idx = subgraph.add_node(None)
            subgraph.add_edge(prev, idx, None)
            frac = (k + 1) / len(pts)
            xyz_list.append(xyz)
            radius_list.append((1 - frac) * r_center + frac * r_end)
            walked += float(np.linalg.norm(xyz - prev_xyz))
            prev, prev_xyz = idx, xyz
        node_mapping[endpoint] = prev
        # Round away accumulated float error so the depth cutoff is exact
        start_dist[endpoint] = round(walked, 9)

    # Expand into the skeleton from both endpoints
    visited = {i, j}
    queue = [(i, start_dist[i]), (j, start_dist[j])]
    while queue:
        u, dist_u = queue.pop()
        for v in graph.neighbors(u):
            dist_v = dist_u + graph.dist(u, v)
            if v not in visited and dist_v < depth:
                idx = subgraph.add_node(None)
                subgraph.add_edge(node_mapping[u], idx, None)
                node_mapping[v] = idx
                xyz_list.append(np.asarray(graph.node_xyz[v], dtype=float))
                radius_list.append(radius_of(v))
                queue.append((v, dist_v))
                visited.add(v)

    subgraph.node_xyz = np.array(xyz_list, dtype=np.asarray(graph.node_xyz).dtype)
    subgraph.node_feats = {"radius": np.array(radius_list, dtype=float)}
    return subgraph, (node_mapping[i], node_mapping[j])


class ProposalTreeFeatureExtractor:
    """
    Turns a proposal into an Arborist TreeSample. Holds the extraction
    settings only, so one instance can serve every fragments graph.

    Parameters
    ----------
    max_depth : float, optional
        Subgraph depth in microns, measured as path length from the proposal
        center (the root). Default is 100.
    min_curve_len : int, optional
        Curves shorter than this are zero-padded so CurveEncoder produces at
        least one segment token. Should match ``curve_segment_len``. Default
        is 10.
    node_spacing : float, optional
        Spacing of bridge nodes. Default is ``graph.node_spacing``.
    bridge_mode : {"linear", "spline"}, optional
        See ``proposal_rooted_subgraph``. Default is "linear".
    spline_context : float, optional
        See ``proposal_rooted_subgraph``. Default is 16.
    transform : callable, optional
        Applied to each curve's raw xyz array before differencing.
    graph_transform : callable, optional
        Applied to the full node_xyz array of the subgraph before
        decomposition, so all curves share one rigid transform.
    """

    def __init__(
        self,
        max_depth=100,
        min_curve_len=10,
        node_spacing=None,
        bridge_mode="linear",
        spline_context=16.0,
        transform=None,
        graph_transform=None,
    ):
        self.max_depth = max_depth
        self.min_curve_len = min_curve_len
        self.node_spacing = node_spacing
        self.bridge_mode = bridge_mode
        self.spline_context = spline_context
        self.transform = transform
        self.graph_transform = graph_transform

    @property
    def config(self):
        return {
            "max_depth": self.max_depth,
            "min_curve_len": self.min_curve_len,
            "node_spacing": self.node_spacing,
            "bridge_mode": self.bridge_mode,
            "spline_context": self.spline_context,
            "transform": type(self.transform).__name__ if self.transform else None,
            "graph_transform": (
                type(self.graph_transform).__name__ if self.graph_transform else None
            ),
        }

    def __call__(self, graph, proposal):
        """
        Parameters
        ----------
        graph : SkeletonGraph
            Fragments graph containing the proposal endpoints.
        proposal : Frozenset[int]
            Pair of node IDs forming the proposal.

        Returns
        -------
        TreeSample
            Curves are first-order differences with a leading zero row.
            ``root_curve_indices`` marks the curves incident to the root,
            i.e. the two sides of the proposal. ``proposal_length`` holds the
            Euclidean distance (microns) between the proposal endpoints,
            which the curves alone do not expose.
        """
        subgraph, _ = proposal_rooted_subgraph(
            graph,
            proposal,
            self.max_depth,
            node_spacing=self.node_spacing,
            bridge_mode=self.bridge_mode,
            spline_context=self.spline_context,
        )
        if self.graph_transform:
            subgraph.node_xyz = self.graph_transform(subgraph.node_xyz)
        _, paths, topo_edge_index = topological_decomposition(subgraph)

        curves = []
        for path in paths:
            xyz = subgraph.node_xyz[path].copy()
            if self.transform:
                xyz = self.transform(xyz)
            xyz -= xyz[0]
            xyz[1:] -= xyz[:-1].copy()
            if len(xyz) < self.min_curve_len:
                pad = np.zeros((self.min_curve_len - len(xyz), 3), dtype=xyz.dtype)
                xyz = np.concatenate([xyz, pad], axis=0)
            curves.append(xyz)

        edge_index = _build_line_graph_edge_index(topo_edge_index)
        root_curve_indices = [
            k for k, (src, _) in enumerate(topo_edge_index) if src == 0
        ]
        sample = TreeSample(
            curves=curves,
            edge_index=edge_index,
            root_curve_indices=root_curve_indices,
        )
        sample.proposal_length = float(graph.dist(*tuple(proposal)))
        return sample
