"""
Created on Sun August 10 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Dataset utilities for training a merge-detection model on graph structure only.

"""

import os

import numpy as np
import rustworkx as rx
import torch
from torch.utils.data import Dataset

from arborist.data.datasets import TreeSample, _build_line_graph_edge_index
from arborist.utils.graph_utils import topological_decomposition
from arborist.utils.swc_loading import Reader
from arborist.utils.util import write_json

from neuron_proofreader.utils.ml_util import TensorDict


class GraphSearchDataset(Dataset):
    """
    Unlabelled dataset over candidate nodes of a FragmentsGraph.

    Used during evaluation: each item is a (node_id, TreeSample) pair so the
    caller can look up xyz coordinates after running inference.

    Parameters
    ----------
    graph : FragmentsGraph
        Graph to search over.
    depth : float
        Subgraph extraction radius in microns.
    min_curve_len : int, optional
        Minimum number of points per curve; shorter curves are zero-padded.
        Default is 10.
    search_type : str, optional
        Strategy for selecting candidate nodes. "sparse" visits only branching
        nodes (degree >= 3); "dense" steps through each connected component
        via DFS at intervals of ``step_size`` microns, also including branching
        nodes. Default is "sparse".
    step_size : float, optional
        Spacing between sampled nodes in dense mode (microns). Ignored when
        search_type is "sparse". Default is 40.
    """

    def __init__(self, graph, depth, min_curve_len=10, search_type="sparse", step_size=40):
        super().__init__()
        self.graph = graph
        self.depth = depth
        self.min_curve_len = min_curve_len
        self.search_type = search_type
        if search_type == "sparse":
            self.nodes = list(graph.branching_nodes())
        elif search_type == "dense":
            self.nodes = self._collect_dense_nodes(step_size)
        else:
            raise ValueError(f"Unknown search_type: {search_type!r}. Expected 'sparse' or 'dense'.")
        self.patch_shape = [depth, depth, depth]
        self.collate_fn = collate_fn

    def estimate_iterations(self):
        return len(self.nodes)

    def node_segment_id(self, i):
        return self.graph.node_component_id[i]

    def _collect_dense_nodes(self, step_size):
        nodes = []
        for component in rx.connected_components(self.graph):
            leaves = [n for n in component if self.graph.degree(n) == 1]
            if not leaves:
                continue
            root = leaves[0]
            last_node = root
            for i, j in rx.dfs_edges(self.graph, source=root):
                is_branching = self.graph.degree(j) >= 3
                is_next = self.graph.dist(last_node, j) >= step_size - 2
                if is_next or is_branching:
                    last_node = j
                    nodes.append(j)
        return nodes

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, i):
        node = self.nodes[i]
        subgraph = self.graph.rooted_subgraph(node, self.depth)
        _, paths, topo_edge_index = topological_decomposition(subgraph)

        curves = []
        for path in paths:
            xyz = subgraph.node_xyz[path].copy()
            xyz -= xyz[0]
            xyz[1:] -= xyz[:-1].copy()
            if len(xyz) < self.min_curve_len:
                pad = np.zeros((self.min_curve_len - len(xyz), 3), dtype=xyz.dtype)
                xyz = np.concatenate([xyz, pad], axis=0)
            curves.append(xyz)

        edge_index = _build_line_graph_edge_index(topo_edge_index)
        root_curve_indices = [j for j, (src, _) in enumerate(topo_edge_index) if src == 0]
        return node, TreeSample(curves=curves, edge_index=edge_index, root_curve_indices=root_curve_indices)


def collate_fn(batch):
    nodes = torch.tensor([item[0] for item in batch])
    samples = [item[1] for item in batch]
    return nodes, TensorDict({"tree_sample": samples})


class MergeGraphDataset(Dataset):
    """
    Labelled rooted-subgraph dataset for merge-error detection.

    Each item is a (TreeSample, label) pair.  TreeSample contains the
    first-difference curves and line-graph edge_index produced by
    topological_decomposition — the same representation consumed by Arborist.
    Labels are 1 for merge sites and 0 for nonmerge sites.

    Parameters
    ----------
    graph : FragmentsGraph
        Full skeleton graph to extract subgraphs from.
    merge_nodes : list[int]
        Graph node IDs at merge sites (positive examples).
    nonmerge_nodes : list[int]
        Graph node IDs at nonmerge sites (negative examples).
    max_depth : float, optional
        Subgraph extraction radius in microns. Default is 100.
    min_curve_len : int, optional
        Minimum number of points per curve. Curves shorter than this are
        zero-padded so CurveEncoder produces at least one segment token.
        Should match the curve_segment_len used in CurveEncoder (default 10).
    class_ratios : tuple[float, float], optional
        (positive_ratio, negative_ratio) used for rebalancing. Default is
        (0.5, 0.5).
    rebalance_classes : bool, optional
        If True, randomly subsample negatives each epoch to match the target
        ratio. Default is True.
    transform : callable, optional
        Applied to each curve's raw xyz array before differencing, e.g.
        CurveTransforms for augmentation. Default is None.
    """

    def __init__(
        self,
        graph,
        merge_nodes,
        nonmerge_nodes,
        max_depth=1000,
        min_curve_len=10,
        class_ratios=(0.5, 0.5),
        rebalance_classes=True,
        transform=None,
    ):
        super().__init__()
        self.graph = graph
        self.merge_nodes = list(merge_nodes)
        self.nonmerge_nodes = list(nonmerge_nodes)
        self.max_depth = max_depth
        self.min_curve_len = min_curve_len
        self.class_ratios = class_ratios
        self.rebalance_classes = rebalance_classes
        self.transform = transform
        self.config = {
            "max_depth": max_depth,
            "min_curve_len": min_curve_len,
            "class_ratios": list(class_ratios),
            "rebalance_classes": rebalance_classes,
            "transform": type(transform).__name__ if transform else None,
        }
        self._build_index()

    # --- Index ---
    def _build_index(self):
        """Builds (node, label) index, resampling negatives if configured."""
        pos_ratio, neg_ratio = self.class_ratios
        n_pos = len(self.merge_nodes)
        n_neg = len(self.nonmerge_nodes)

        if self.rebalance_classes and n_pos > 0 and n_neg > 0:
            n_target = int(n_pos * neg_ratio / pos_ratio)
            n_sample = min(n_target, n_neg)
            neg_idxs = np.random.choice(n_neg, size=n_sample, replace=False)
            neg_sample = [self.nonmerge_nodes[i] for i in neg_idxs]
        else:
            neg_sample = self.nonmerge_nodes

        self._index = (
            [(node, 1) for node in self.merge_nodes]
            + [(node, 0) for node in neg_sample]
        )

    def rebuild_index(self):
        """Resample negatives — call once at the start of each training epoch."""
        self._build_index()

    # --- Dataset Interface ---
    def __getitem__(self, i):
        """
        Returns one (TreeSample, label) pair.

        Parameters
        ----------
        i : int

        Returns
        -------
        sample : TreeSample
        label : int
            1 for merge, 0 for nonmerge.
        """
        node, label = self._index[i]
        subgraph = self.graph.rooted_subgraph(node, self.max_depth)
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
        root_curve_indices = [i for i, (src, _) in enumerate(topo_edge_index) if src == 0]
        return TreeSample(curves=curves, edge_index=edge_index, root_curve_indices=root_curve_indices), label

    def __len__(self):
        return len(self._index)

    def save_config(self, path):
        """
        Saves dataset parameters to a JSON file.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        write_json(path, self.config)

    def __repr__(self):
        return (
            f"MergeGraphDataset("
            f"n_examples={len(self)}, "
            f"n_merge={len(self.merge_nodes)}, "
            f"n_nonmerge={len(self.nonmerge_nodes)})"
        )


# --- Site Loading ---
def load_merge_sites(graph, sites_prefix):
    """
    Reads merge and nonmerge SWC files and maps them to graph node IDs.

    Expects the directory structure::

        sites_prefix/
            merge_sites/    ← positive examples
            nonmerge_sites/ ← negative examples

    Each SWC file represents one site; the first xyz coordinate in the file
    is snapped to the nearest graph node via the graph's KD-tree.

    Parameters
    ----------
    graph : FragmentsGraph
        Graph whose KD-tree is used to snap site xyz coordinates to nodes.
        Must have been loaded (i.e. graph.kdtree must be set).
    sites_prefix : str
        Cloud (gs://) or local path prefix containing merge_sites/ and
        nonmerge_sites/ subdirectories.

    Returns
    -------
    merge_nodes : list[int]
        Node IDs for merge sites.
    nonmerge_nodes : list[int]
        Node IDs for nonmerge sites.
    """
    reader = Reader(verbose=False)

    def sites_to_nodes(subprefix):
        nodes = []
        for swc_dict in reader(subprefix):
            xyz = swc_dict["xyz"][0]
            _, node = graph.kdtree.query(xyz)
            nodes.append(int(node))
        return nodes

    merge_nodes = sites_to_nodes(os.path.join(sites_prefix, "merge_sites"))
    nonmerge_nodes = sites_to_nodes(os.path.join(sites_prefix, "nonmerge_sites"))
    return merge_nodes, nonmerge_nodes


# --- Collate ---
def collate_merge_samples(batch):
    """
    Collate function for use with DataLoader(collate_fn=collate_merge_samples).

    Parameters
    ----------
    batch : list[tuple[TreeSample, int]]
        Raw items from MergeGraphDataset.__getitem__.

    Returns
    -------
    samples : list[TreeSample]
        One TreeSample per item in the batch.
    labels : torch.Tensor
        Shape (B, 1), dtype float32.
    """
    samples = [item[0] for item in batch]
    labels = torch.tensor(
        [item[1] for item in batch], dtype=torch.float32
    ).unsqueeze(1)
    return samples, labels
