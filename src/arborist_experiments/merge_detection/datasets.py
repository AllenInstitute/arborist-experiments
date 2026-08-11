"""
Created on Sun August 10 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Dataset utilities for training a merge-detection model on graph structure only.

Architecture
------------
MergeGraphDataset
    Flat dataset of (TreeSample, label) pairs over a single FragmentsGraph.
    Each item extracts a rooted subgraph centred on a merge or nonmerge site,
    decomposes it via topological_decomposition, and returns a TreeSample with
    first-difference curves and line-graph edge_index — the same format
    consumed by Arborist.

load_merge_sites
    Helper that reads SWC files from a cloud or local sites prefix and maps
    each site's first xyz coordinate to the nearest graph node.

collate_merge_samples
    Custom DataLoader collate function that handles the variable-structure
    TreeSample objects.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from arborist.data.datasets import TreeSample, _build_line_graph_edge_index
from arborist.utils.graph_utils import topological_decomposition
from arborist.utils.swc_loading import Reader
from arborist.utils.util import write_json


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
        Should match the segment_len used in CurveEncoder (default 10).
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
        return TreeSample(curves=curves, edge_index=edge_index), label

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
