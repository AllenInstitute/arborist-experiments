"""
Created on Fri Sep 19 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Hooks that attach proposal-rooted TreeSamples to the split-correction
feature pipeline so ArboristVisionHGAT receives them alongside the
geometric features and image patches.

The classes here are thin subclasses of the neuron_proofreader split
pipeline. Each batch's TreeSamples are ordered by proposal index, matching
the row order of the proposal feature matrix and image patches.

"""

import numpy as np

from neuron_proofreader.split_proofreading.split_datamodules import (
    FragmentsDataset,
    FragmentsLoader,
)
from neuron_proofreader.split_proofreading.split_feature_extraction import (
    FeaturePipeline,
    HeteroGraphData,
)

from arborist_experiments.split_correction.graph_extraction import (
    proposal_tree_sample,
)


class ProposalTreeFeatureExtractor:
    """
    Builds one TreeSample per proposal of a computation subgraph, and one
    curve per branch, so that neither node type needs hand-crafted features.

    Parameters
    ----------
    graph : ProposalGraph
        Full fragments graph, used for skeleton traversal.
    max_depth : float, optional
        Subgraph extraction radius in microns from the proposal midpoint.
        Default is 100.
    min_curve_len : int, optional
        Minimum number of points per curve; shorter curves are zero-padded.
        Should match the curve_segment_len used in CurveEncoder. Default is
        10.
    node_spacing : float, optional
        Spacing of bridge nodes. Default is ``graph.node_spacing``.
    bridge_mode : {"linear", "spline"}, optional
        Bridge geometry, see ``proposal_rooted_subgraph``. Default is
        "linear".
    spline_context : float, optional
        Fragment tail length used for the spline fit.
    transform : callable, optional
        Applied to each curve's raw xyz array before differencing.
    graph_transform : callable, optional
        Applied to the subgraph's node_xyz array before decomposition.
    """

    def __init__(
        self,
        graph,
        max_depth=100,
        min_curve_len=10,
        node_spacing=None,
        bridge_mode="linear",
        spline_context=None,
        transform=None,
        graph_transform=None,
    ):
        self.graph = graph
        self.max_depth = max_depth
        self.min_curve_len = min_curve_len
        self.node_spacing = node_spacing
        self.bridge_mode = bridge_mode
        self.spline_context = spline_context
        self.transform = transform
        self.graph_transform = graph_transform

    def __call__(self, features):
        """
        Attaches ``features.tree_samples`` in proposal-index order and
        ``features.branch_curves`` in branch-index order.

        Parameters
        ----------
        features : FeatureSet
            Feature set whose ``proposal_index_mapping`` and
            ``edge_index_mapping`` fix the orders. ``features.graph`` must be
            the ProposalComputationGraph, whose ``edge_to_path`` gives the
            skeleton path behind each branch.
        """
        features.branch_curves = self.extract_branch_curves(features)
        idx_to_id = features.proposal_index_mapping.idx_to_id
        features.tree_samples = [
            proposal_tree_sample(
                self.graph,
                idx_to_id[idx],
                self.max_depth,
                min_curve_len=self.min_curve_len,
                node_spacing=self.node_spacing,
                bridge_mode=self.bridge_mode,
                spline_context=self.spline_context,
                transform=self.transform,
                graph_transform=self.graph_transform,
            )
            for idx in range(len(idx_to_id))
        ]
        return features

    def extract_branch_curves(self, features):
        """
        Converts each branch (irreducible path between computation nodes)
        into a first-difference curve in the CurveEncoder convention.

        Returns
        -------
        List[numpy.ndarray]
            One (N_k, 3) array per branch, in branch-index order, zero-padded
            to at least ``min_curve_len`` rows. Empty if the feature set has
            no ``edge_index_mapping`` or its graph has no ``edge_to_path``.
        """
        mapping = getattr(features, "edge_index_mapping", None)
        edge_to_path = getattr(getattr(features, "graph", None), "edge_to_path", None)
        if mapping is None or edge_to_path is None:
            return []
        curves = []
        for idx in range(len(mapping.idx_to_id)):
            edge = mapping.idx_to_id[idx]
            path = np.asarray(edge_to_path[edge], dtype=int)
            xyz = np.asarray(self.graph.node_xyz[path], dtype=np.float32).copy()
            if self.graph_transform:
                xyz = self.graph_transform(xyz)
            if self.transform:
                xyz = self.transform(xyz)
            xyz -= xyz[0]
            xyz[1:] -= xyz[:-1].copy()
            if len(xyz) < self.min_curve_len:
                pad = np.zeros((self.min_curve_len - len(xyz), 3), dtype=xyz.dtype)
                xyz = np.concatenate([xyz, pad], axis=0)
            curves.append(xyz)
        return curves


class ArboristFeaturePipeline(FeaturePipeline):
    """
    FeaturePipeline that additionally produces TreeSamples per proposal.

    Parameters
    ----------
    graph : ProposalGraph
        Graph to extract features from.
    img_path : str
        Path to image of whole-brain dataset.
    tree_kwargs : dict, optional
        Keyword arguments for ProposalTreeFeatureExtractor (max_depth,
        min_curve_len, node_spacing, transform, graph_transform).
    **kwargs
        Forwarded to FeaturePipeline.
    """

    def __init__(self, graph, img_path, tree_kwargs=None, **kwargs):
        super().__init__(graph, img_path, **kwargs)
        self.tree_extractor = ProposalTreeFeatureExtractor(
            graph, **(tree_kwargs or {})
        )

    def __call__(self, subgraph):
        features = super().__call__(subgraph)
        return self.tree_extractor(features)


class ArboristHeteroGraphData(HeteroGraphData):
    """
    HeteroGraphData that carries TreeSamples and branch curves through to
    the model inputs under the "tree_samples" and "branch_curves" keys.
    """

    def __init__(self, features):
        super().__init__(features)
        self.tree_samples = getattr(features, "tree_samples", None)
        self.branch_curves = getattr(features, "branch_curves", None)

    def get_inputs(self):
        inputs = super().get_inputs()
        inputs["tree_samples"] = self.tree_samples
        inputs["branch_curves"] = self.branch_curves
        return inputs


class ArboristFragmentsDataset(FragmentsDataset):
    """
    FragmentsDataset whose feature extractor is an ArboristFeaturePipeline.

    Parameters
    ----------
    fragments_graph : ProposalGraph
        Graph with proposals generated.
    img_config : ImageConfig
        Image configuration (img_path, brightness_clip, patch_shape).
    tree_kwargs : dict, optional
        Keyword arguments for ProposalTreeFeatureExtractor.
    """

    def __init__(self, fragments_graph, img_config, tree_kwargs=None):
        self.graph = fragments_graph
        self.feature_extractor = ArboristFeaturePipeline(
            self.graph,
            img_config.img_path,
            tree_kwargs=tree_kwargs,
            brightness_clip=img_config.brightness_clip,
            patch_shape=img_config.patch_shape,
        )
        self.sampler = None

    def __iter__(self):
        for subgraph in iter(self.sampler):
            yield ArboristHeteroGraphData(self.feature_extractor(subgraph))


class ArboristFragmentsLoader(FragmentsLoader):
    """
    FragmentsLoader that emits inputs containing "tree_samples".
    """

    def _fetch_batch(self, key, sampler):
        try:
            subgraph = next(sampler)
        except StopIteration:
            return None
        dataset = self.collection[key]
        features = dataset.feature_extractor(subgraph)
        data = ArboristHeteroGraphData(features)
        return data.get_inputs(), data.get_targets()
