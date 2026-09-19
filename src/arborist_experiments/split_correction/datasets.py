"""
Created on Fri Sep 19 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Dataset utilities for training a split-correction model on graph structure
only. Each example is a proposal from a neuron_proofreader ProposalGraph,
rendered as an Arborist TreeSample rooted at the proposal center.

"""

import numpy as np
import torch
from torch.utils.data import Dataset

from arborist.utils.util import write_json
from neuron_proofreader.utils.ml_util import TensorDict


class ProposalTreeDataset(Dataset):
    """
    Labelled dataset over the proposals of a single ProposalGraph.

    Each item is a (TreeSample, label) pair. Labels are 1 when the proposal
    is in ``graph.gt_accepts`` and 0 otherwise.

    Parameters
    ----------
    graph : ProposalGraph
        Graph with proposals generated and ``gt_accepts`` populated.
    tree_extractor : ProposalTreeFeatureExtractor
        Extraction settings for the rooted subgraphs.
    proposals : Iterable[Frozenset[int]], optional
        Proposals to include. Default is ``graph.list_proposals()``.
    class_ratios : Tuple[float, float], optional
        (positive_ratio, negative_ratio) used for rebalancing. Default is
        (0.5, 0.5).
    rebalance_classes : bool, optional
        If True, negatives are randomly subsampled each epoch to match the
        target ratio. Default is True.
    """

    def __init__(
        self,
        graph,
        tree_extractor,
        proposals=None,
        class_ratios=(0.5, 0.5),
        rebalance_classes=True,
    ):
        super().__init__()
        self.graph = graph
        self.tree_extractor = tree_extractor
        self.class_ratios = class_ratios
        self.rebalance_classes = rebalance_classes

        proposals = list(proposals) if proposals is not None else graph.list_proposals()
        gt_accepts = getattr(graph, "gt_accepts", set())
        self.positives = [p for p in proposals if p in gt_accepts]
        self.negatives = [p for p in proposals if p not in gt_accepts]

        self.config = {
            **self.tree_extractor.config,
            "class_ratios": list(class_ratios),
            "rebalance_classes": rebalance_classes,
        }
        self._build_index()

    # --- Index ---
    def _build_index(self):
        """Builds the (proposal, label) index, resampling negatives if configured."""
        pos_ratio, neg_ratio = self.class_ratios
        n_pos, n_neg = len(self.positives), len(self.negatives)
        if self.rebalance_classes and n_pos > 0 and n_neg > 0:
            n_target = int(n_pos * neg_ratio / pos_ratio)
            n_sample = min(n_target, n_neg)
            neg_idxs = np.random.choice(n_neg, size=n_sample, replace=False)
            negatives = [self.negatives[k] for k in neg_idxs]
        else:
            negatives = self.negatives
        self._index = [(p, 1) for p in self.positives] + [
            (p, 0) for p in negatives
        ]

    def rebuild_index(self):
        """Resamples negatives. Call once at the start of each epoch."""
        self._build_index()

    # --- Dataset Interface ---
    def __getitem__(self, i):
        proposal, label = self._index[i]
        return self.tree_extractor(self.graph, proposal), label

    def __len__(self):
        return len(self._index)

    def n_proposals(self):
        return len(self.positives) + len(self.negatives)

    def save_config(self, path):
        write_json(path, self.config)

    def __repr__(self):
        return (
            f"ProposalTreeDataset("
            f"n_examples={len(self)}, "
            f"n_accepts={len(self.positives)}, "
            f"n_rejects={len(self.negatives)})"
        )


class ProposalSearchDataset(Dataset):
    """
    Unlabelled dataset over the proposals of a ProposalGraph, for inference.

    Each item is a (proposal_index, TreeSample) pair so the caller can map
    predictions back to proposals via ``dataset.proposals``.

    Parameters
    ----------
    graph : ProposalGraph
        Graph with proposals generated.
    tree_extractor : ProposalTreeFeatureExtractor
        Extraction settings for the rooted subgraphs.
    """

    def __init__(self, graph, tree_extractor):
        super().__init__()
        self.graph = graph
        self.tree_extractor = tree_extractor
        self.proposals = graph.list_proposals()
        self.collate_fn = collate_search_samples

    def __getitem__(self, i):
        return i, self.tree_extractor(self.graph, self.proposals[i])

    def __len__(self):
        return len(self.proposals)


# --- Collate ---
def collate_proposal_samples(batch):
    """
    Collate function for ProposalTreeDataset.

    Returns
    -------
    samples : List[TreeSample]
    labels : torch.Tensor
        Shape (B, 1), dtype float32.
    """
    samples = [item[0] for item in batch]
    labels = torch.tensor(
        [item[1] for item in batch], dtype=torch.float32
    ).unsqueeze(1)
    return samples, labels


def collate_search_samples(batch):
    """
    Collate function for ProposalSearchDataset.

    Returns
    -------
    idxs : torch.Tensor
        Proposal indices, shape (B,).
    x : TensorDict
        Holds the list of TreeSamples under "tree_sample".
    """
    idxs = torch.tensor([item[0] for item in batch])
    samples = [item[1] for item in batch]
    return idxs, TensorDict({"tree_sample": samples})
