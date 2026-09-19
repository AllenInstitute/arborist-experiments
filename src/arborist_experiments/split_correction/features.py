"""
Created on Fri Sep 19 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Hooks that attach proposal-rooted TreeSamples to the split-correction data
pipeline so ArboristVisionHGAT receives them alongside the image patches.
Each batch's TreeSamples are ordered by proposal index, matching the row
order of the proposal feature matrix and image patches.

"""

from neuron_proofreader.split_proofreading.split_datamodules import (
    FragmentsDataset,
    FragmentsLoader,
)
from neuron_proofreader.split_proofreading.split_feature_extraction import (
    HeteroGraphData,
)


class ArboristHeteroGraphData(HeteroGraphData):
    """
    HeteroGraphData that carries TreeSamples through to the model inputs
    under the "tree_samples" key.
    """

    def __init__(self, features):
        super().__init__(features)
        self.tree_samples = getattr(features, "tree_samples", None)

    def get_inputs(self):
        inputs = super().get_inputs()
        inputs["tree_samples"] = self.tree_samples
        return inputs


class ArboristFragmentsDataset(FragmentsDataset):
    """
    FragmentsDataset whose feature extraction also attaches one TreeSample
    per proposal.

    Parameters
    ----------
    fragments_graph : ProposalGraph
        Graph with proposals generated.
    img_config : ImageConfig
        Image configuration.
    tree_extractor : ProposalTreeFeatureExtractor
        Extraction settings for the rooted subgraphs.
    """

    def __init__(self, fragments_graph, img_config, tree_extractor):
        # Call parent class
        super().__init__(fragments_graph, img_config)

        # Instance attributes
        self.tree_extractor = tree_extractor

    def extract_features(self, subgraph):
        """
        Runs the standard feature pipeline, then attaches TreeSamples in
        proposal-index order.

        Parameters
        ----------
        subgraph : ProposalComputationGraph
            Subgraph to extract features for.

        Returns
        -------
        FeatureSet
            Standard features plus ``tree_samples``.
        """
        features = self.feature_extractor(subgraph)
        idx_to_id = features.proposal_index_mapping.idx_to_id
        features.tree_samples = [
            self.tree_extractor(self.graph, idx_to_id[idx])
            for idx in range(len(idx_to_id))
        ]
        return features

    def __iter__(self):
        for subgraph in iter(self.sampler):
            yield ArboristHeteroGraphData(self.extract_features(subgraph))


class ArboristFragmentsLoader(FragmentsLoader):
    """
    FragmentsLoader that calls ``ArboristFragmentsDataset.extract_features``
    so emitted inputs contain "tree_samples".
    """

    def _fetch_batch(self, key, sampler):
        try:
            subgraph = next(sampler)
        except StopIteration:
            return None
        dataset = self.collection[key]
        features = dataset.extract_features(subgraph)
        data = ArboristHeteroGraphData(features)
        return data.get_inputs(), data.get_targets()
