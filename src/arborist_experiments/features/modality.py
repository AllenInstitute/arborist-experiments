"""Which SWC compartment types each modality keeps.

Which modalities are *meaningful* on a given dataset is a separate question that `filter_cell`
cannot answer: it depends on which compartments the dataset reconstructs.
"""


from arborist.skeleton_graph import SkeletonGraph

from arborist_experiments.utils.cells import (
    TYPE_APICAL,
    TYPE_AXON,
    TYPE_BASAL,
    TYPE_SOMA,
    filtered,
)

MODALITY_TYPES = {
    "full":     {TYPE_SOMA, TYPE_AXON, TYPE_BASAL, TYPE_APICAL},
    "axon":     {TYPE_SOMA, TYPE_AXON},
    "dendrite": {TYPE_SOMA, TYPE_BASAL, TYPE_APICAL},
    "basal":    {TYPE_SOMA, TYPE_BASAL},
    "apical":   {TYPE_SOMA, TYPE_APICAL},
}


def filter_cell(cell: SkeletonGraph, modality: str) -> SkeletonGraph | None:
    if modality not in MODALITY_TYPES:
        raise ValueError(f"Unknown modality {modality!r}")
    return filtered(cell, MODALITY_TYPES[modality])
