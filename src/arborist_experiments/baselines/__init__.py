"""
Baseline methods benchmarked against Arborist. Each method is an encoder
that maps a rooted subgraph (arborist TreeSample) to an embedding, one
module per method, so the task heads in ``tasks`` can wrap any of them.
"""

from arborist_experiments.baselines.tmd import (
    TMDDescriptor,
    persistence_barcode,
    persistence_image,
    tree_distances,
)
