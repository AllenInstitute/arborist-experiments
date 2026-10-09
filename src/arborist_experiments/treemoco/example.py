"""Worked example: from an arborist `SkeletonGraph` to one TreeMoCo embedding per neuron.

The flow every caller of this package follows, in four steps:

1. **Split the graph into cells.** `SkeletonGraph.load` puts every SWC it reads into one graph, one
   connected component per file; `cells_from_graph` splits the components apart and derives the
   parent array the loader does not store.
2. **Collapse each cell to its irreducible tree.** `collapse` keeps the root, branch points and
   tips, and stores each dropped chain of degree-2 nodes as a fixed-length polyline on the node it
   leads to. A `CollapsedTree` is the unit everything else consumes. A cell `collapse` refuses
   (no `node_feats["type"]`, for instance) raises `ValueError` naming the reason; the example
   records it and moves on.
3. **Pretrain.** `pretrain` runs MoCo over the trees -- two augmented views of each tree are a
   positive pair, a queue of earlier keys the negatives -- and returns the query network's state
   dict. It never sees a label. The `AugmentConfig` is the one real modelling choice:
   `upright_preset` assumes the cells share an upright frame (y towards pia) and rotates only about
   that axis, while `paper_preset` PCA-normalizes each tree and rotates freely, as in Chen et al.
   2022.
4. **Embed.** Load the encoder half of the state dict into a `TreeEncoder` and call `embed`, which
   runs every tree un-augmented and returns the root state, one `d_hidden` vector per tree.

For supervised fine-tuning instead of a frozen embedding, wrap the same encoder in
`model.TreeClassifier(encoder, n_classes)` and train it on `batching.collate` batches of
`augment.identity_view(tree, aug)`.

`pretrain` drops the last partial batch, so it needs at least `batch_size` trees; the example
lowers `batch_size` for a small input. `PretrainConfig.device` defaults to CUDA when available.

Run as a script over a directory of SWC files:

    python -m arborist_experiments.treemoco.example /path/to/swcs
"""
from __future__ import annotations

import sys

import numpy as np
from arborist.skeleton_graph import SkeletonGraph

from arborist_experiments.treemoco.augment import upright_preset
from arborist_experiments.treemoco.model import TreeEncoder
from arborist_experiments.treemoco.pretrain import PretrainConfig, embed, pretrain
from arborist_experiments.treemoco.tree_features import (
    CollapsedTree,
    collapse,
    n_features,
)
from arborist_experiments.utils.cells import cells_from_graph, specimen_id


def collapse_all(graph: SkeletonGraph) -> tuple[list[CollapsedTree], list[tuple[str, str]]]:
    """One `CollapsedTree` per neuron in `graph`, plus (specimen id, reason) for each refused."""
    trees, skipped = [], []
    for cell in cells_from_graph(graph):
        try:
            trees.append(collapse(cell))
        except ValueError as e:
            skipped.append((specimen_id(cell), str(e)))
    return trees, skipped


def embeddings(graph: SkeletonGraph, epochs: int = 20
               ) -> tuple[np.ndarray, list[str], list[tuple[str, str]]]:
    """Pretrain TreeMoCo on the neurons in `graph` and embed them.

    Parameters
    ----------
    graph : a `SkeletonGraph`, one connected component per neuron, carrying `node_feats["type"]`.
    epochs : pretraining epochs; `PretrainConfig` defaults to 200. `epochs=0` returns the randomly
        initialized encoder, the control any pretraining gain should be measured against.

    Returns
    -------
    Z : (n_trees, d_hidden) float32, one row per embedded neuron.
    ids : specimen ids of the rows of `Z`, in order.
    skipped : (specimen id, reason) for every cell `collapse` refused.
    """
    trees, skipped = collapse_all(graph)
    aug = upright_preset()
    cfg = PretrainConfig(epochs=epochs, batch_size=min(PretrainConfig.batch_size, len(trees)),
                         num_workers=0)
    state, _history = pretrain(trees, aug, cfg)

    encoder = TreeEncoder(d_in=n_features(aug.include_type), d_hidden=cfg.d_hidden, d_phi=cfg.d_phi)
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items()
                             if k.startswith("encoder.")})
    Z = embed(encoder, trees, aug, cfg)
    return Z, [t.specimen_id for t in trees], skipped


if __name__ == "__main__":
    graph = SkeletonGraph()
    graph.load(sys.argv[1])
    Z, ids, skipped = embeddings(graph)
    print(f"{Z.shape[0]} neurons x {Z.shape[1]}-dim embedding")
    for sid, reason in skipped:
        print(f"skipped {sid}: {reason}")
