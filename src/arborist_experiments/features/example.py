"""Worked example: from an arborist `SkeletonGraph` to a cells x features matrix.

The flow every caller of this package follows, in four steps:

1. **Split the graph into cells.** `SkeletonGraph.load` puts every SWC it reads into one graph, one
   connected component per file. The extractors take one graph per neuron, so `cells_from_graph`
   splits the components apart, derives the parent array the loader does not store, and widens the
   loader's float16/float32 storage to float64.
2. **Extract feature blocks.** Each `extract_*` takes the list of cells plus a `modality` (`"full"`,
   `"axon"`, `"dendrite"`, `"basal"`, `"apical"`, selected by SWC type code) and returns one
   `FeatureResult` per cell, in order. A cell an extractor cannot use (no soma, no nodes of the
   modality) is not dropped: its result carries the reason in `skipped`, so positions stay aligned.
   Every constant that differs between datasets -- bin edges, Sholl radii -- is an explicit
   argument; the values below are illustrative, not tuned.
3. **Combine the blocks.** `concat_features` joins blocks cell by cell (a cell skipped by any block
   is skipped in the result), and `drop_zero_variance_and_impute` drops columns that are constant
   across the batch and fills NaNs with column means. Both are batch-level operations, so run them
   on the cohort you will fit on, not per cell.
4. **Build the matrix.** `design_matrix` stacks the surviving rows and returns the ids of the cells
   they came from; `skips` lists the ones that did not survive and why.

What the graph must satisfy -- checked per cell by `utils.cells.check`, which skips a cell rather
than raising:

- `node_feats["type"]` (SWC compartment codes) on every extractor, and `node_feats["radius"]` on
  all but the radial ones.
- A soma node (type 1) for morphometrics, Sholl and upright morphometrics, which measure outwards
  from it.
- **Soma-centered coordinates** for `extract_radial_hist` and `extract_radial_sholl`, which read
  `node_xyz` as a displacement from the soma: the soma centroid must sit at the origin.

`extract_upright_morphometrics` needs `neuron-morphology`, which `pyproject.toml` does not install;
it is added only when that package is importable. Every other extractor here has no such
requirement.

Run as a script over a directory of SWC files:

    python -m arborist_experiments.features.example /path/to/swcs
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
from arborist.skeleton_graph import SkeletonGraph

from arborist_experiments.features import (
    FeatureResult,
    concat_features,
    design_matrix,
    drop_zero_variance_and_impute,
    extract_morphometrics,
    extract_radial_hist,
    extract_radial_sholl,
    extract_sholl,
    skips,
)
from arborist_experiments.utils.cells import cells_from_graph, specimen_id

# Illustrative constants, in microns: 12 in-plane radial bins and 12 soma-relative depth bins for the
# radial histogram, and Sholl shells every 20 um out to 300 um.
R_EDGES = np.linspace(0.0, 300.0, 13)
Z_EDGES = np.linspace(-300.0, 300.0, 13)
SHOLL_RADII = list(range(10, 310, 20))


def feature_blocks(cells: list[SkeletonGraph], modality: str) -> list[list[FeatureResult]]:
    """One list of per-cell `FeatureResult`s per feature family, all aligned with `cells`."""
    blocks = [
        extract_morphometrics(cells, modality),
        extract_sholl(cells, modality, sholl_radii=SHOLL_RADII),
        extract_radial_hist(cells, modality, r_edges=R_EDGES, z_edges=Z_EDGES),
        extract_radial_sholl(cells, modality),
    ]
    if importlib.util.find_spec("neuron_morphology") is not None:
        from arborist_experiments.features import extract_upright_morphometrics

        blocks.append(extract_upright_morphometrics(cells))   # whole cell; takes no modality
    return blocks


def feature_matrix(graph: SkeletonGraph, modality: str = "dendrite"
                   ) -> tuple[np.ndarray, list[str], list[str], list[tuple[str, str]]]:
    """The cells x features matrix for every neuron in `graph`.

    Parameters
    ----------
    graph : a `SkeletonGraph`, one connected component per neuron, carrying `node_feats["type"]`
        and `node_feats["radius"]` -- what `SkeletonGraph.load` produces from SWC files.
    modality : which compartments to measure.

    Returns
    -------
    X : (n_kept, n_features) float64.
    names : the column names of `X`.
    kept : specimen ids of the rows of `X`, in order.
    skipped : (specimen id, reason) for every cell that has no row.
    """
    cells = cells_from_graph(graph)
    ids = [specimen_id(c) for c in cells]
    combined = drop_zero_variance_and_impute(concat_features(*feature_blocks(cells, modality)))
    X, kept = design_matrix(combined, ids)
    names = next((r.feature_names for r in combined if r.skipped is None), [])
    return X, names, kept, skips(combined, ids)


if __name__ == "__main__":
    graph = SkeletonGraph()
    graph.load(sys.argv[1])
    X, names, kept, skipped = feature_matrix(graph)
    print(f"{X.shape[0]} cells x {X.shape[1]} features; first columns {names[:5]}")
    for sid, reason in skipped:
        print(f"skipped {sid}: {reason}")
