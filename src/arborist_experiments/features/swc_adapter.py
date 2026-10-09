import numpy as np
import pandas as pd
from morphopy.neurontree.NeuronTree import NeuronTree

from arborist.skeleton_graph import SkeletonGraph


def cell_to_swc_df(cell: SkeletonGraph) -> pd.DataFrame:
    feats, xyz = cell.node_feats, cell.node_xyz
    parents = feats["parent"]
    n = len(parents)
    parent_1based = np.where(parents == -1, -1, parents + 1)
    return pd.DataFrame({
        "n":      np.arange(1, n + 1, dtype=np.int64),
        "type":   feats["type"].astype(np.int64),
        "x":      xyz[:, 0],
        "y":      xyz[:, 1],
        "z":      xyz[:, 2],
        "radius": feats["radius"],
        "parent": parent_1based.astype(np.int64),
    })


def cell_to_neurontree(cell: SkeletonGraph) -> NeuronTree:
    return NeuronTree(swc=cell_to_swc_df(cell))
