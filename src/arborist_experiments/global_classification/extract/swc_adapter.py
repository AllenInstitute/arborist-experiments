import numpy as np
import pandas as pd

from morphopy.neurontree.NeuronTree import NeuronTree

from arborist_experiments.utils.cells import CellGraph


def cell_to_swc_df(cell: CellGraph) -> pd.DataFrame:
    n = len(cell.types)
    parent_1based = np.where(cell.parents == -1, -1, cell.parents + 1)
    return pd.DataFrame({
        "n":      np.arange(1, n + 1, dtype=np.int64),
        "type":   cell.types.astype(np.int64),
        "x":      cell.xyz[:, 0],
        "y":      cell.xyz[:, 1],
        "z":      cell.xyz[:, 2],
        "radius": cell.radius,
        "parent": parent_1based.astype(np.int64),
    })


def cell_to_neurontree(cell: CellGraph) -> NeuronTree:
    return NeuronTree(swc=cell_to_swc_df(cell))
