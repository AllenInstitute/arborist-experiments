"""A labelled collection of cells, shared by both benchmark packages."""

from dataclasses import dataclass, field

import numpy as np

from .cell_graph import CellGraph


@dataclass
class Cohort:
    cells: list[CellGraph]
    labels: np.ndarray                 # (n_cells,) int label index
    label_names: list[str]             # class name per label index
    subclass_by_id: dict[str, str] = field(default_factory=dict)

    def __len__(self):
        return len(self.cells)
