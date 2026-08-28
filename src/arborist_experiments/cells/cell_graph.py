"""A single cell as a SkeletonGraph.

Replaces the `Cell` dataclass that used to live in
`patchseq_morphology_benchmark.data_loading`. Both benchmark packages consume it,
and every module they would otherwise share was coupled to that dataclass, so the
type had to move somewhere neutral before anything else could.

`SkeletonGraph` can carry the extra per-node metadata a cell needs because its
`NODE_ATTR_SPEC` registry lets a subclass declare attributes by extending a dict
rather than allocating its own arrays. It became affordable to use one graph per
cell only after the graph stopped subclassing `nx.Graph`: nx costs ~554 B/node,
which puts the MICrONS excitatory cohort (49,803 cells, 94.2M nodes) at ~52 GB on
a 62 GB machine. CSR-backed it is ~42 B/node.
"""

from typing import Optional

import numpy as np

from arborist.skeleton_graph import SkeletonGraph

TYPE_SOMA, TYPE_AXON, TYPE_BASAL, TYPE_APICAL = 1, 2, 3, 4


class CellGraph(SkeletonGraph):
    """One cell: a rooted tree with per-node type, radius and provenance."""

    NODE_ATTR_SPEC = {
        **SkeletonGraph.NODE_ATTR_SPEC,
        # 0-based parent index, -1 at a root. Kept as a stored array rather
        # than derived from adjacency because the hot feature primitive
        # vectorises over it -- `cell.xyz[cell.parents[child]]` in
        # radial_arbor_features._resample_cable -- and materializing it from a
        # BFS dominates the vectorised work it feeds.
        "parent": (np.int64, ()),
        # The node's parent link was invented, not traced. See
        # data_loading._reparent_to_soma: these are straight soma-to-fragment
        # lines making up ~32% of total edge length across the Patch-seq
        # cohort, so every length-weighted feature excludes them.
        "synthetic_edge": (bool, ()),
        # Dtype overrides, not new attributes. The registry defaults are xyz
        # float32 and radius float16; Patch-seq `swc_upright` is float64 at
        # source and every feature consumes float64, so narrowing storage moves
        # the published rows in their last digits.
        "xyz": (np.float64, (3,)),
        "radius": (np.float64, ()),
    }

    @classmethod
    def from_arrays(
        cls,
        specimen_id: str,
        types: np.ndarray,
        parents: np.ndarray,
        xyz: np.ndarray,
        radius: np.ndarray,
        synthetic_edge: np.ndarray,
    ) -> "CellGraph":
        """Builds a cell from the six arrays the SWC loaders produce."""
        n = len(types)
        child = np.flatnonzero(parents != -1)
        edges = np.stack([parents[child], child], axis=1)

        graph = cls(verbose=False)
        graph.build_from_edges(n, edges)
        graph.init_node_attrs(n)
        graph.node_type[:] = types
        graph.node_parent[:] = parents
        graph.node_xyz[:] = xyz
        graph.node_radius[:] = radius
        graph.node_synthetic_edge[:] = synthetic_edge
        graph.component_id_to_swc_id[0] = specimen_id
        return graph

    # --- Cell field aliases ---
    # The dataclass this replaces exposed these six names and ~170 accesses
    # across both benchmarks read them, so they stay as read-only views onto
    # the registry arrays rather than becoming copies.
    @property
    def specimen_id(self) -> str:
        return self.component_id_to_swc_id[0]

    @property
    def types(self) -> np.ndarray:
        return self.node_type

    @property
    def parents(self) -> np.ndarray:
        return self.node_parent

    @property
    def xyz(self) -> np.ndarray:
        return self.node_xyz

    @property
    def radius(self) -> np.ndarray:
        return self.node_radius

    @property
    def synthetic_edge(self) -> np.ndarray:
        return self.node_synthetic_edge

    # --- Node Attributes ---
    def reindex_node_attrs(self, old_node_ids: np.ndarray) -> None:
        """
        Reorders node attributes to match a new node ID ordering.

        The base implementation permutes attribute rows, which is correct for
        every attribute whose value does not name a node. "parent" does, so its
        values have to be remapped through the same permutation as well as
        moved, with -1 where the parent itself was dropped.
        """
        old_parent = self.node_parent
        old_to_new = -np.ones(len(old_parent), dtype=np.int64)
        old_to_new[old_node_ids] = np.arange(len(old_node_ids))

        super().reindex_node_attrs(old_node_ids)

        gathered = old_parent[old_node_ids]
        self.node_parent = np.where(gathered < 0, -1, old_to_new[gathered])

    # --- Filtering ---
    def filtered(self, keep_types) -> Optional["CellGraph"]:
        """
        Returns a cell holding only the nodes whose type is in "keep_types",
        with each kept node reparented to its nearest kept ancestor.

        Generic node removal cannot do this. It would orphan every node whose
        parent was dropped rather than reconnecting it, and it would lose the
        fact that a rewired link now spans an invented edge, which is why
        "synthetic_edge" is OR-ed along the skipped chain.

        Returns None when too little survives to be a usable arbor: fewer than
        3 nodes, or nothing but soma. Returns self when nothing is dropped,
        which is 99.6% of the MICrONS cells for the default modality.
        """
        types, parents = self.types, self.parents
        n = len(types)

        keep_mask = np.isin(types, list(keep_types))
        if keep_mask.all():
            return self
        if keep_mask.sum() < 3:
            return None
        if not (keep_mask & (types != TYPE_SOMA)).any():
            return None

        old_to_new = -np.ones(n, dtype=np.int64)
        new_indices = np.flatnonzero(keep_mask)
        old_to_new[new_indices] = np.arange(len(new_indices))

        synthetic_edge = self.synthetic_edge
        new_parents = np.full(len(new_indices), -1, dtype=np.int64)
        new_synthetic = np.zeros(len(new_indices), dtype=bool)
        for new_i, old_i in enumerate(new_indices):
            p = int(parents[old_i])
            syn = bool(synthetic_edge[old_i])
            while p != -1 and not keep_mask[p]:
                syn = syn or bool(synthetic_edge[p])
                p = int(parents[p])
            new_parents[new_i] = -1 if p == -1 else int(old_to_new[p])
            new_synthetic[new_i] = syn

        return type(self).from_arrays(
            specimen_id=self.specimen_id,
            types=types[keep_mask].copy(),
            parents=new_parents,
            xyz=self.xyz[keep_mask].copy(),
            radius=self.radius[keep_mask].copy(),
            synthetic_edge=new_synthetic,
        )

    def __repr__(self) -> str:
        return (
            f"CellGraph(specimen_id={self.specimen_id!r},"
            f" n_nodes={self.number_of_nodes()})"
        )
