"""A single cell as an arborist `SkeletonGraph`.

There is no subclass: a cell is a plain `SkeletonGraph` whose per-node arrays live in its
`node_feats` dict, which a caller extends by assigning to it.
"""

from typing import Optional

import numpy as np
import rustworkx as rx
from arborist.skeleton_graph import SkeletonGraph

TYPE_SOMA, TYPE_AXON, TYPE_BASAL, TYPE_APICAL = 1, 2, 3, 4

# Nothing but an annotation alias -- a cell *is* a plain SkeletonGraph.
Cell = SkeletonGraph

# What `make_cell` produces, and therefore what a cell that needs no normalization looks like. Read
# by `as_cell` to decide whether it can hand the input straight back.
_CELL_DTYPES = {
    "type": np.uint8,
    "radius": np.float64,
    "parent": np.int64,
    "synthetic_edge": np.bool_,
}


def specimen_id(cell: SkeletonGraph) -> str:
    """The cell's own id.

    arborist keys ids by connected component because it loads whole fragment graphs; a cell is one
    reconstruction, so its id is component 0's. Worth a name because `feature_cache.cohort_key`
    digests these into the cache directory name -- read the wrong field there and a cached design
    matrix silently mismatches the cohort it is served to.
    """
    return cell.component_id_to_swc_id[0]


def graph_label(graph: SkeletonGraph, i: int) -> str:
    """This graph's id, or its position when it has none.

    `specimen_id` reads component 0, and a graph keyed per connected component is exactly the case
    that has to be reported, so the position is the fallback name.
    """
    try:
        return specimen_id(graph)
    except Exception:
        return f"<graph {i}>"


def graph_labels(graphs) -> list[str]:
    """Ids for a batch, in input order.

    The extractors return one result per input graph and no ids, so this is how a caller recovers
    them from the list it passed. Computed in the parent rather than in a worker, which changes no
    id: `as_cell` either hands its argument back or rebuilds through `make_cell(specimen_id(graph),
    ...)`. That matters because `feature_cache.cohort_key` digests these into a directory name.
    """
    return [graph_label(g, i) for i, g in enumerate(graphs)]


def make_cell(
    specimen_id: str,
    types: np.ndarray,
    parents: np.ndarray,
    xyz: np.ndarray,
    radius: np.ndarray,
    synthetic_edge: np.ndarray,
) -> SkeletonGraph:
    """Builds a cell from the six arrays the SWC loaders produce.

    The dtypes are not arborist's. `SkeletonGraph.load` allocates node_xyz float32 and
    node_feats["radius"] float16, but source coordinates are typically float64 and every feature
    consumes float64, so narrowing storage would move feature values in their last digits.
    Building the graph here rather than through `load` is what makes the choice ours to make.
    """
    n = len(types)
    child = np.flatnonzero(parents != -1)

    graph = SkeletonGraph(verbose=False)
    graph.add_nodes_from([None] * n)
    graph.add_edges_from_no_data([(int(p), int(c)) for p, c in zip(parents[child], child)])
    assert graph.num_nodes() == n, "an edge named a node outside range(n)"

    graph.node_xyz = np.asarray(xyz, dtype=np.float64)
    graph.node_component_id = np.zeros(n, dtype=int)
    graph.node_feats = {
        "type": np.asarray(types, dtype=np.uint8),
        "radius": np.asarray(radius, dtype=np.float64),
        # 0-based parent index, -1 at a root. Stored rather than derived from adjacency because
        # the hot feature primitive vectorises over it -- `cell.node_xyz[parents[child]]` in
        # radial_arbor_features._resample_cable -- and materializing it from a BFS dominates the
        # vectorised work it feeds.
        "parent": np.asarray(parents, dtype=np.int64),
        # The node's parent link was invented, not traced. See reparent_to_soma below:
        # these are straight soma-to-fragment lines making up ~32% of total edge length across
        # the Patch-seq cohort, so every length-weighted feature excludes them.
        "synthetic_edge": np.asarray(synthetic_edge, dtype=bool),
    }
    graph.component_id_to_swc_id[0] = specimen_id
    return graph


def reparent_to_soma(types: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reparent every orphan (parent == -1) that is not the soma to the soma node.

    Patch-seq SWC HDF5 stores many cells as several disconnected subtrees rooted at
    parent = -1. morphopy assumes a single connected tree; without this the
    topological-minor traversal walks off the graph.

    Returns (parents, synthetic) where synthetic marks the invented links. They are not
    traced cable and are long: 42% of cells are fragmented, median 74 fragments each,
    median soma-to-fragment distance 96 um against a real edge length of 1.15 um. Any
    length-weighted feature must exclude them or they contribute ~32% of total edge
    length as straight lines nobody traced. That measured cost is why `as_cell` will not do
    this unless asked -- inventing soma-to-fragment cable is a decision, not a default.
    """
    synthetic = np.zeros(len(parents), dtype=bool)
    soma_idx = np.flatnonzero(types == TYPE_SOMA)
    if soma_idx.size == 0:
        return parents, synthetic
    soma = int(soma_idx[0])
    new_parents = parents.copy()
    orphan_mask = (new_parents == -1)
    orphan_mask[soma] = False
    new_parents[orphan_mask] = soma
    return new_parents, orphan_mask


def parents_from_edges(graph: SkeletonGraph, types: np.ndarray) -> np.ndarray:
    """The 0-based parent index of every node, read off the edge set. -1 at a root.

    `make_cell` stores `parent` because the hot feature primitive vectorises over it, but the edge
    set already encodes the tree, so a graph that arrives without it is not missing data -- this is
    the recovery. One `dfs_edges` pass per component, the same trick arborist's own SWC writer uses
    (`component_to_zipped_swc`).

    Rooting: a soma node if the component has one, else its lowest node index. Deterministic, and
    there is no stronger convention to honour -- arborist's writer roots at an arbitrary component
    node. Isolated nodes and unreachable ones keep -1, which is already what `_real_edges` and
    `_resample_cable` expect of a root.
    """
    n = graph.num_nodes()
    parents = np.full(n, -1, dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    soma = np.flatnonzero(np.asarray(types) == TYPE_SOMA).tolist()
    for root in soma + list(range(n)):
        if seen[root]:
            continue
        seen[root] = True
        for u, v in rx.dfs_edges(graph, source=int(root)):
            if not seen[v]:
                parents[v] = u
                seen[v] = True
    # A skeleton is a tree, so the DFS forest covers every edge. If it did not, the graph holds a
    # cycle and rebuilding from `parents` would silently drop the edges that closed it.
    n_links = int((parents != -1).sum())
    if n_links != graph.num_edges():
        raise ValueError(
            f"graph has {graph.num_edges()} edges but its DFS forest has {n_links}; "
            "not a tree, so its edges cannot be carried by a parent array"
        )
    return parents


def _cell_arrays(graph: SkeletonGraph) -> tuple[np.ndarray, ...]:
    """(types, parents, xyz, radius, synthetic_edge) for the whole graph, deriving what is absent."""
    feats = graph.node_feats
    n = graph.num_nodes()
    if "type" not in feats:
        raise ValueError("node_feats['type'] is a measurement, not derivable; see requirements.check")
    types = feats["type"]
    parents = feats["parent"] if "parent" in feats else parents_from_edges(graph, types)
    # NaN rather than a placeholder constant: a missing radius is unmeasured, and a
    # plausible-looking number here would reach morphopy's thickness statistics as if it were
    # data. Extractors that read radius ask for it, so `check` refuses the graph before this matters.
    radius = feats["radius"] if "radius" in feats else np.full(n, np.nan)
    synthetic = feats.get("synthetic_edge", np.zeros(n, dtype=bool))
    return types, np.asarray(parents), graph.node_xyz, radius, synthetic


def as_cell(graph: SkeletonGraph, *, reparent_orphans: bool = False) -> SkeletonGraph:
    """A cell equivalent to `graph`, with the derivable filled in and the narrow widened.

    This is what makes `parent` and `synthetic_edge` derived rather than required: a graph straight
    out of `SkeletonGraph.load()` carries only `type` and `radius`, at float16/float32.

    Returns the input object unchanged when nothing is missing, which is every cell `make_cell`
    builds. That fast path keeps the extractors bit-exact and costs nothing on a large cohort.
    Otherwise it rebuilds through `make_cell`, so it never mutates its argument.

    Widening is not recovery: float32 storage was measured to move a radial feature in its last
    digits, so a graph that arrives narrow yields the numbers its own precision allows, not ours.
    """
    feats = graph.node_feats
    if (
        not reparent_orphans
        and all(k in feats and feats[k].dtype == dt for k, dt in _CELL_DTYPES.items())
        and getattr(graph, "node_xyz", None) is not None
        and graph.node_xyz.dtype == np.float64
    ):
        return graph

    types, parents, xyz, radius, synthetic = _cell_arrays(graph)
    if reparent_orphans:
        parents, invented = reparent_to_soma(np.asarray(types), parents)
        synthetic = np.asarray(synthetic) | invented
    return make_cell(specimen_id(graph), types, parents, xyz, radius, synthetic)


def cells_from_graph(graph: SkeletonGraph, *, reparent_orphans: bool = False) -> list[SkeletonGraph]:
    """One cell per connected component, in component-id order.

    arborist loads whole fragment graphs, one component per SWC file (`load`, `check_swc_ids`), but
    a cell is one reconstruction and everything downstream of `specimen_id` assumes that. This is
    the split, kept explicit: `check` reports a multi-component graph rather than dividing it,
    because a feature extractor that silently returned more rows than it was given graphs would
    break the positional alignment its callers rely on.
    """
    types, parents, xyz, radius, synthetic = _cell_arrays(graph)
    component = graph.node_component_id
    cells = []
    for cid, sid in sorted(graph.component_id_to_swc_id.items()):
        keep = np.flatnonzero(component == cid)
        old_to_new = np.full(graph.num_nodes(), -1, dtype=np.int64)
        old_to_new[keep] = np.arange(len(keep))
        p = parents[keep]
        local = np.where(p == -1, -1, old_to_new[np.where(p == -1, 0, p)])
        if np.any(local[p != -1] == -1):
            raise ValueError(f"component {cid} has an edge to a node outside it")
        if reparent_orphans:
            local, invented = reparent_to_soma(np.asarray(types)[keep], local)
            syn = np.asarray(synthetic)[keep] | invented
        else:
            syn = np.asarray(synthetic)[keep]
        cells.append(make_cell(sid, np.asarray(types)[keep], local, np.asarray(xyz)[keep],
                               np.asarray(radius)[keep], syn))
    return cells


def filtered(cell: SkeletonGraph, keep_types) -> Optional[SkeletonGraph]:
    """
    Returns a cell holding only the nodes whose type is in "keep_types", with each kept node
    reparented to its nearest kept ancestor.

    Generic node removal cannot do this. It would orphan every node whose parent was dropped
    rather than reconnecting it, and it would lose the fact that a rewired link now spans an
    invented edge, which is why "synthetic_edge" is OR-ed along the skipped chain. Note also that
    `remove_nodes`/`relabel_nodes` would silently corrupt a cell: they re-slice every node_feats
    array positionally, which is right for every value except "parent", whose entries name nodes
    and would need remapping too.

    Returns None when too little survives to be a usable arbor: fewer than 3 nodes, or nothing but
    soma. Returns the input when nothing is dropped.
    """
    types, parents = cell.node_feats["type"], cell.node_feats["parent"]
    n = len(types)

    keep_mask = np.isin(types, list(keep_types))
    if keep_mask.all():
        return cell
    if keep_mask.sum() < 3:
        return None
    if not (keep_mask & (types != TYPE_SOMA)).any():
        return None

    old_to_new = -np.ones(n, dtype=np.int64)
    new_indices = np.flatnonzero(keep_mask)
    old_to_new[new_indices] = np.arange(len(new_indices))

    synthetic_edge = cell.node_feats["synthetic_edge"]
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

    return make_cell(
        specimen_id=specimen_id(cell),
        types=types[keep_mask].copy(),
        parents=new_parents,
        xyz=cell.node_xyz[keep_mask].copy(),
        radius=cell.node_feats["radius"][keep_mask].copy(),
        synthetic_edge=new_synthetic,
    )
