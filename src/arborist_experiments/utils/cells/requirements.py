"""What a caller needs from a graph, checked before the graph is dispatched to a worker.

Checking up front means a graph a feature cannot use is skipped with a stated reason, rather than
failing inside a worker or -- worse -- yielding a silent wrong answer.

There is no single contract here, because requirements vary by feature and the varying part is the
primary object: each extractor declares its own asks as a module-level `REQ` dict beside the
function, and passes them as keywords. What `check` adds on top is the pair of invariants that follow
from the input *type* rather than from any feature -- one cell per graph, and arrays sized to the node
count. `type` is deliberately not in that pair even though all seven extractors need it: they need it
because all seven take a `modality`, not because a `SkeletonGraph` is invalid without it.

The reason strings live in one place on purpose. The promise this file exists to keep is "skip and
complain, never silently"; five extractors inventing their own phrasings is how that promise decays.

Two things a requirement is not:

  - **Derivable data.** `parent` is a cache of the edge set (see `make_cell`) and `synthetic_edge`
    defaults to all-False, so `as_cell` supplies both and neither is ever asked for. A graph is not
    rejected for lacking something it already implies.
  - **A dtype.** arborist's `load()` allocates float32 `node_xyz` and float16 `radius`; refusing to
    extract features from an arborist-loaded graph would be absurd. `as_cell` widens instead.

Connectivity is not checked, and must not be. A `filtered()` output is legitimately a forest --
dropping a type strands its children at `parent == -1` -- and Patch-seq reconstructions are
fragmented (91% of the unlabelled tier, median 74 fragments). A `connected_components` test would
reject real cells by the thousand, which is the failure mode this file exists to remove. The one-cell
test below is about ids, not edges.
"""

import numpy as np
from arborist.skeleton_graph import SkeletonGraph

from .cell_graph import TYPE_SOMA

# Microns. A tripwire for a graph in an absolute frame, not a tolerance anything real sits near: a
# soma-centered cell's soma centroid is 0 to within float rounding, against the tens-to-hundreds of
# microns an uncentered cell shows.
SOMA_ORIGIN_TOL = 1e-6


def check(
    cell: SkeletonGraph,
    *,
    keys: tuple[str, ...] = (),
    needs_soma: bool = False,
    soma_at_origin: bool = False,
) -> list[str]:
    """Reasons `cell` cannot be used. Empty means it can.

    Every keyword is optional and names something a particular feature reads:

      - `keys`: `node_feats` entries that are real measurements, so a missing one cannot be
        invented. `type` (nothing else can drive a modality filter) and `radius` (reaches the
        morphopy thickness/surface/volume statistics through `cell_to_swc_df`).
      - `needs_soma`: a node of type `TYPE_SOMA` must exist, because the feature is measured from
        it -- morphopy's `get_sholl_intersection_profile(centroid="soma")`.
      - `soma_at_origin`: the coordinates must already be soma-relative, because the feature reads
        `node_xyz` as a soma-centered displacement without re-centering it
        (`radial_arbor_features.sholl_profile` takes `norm(node_xyz)` directly). Needs a soma node
        to verify against, so a graph without one is skipped rather than assumed centered.
    """
    reasons: list[str] = []
    n = cell.num_nodes()
    feats = cell.node_feats

    xyz = getattr(cell, "node_xyz", None)
    if xyz is None:
        reasons.append("no node_xyz")
    elif np.shape(xyz) != (n, 3):
        reasons.append(f"node_xyz has shape {np.shape(xyz)}, expected {(n, 3)}")

    for key in sorted(feats):
        if len(feats[key]) != n:
            reasons.append(f"node_feats[{key!r}] has {len(feats[key])} entries for {n} nodes")

    # The invariant `make_cell` establishes and `specimen_id` reads. No feature computation needs it
    # -- `radial_histogram` would happily pool two cells' cable and return a number -- but the unit
    # of work does: `specimen_id`, `FeatureResult` and `cohort_key` all assume the graph names one
    # reconstruction. Splitting here instead would break the positional alignment callers rely on.
    swc_ids = cell.component_id_to_swc_id
    component = getattr(cell, "node_component_id", None)
    if set(swc_ids) != {0}:
        reasons.append(
            f"one cell per graph, but component_id_to_swc_id names {len(swc_ids)}; "
            "split it with cells_from_graph"
        )
    elif component is None or (n and not np.all(np.asarray(component) == 0)):
        reasons.append("node_component_id disagrees with component_id_to_swc_id")

    wanted = list(keys) + (["type"] if needs_soma or soma_at_origin else [])
    missing = [key for key in dict.fromkeys(wanted) if key not in feats]
    if missing:
        reasons.append("missing node_feats " + ", ".join(repr(key) for key in missing))

    if (needs_soma or soma_at_origin) and "type" in feats:
        soma = np.asarray(feats["type"]) == TYPE_SOMA
        if not soma.any():
            reasons.append("no soma node")
        elif soma_at_origin and xyz is not None and np.shape(xyz) == (n, 3):
            offset = float(np.linalg.norm(np.asarray(xyz, dtype=np.float64)[soma].mean(axis=0)))
            if offset > SOMA_ORIGIN_TOL:
                reasons.append(f"soma centroid is {offset:.3g} um from the origin, not centered")

    return reasons
