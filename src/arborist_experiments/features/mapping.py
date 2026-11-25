"""One result per input graph, in order, with a reason wherever there is no result.

Requirements are checked here, in the parent, before anything is pickled to a worker. A graph that
fails them would otherwise raise inside a worker or -- where an extractor catches exceptions per
cell -- come back as an empty row, which reads exactly like an unusable cell (see
`utils.cells.requirements`).

The division of labour matters for cost, not tidiness. `check` is a handful of array attribute reads,
so it runs on every graph in the parent. `as_cell` may walk the edge set to recover `parent`, which
is an O(n) Python pass, so it runs inside the worker on the graphs that survived -- doing it here
would serialize the very work the pool exists to parallelise.

The return is a 3-tuple `(value, names, reason)` per input graph. No id: a `FeatureResult` has
none, and the caller recovers ids from the graphs it passed with `graph_labels`. Requirements
failures fill the `reason` slot in the parent; runtime failures ("no nodes for modality",
"persistence: empty diagram") come back in the same slot from the workers, and are not absorbed or
reworded here.
"""

import logging

import numpy as np

from arborist_experiments.utils.cells import as_cell, check, graph_label, run_jobs

log = logging.getLogger(__name__)


def _is_narrow(graph) -> bool:
    radius = graph.node_feats.get("radius")
    return (
        getattr(graph, "node_xyz", None) is not None and graph.node_xyz.dtype != np.float64
    ) or (radius is not None and radius.dtype != np.float64)


def _normalized(cell, job, *job_args):
    """`job`, run on a cell rather than on whatever the caller happened to have."""
    return job(as_cell(cell), *job_args)


def map_graphs(graphs, req: dict, job, *job_args) -> list[tuple]:
    """Run `job` on each graph that satisfies `req`, keeping one entry per input graph.

    `req` is the extractor's own `REQ` dict, passed to `check` as keywords. Skipped positions carry
    `(None, None, reason)`, so the output is positionally aligned with `graphs` and a caller can zip
    it against labels without tracking which cells dropped out.
    """
    results: list[tuple | None] = [None] * len(graphs)
    runnable, positions = [], []
    for i, graph in enumerate(graphs):
        reasons = check(graph, **req)
        if reasons:
            reason = "; ".join(reasons)
            log.warning("skipping %s: %s", graph_label(graph, i), reason)
            results[i] = (None, None, reason)
        else:
            runnable.append(graph)
            positions.append(i)

    narrow = sum(_is_narrow(g) for g in runnable)
    if narrow:
        # Once per batch, not per cell. arborist's `load()` allocates float32 node_xyz and float16
        # radius; widening in `as_cell` makes those graphs usable but cannot recover the precision,
        # so their feature values can differ from a float64 source's in the last digits.
        log.warning(
            "%d/%d graphs carry float32 node_xyz or float16 radius; widening to float64 does not "
            "recover the lost precision", narrow, len(runnable),
        )

    for position, result in zip(positions, run_jobs(runnable, _normalized, job, *job_args)):
        results[position] = result
    return results
