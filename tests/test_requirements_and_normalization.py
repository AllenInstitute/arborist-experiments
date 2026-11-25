"""A graph arborist itself produced must extract, and anything unusable must say why.

The regression these pin down was measured, not imagined. `make_cell` writes `node_feats`
`{type, radius, parent, synthetic_edge}`; `SkeletonGraph.load()` writes `{radius, type}` only, at
float16/float32. Handing the second kind to `extract_morphometrics` raised `KeyError: 'parent'`, and
handing it to the radial extractors was worse -- `_job_radial_hist` caught that KeyError per cell
and returned a `(0, 164)` matrix with all 160 cells in `skipped`, which is indistinguishable from a
cohort of genuinely unusable cells. `test_radial_features_on_an_arborist_graph_are_not_empty` is the
test that would have caught it.

The fixtures are built by hand rather than by `load()` on a temp SWC on purpose. arborist's loader
resamples edges to `node_spacing` through a spline and assigns compartment codes per resampled edge,
so a `load()`-built fixture has smeared types, ~8x the nodes it was given and a soma centroid that is
no longer where the SWC put it. That is the loader's business; what these tests need is exactly its
`node_feats`/dtype contract, which is what `_arborist_native` reproduces.
"""

from __future__ import annotations

import numpy as np
import pytest
from arborist.skeleton_graph import SkeletonGraph

from arborist_experiments.features import BinningConfig, concat_features
from arborist_experiments.features import features as F
from arborist_experiments.features import radial_arbor_features as RA
from arborist_experiments.features.mapping import map_graphs
from arborist_experiments.utils.cells import (
    TYPE_AXON,
    TYPE_BASAL,
    TYPE_SOMA,
    as_cell,
    cells_from_graph,
    check,
    graph_labels,
    make_cell,
    specimen_id,
)

# One soma at the origin, a two-branch basal dendrite and an axon stub. Small, but every feature
# family has something to measure: two arbor groups, a branch point and a soma to be centered on.
TYPES = np.array([TYPE_SOMA, TYPE_BASAL, TYPE_BASAL, TYPE_BASAL, TYPE_BASAL, TYPE_AXON])
PARENTS = np.array([-1, 0, 1, 2, 2, 0])
XYZ = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [0.0, 25.0, 0.0],
    [12.0, 40.0, 0.0],
    [-12.0, 40.0, 0.0],
    [0.0, -30.0, 5.0],
])
RADIUS = np.array([5.0, 1.0, 0.9, 0.5, 0.5, 0.6])

# A cohort's binning constants, owned by the test suite rather than imported from one. These are
# the Patch-seq numbers verbatim, because `test_per_graph_api.WIDTHS` hardcodes the matrix widths
# they imply and this config is what makes those expectations true. Copied rather than imported so
# that the tests below pin the *extractors'* contract: a real cohort config lives in capsule code,
# is free to be retuned for its dataset, and must not be able to break this suite when it is.
BINNING = BinningConfig(
    name="test_cohort",
    density_config={
        "density": True,
        "smooth": True,
        "sigma": 1,
        "bin_size": 20,
        "r_min_x": -300.0, "r_max_x": 300.0,
        "r_min_y": -350.0, "r_max_y": 600.0,
        "r_min_z":  -50.0, "r_max_z": 150.0,
    },
    sholl_radii=list(np.arange(10.0, 610.0, 10.0)),
    persim_kwargs={
        "birth_range": (0.0, 500.0),
        "pers_range": (0.0, 500.0),
        "pixel_size": 25.0,
    },
    r_edges=np.linspace(0.0, 320.0, 13),
    z_edges=np.linspace(-350.0, 600.0, 13),
)


def _our_cell(sid: str = "cell_a") -> SkeletonGraph:
    return make_cell(sid, TYPES, PARENTS, XYZ, RADIUS, np.zeros(len(TYPES), dtype=bool))


def _arborist_native(sid: str = "cell_a", n_copies: int = 1) -> SkeletonGraph:
    """A graph carrying exactly what `SkeletonGraph.load()` allocates, and nothing else."""
    n = len(TYPES)
    graph = SkeletonGraph(verbose=False)
    graph.add_nodes_from([None] * (n * n_copies))
    edges = [
        (int(p) + c * n, i + c * n)
        for c in range(n_copies)
        for i, p in enumerate(PARENTS)
        if p != -1
    ]
    graph.add_edges_from_no_data(edges)
    graph.node_xyz = np.tile(XYZ, (n_copies, 1)).astype(np.float32)
    graph.node_component_id = np.repeat(np.arange(n_copies), n)
    graph.node_feats = {
        "radius": np.tile(RADIUS, n_copies).astype(np.float16),
        "type": np.tile(TYPES, n_copies).astype(np.uint8),
    }
    for c in range(n_copies):
        graph.component_id_to_swc_id[c] = f"{sid}.{c}" if n_copies > 1 else sid
    return graph


@pytest.mark.parametrize("req", [RA.REQ, F.REQ, F.REQ_SOMA], ids=["radial", "morphopy", "soma"])
def test_an_arborist_native_graph_satisfies_every_requirement(req):
    # The point of asking only for irreducible data: `load()` populates `type` and `radius`, and
    # everything else an extractor reads is derived or defaulted, so nothing here is a skip.
    assert check(_arborist_native(), **req) == []


def _radial(cell):
    """Both radial families as one block, so these checks cover the pair rather than one of them."""
    return concat_features(
        RA.extract_radial_hist([cell], "dendrite", r_edges=BINNING.r_edges,
                               z_edges=BINNING.z_edges),
        RA.extract_radial_sholl([cell], "dendrite"),
    )


def test_radial_features_on_an_arborist_graph_are_not_empty():
    results = _radial(_arborist_native())
    assert len(results) == 1 and results[0].skipped is None
    assert results[0].X.shape == (len(results[0].feature_names),)
    assert results[0].X.any(), "an arbor with 4 dendrite nodes cannot produce an all-zero row"


def test_the_arborist_graph_and_ours_agree():
    # Same cell, two dtypes. Not bit-equality -- float32 xyz and float16 radius are lossy, which is
    # why `as_cell` logs rather than claims to have recovered anything -- but the same features.
    ours = _radial(_our_cell())
    theirs = _radial(_arborist_native())
    assert np.allclose(ours[0].X, theirs[0].X, rtol=1e-3, atol=1e-6)


def test_a_missing_measurement_is_a_reason_and_a_derivable_one_is_not():
    graph = _arborist_native()
    del graph.node_feats["type"]
    assert any("'type'" in r for r in check(graph, **RA.REQ))

    graph = _arborist_native()
    del graph.node_feats["radius"]
    assert check(graph, **RA.REQ) == [], "radial arbors never read node radius"
    assert any("'radius'" in r for r in check(graph, **F.REQ))


def test_a_multi_component_graph_is_a_reason_that_names_the_split():
    graph = _arborist_native(n_copies=3)
    reasons = check(graph, **RA.REQ)
    assert len(reasons) == 1 and "cells_from_graph" in reasons[0]

    cells = cells_from_graph(graph)
    assert [specimen_id(c) for c in cells] == ["cell_a.0", "cell_a.1", "cell_a.2"]
    for cell in cells:
        assert check(cell, **F.REQ_SOMA) == []
        assert np.array_equal(cell.node_feats["parent"], PARENTS)


def test_an_uncentered_graph_is_a_reason():
    graph = _arborist_native()
    graph.node_xyz = graph.node_xyz + np.float32(400.0)
    assert not check(graph, **F.REQ), "the morphopy four do not require a frame"
    assert any("origin" in r for r in check(graph, **RA.REQ))


def test_a_forest_is_not_rejected():
    # `filtered()` output is legitimately a forest and 91% of the unlabelled Patch-seq tier is
    # fragmented, so a connectivity test here would reject real cells by the thousand.
    parents = PARENTS.copy()
    parents[1] = -1
    forest = make_cell("cell_a", TYPES, parents, XYZ, RADIUS, np.zeros(len(TYPES), dtype=bool))
    assert check(forest, **F.REQ_SOMA) == []
    assert as_cell(forest) is forest


def test_as_cell_is_identity_on_our_own_cells():
    cell = _our_cell()
    assert as_cell(cell) is cell


def test_as_cell_fills_in_what_the_graph_already_implies():
    graph = _arborist_native()
    cell = as_cell(graph)
    assert cell is not graph, "must not mutate its argument"
    assert "parent" not in graph.node_feats
    assert np.array_equal(cell.node_feats["parent"], PARENTS)
    assert not cell.node_feats["synthetic_edge"].any()
    assert cell.node_xyz.dtype == np.float64 and cell.node_feats["radius"].dtype == np.float64
    assert as_cell(cell) is cell, "normalizing twice must cost nothing"


def test_as_cell_reparents_orphans_only_when_asked():
    parents = PARENTS.copy()
    parents[5] = -1                                     # the axon stub, adrift
    graph = make_cell("cell_a", TYPES, parents, XYZ, RADIUS, np.zeros(len(TYPES), dtype=bool))

    reparented = as_cell(graph, reparent_orphans=True)
    assert reparented.node_feats["parent"][5] == 0
    assert list(reparented.node_feats["synthetic_edge"]) == [False] * 5 + [True]
    assert graph.node_feats["parent"][5] == -1, "the input must be untouched"


def _job_nodes(cell, scale):
    return np.array([cell.num_nodes() * scale], dtype=float), None, None


def test_map_graphs_returns_one_entry_per_graph_in_order():
    graphs = [_arborist_native("a"), _arborist_native("b"), _arborist_native("c")]
    del graphs[1].node_feats["type"]

    results = map_graphs(graphs, RA.REQ, _job_nodes, 2)
    assert len(results) == len(graphs)
    assert results[0][0][0] == 12.0 and results[2][0][0] == 12.0
    assert results[0][2] is None and results[2][2] is None
    assert results[1][0] is None and "'type'" in results[1][2]


def test_map_graphs_labels_a_graph_that_has_no_id_by_position():
    # A one-cell-per-graph violation is exactly the case that can take the id away, so the skip is
    # reported against the position instead of guessing at a name. `map_graphs` no longer carries the
    # id -- the caller derives it from the graphs it still holds -- so `graph_labels` is where the
    # fallback shows up.
    graph = _arborist_native("b")
    graph.component_id_to_swc_id = {}
    graphs = [_our_cell("a"), graph]
    results = map_graphs(graphs, RA.REQ, _job_nodes, 1)
    assert graph_labels(graphs) == ["a", "<graph 1>"]
    assert "cells_from_graph" in results[1][2]


def test_map_graphs_with_nothing_runnable_still_aligns():
    graphs = [_arborist_native("a"), _arborist_native("b")]
    for graph in graphs:
        del graph.node_feats["type"]
    results = map_graphs(graphs, RA.REQ, _job_nodes, 1)
    assert [r[0] for r in results] == [None, None]
    assert all(r[2] for r in results)
