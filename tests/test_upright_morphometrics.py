"""The tier A/B/B' morphometrics: graphs in, one 51-column row out, NaN where there is nothing to measure.

Its own module rather than a section of `test_per_graph_api.py` for two reasons. Its call shape is
`graphs` alone -- no modality, because every column names its own compartment -- so it does not fit
that file's `(graphs, modality)` table. And importing it imports `neuron_morphology` and the vendored
`SimpleTree` beneath it, which is exactly the cost the package's `__getattr__` keeps out of a plain
`import arborist_experiments.features`; keeping it in one test module keeps that true of the suite too.
`test_the_lazy_re_export_does_not_pull_neuron_morphology` is what stops that lazy export from being
flattened into an eager one, and has to run in a subprocess because this module has already paid the
import cost by the time any test here runs.

What is pinned here is the contract, not the numbers. The numbers are gated against the published
cohorts in `scratch/upright_features_api/v1/verify_graph_path_morphometrics.py`, which compares real
cells to `scratch/features/upright_tiers/v2/*.h5` bit-exactly -- a hand-built six-node cell cannot do
that, and a hard-coded expected row here would only pin whatever this environment happened to produce.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("neuron_morphology")

from arborist_experiments.features import batch_names, design_matrix, skips
from arborist_experiments.features.upright_morphometrics import (
    FEATURES,
    INDEX,
    TIER_OF,
    extract_upright_morphometrics,
)
from arborist_experiments.utils.cells import TYPE_BASAL, graph_labels, make_cell

from .test_requirements_and_normalization import PARENTS, RADIUS, TYPES, XYZ


def _good(sid: str = "cell_a"):
    return make_cell(sid, TYPES, PARENTS, XYZ, RADIUS, np.zeros(len(TYPES), dtype=bool))


def _no_type(sid: str = "cell_b"):
    graph = _good(sid)
    del graph.node_feats["type"]
    return graph


def _no_soma(sid: str = "cell_c"):
    """Every one of the 51 is measured from the soma, so a cell without one is a reason, not a row."""
    types = TYPES.copy()
    types[0] = TYPE_BASAL
    return make_cell(sid, types, PARENTS, XYZ, RADIUS, np.zeros(len(types), dtype=bool))


def test_the_lazy_re_export_is_the_same_function_this_module_imported():
    import arborist_experiments.features as pkg

    assert pkg.extract_upright_morphometrics is extract_upright_morphometrics
    assert "extract_upright_morphometrics" in pkg.__all__


def test_the_lazy_re_export_does_not_pull_neuron_morphology():
    """Importing the package must not require `neuron_morphology` to be installed at all.

    A subprocess because this module imports it at module scope, so `sys.modules` is already polluted
    for every test in this file. Asserts the extractor is reachable in the same interpreter that has
    not imported the dependency, which is the whole claim the laziness makes.
    """
    snippet = (
        "import sys, arborist_experiments.features as f;"
        "print(any(m.startswith('neuron_morphology') for m in sys.modules));"
        "print(callable(f.extract_upright_morphometrics));"
        "print(any(m.startswith('neuron_morphology') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", snippet],
                         capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "True", "True"]


def test_the_columns_are_the_51_published_ones():
    assert len(FEATURES) == 51
    assert len(set(FEATURES)) == 51
    counts = {tier: sum(t == tier for t in TIER_OF.values()) for tier in ("A", "B", "B'")}
    assert counts == {"A": 29, "B": 13, "B'": 9}


def test_one_result_per_graph_with_the_width_declared_on_the_skipped_ones():
    graphs = [_good("a"), _no_type("b"), _no_soma("c")]
    results = extract_upright_morphometrics(graphs)

    assert len(results) == len(graphs)
    assert results[0].skipped is None
    assert results[0].X.shape == (51,) and results[0].X.dtype == np.float64
    assert results[1].X is None and "'type'" in results[1].skipped
    assert results[2].X is None and "soma" in results[2].skipped
    # The width is known before any graph is read, so it is on every element, skipped included.
    assert all(r.feature_names == FEATURES for r in results)

    X, kept = design_matrix(results, graph_labels(graphs))
    assert kept == ["a"] and X.shape == (1, 51)
    assert batch_names(results) == FEATURES
    assert [lab for lab, _r in skips(results, graph_labels(graphs))] == ["b", "c"]


def test_an_all_skipped_batch_still_reports_the_width():
    graphs = [_no_type("a"), _no_type("b")]
    X, kept = design_matrix(extract_upright_morphometrics(graphs), graph_labels(graphs))
    assert kept == [] and X.shape == (0, 51)


def test_nan_marks_an_absent_compartment_rather_than_a_missing_measurement():
    # The fixture is soma + basal + axon, no apical.
    row = extract_upright_morphometrics([_good("a")])[0].X
    apical = [n for n in FEATURES if n.startswith("apical_dendrite_")]
    assert len(apical) == 18   # 12 shared + surface_area/mean_diameter + the 4 apical-only
    assert all(np.isnan(row[INDEX[n]]) for n in apical)
    # Everything the cell does have is measured, with one exception that is the other half of the
    # NaN rule: the axon here is a single unbranched stub, and contraction is a per-branch ratio, so
    # upstream has nothing to average. NaN means "nothing to measure", not "measurement missing".
    present = [n for n in FEATURES if not n.startswith("apical_dendrite_")]
    assert [n for n in present if np.isnan(row[INDEX[n]])] == ["axon_mean_contraction"]


def test_the_row_is_translation_invariant():
    # Why there is no `soma_at_origin` requirement, unlike the radial features': upstream centers on
    # `get_root()` and reads `get_soma()`, so a cell in a foreign frame is still measured correctly.
    # Bit-exact here because the fixture's coordinates and the offset are all exact in binary; the
    # claim under test is the invariance, and a difference of any size would break it.
    at_origin = _good("a")
    moved = _good("a")
    moved.node_xyz = moved.node_xyz + 400.0
    here, there = extract_upright_morphometrics([at_origin, moved])
    assert there.skipped is None
    assert np.array_equal(here.X, there.X, equal_nan=True)


def test_a_fabricated_soma_link_is_not_counted_as_a_stem():
    # A fragmented reconstruction has no single root, which upstream requires, so the job reparents
    # orphans onto the soma. That invents cable at the one place it matters: an invented link makes a
    # fragment root look like a genuine stem. One real basal stem plus one detached basal fragment
    # must still count 1.
    types = np.array([TYPES[0], TYPE_BASAL, TYPE_BASAL, TYPE_BASAL])
    parents = np.array([-1, 0, -1, 2])
    xyz = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [30.0, 20.0, 0.0], [30.0, 40.0, 0.0]])
    radius = np.array([5.0, 1.0, 1.0, 0.8])
    fragmented = make_cell("frag", types, parents, xyz, radius, np.zeros(len(types), dtype=bool))

    result = extract_upright_morphometrics([fragmented])[0]
    assert result.skipped is None
    assert result.X[INDEX["basal_dendrite_calculate_number_of_stems"]] == 1.0
