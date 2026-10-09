"""Every extractor returns one result per input graph, in order, and says why when it has none.

This is the property the whole family was reshaped for. Before it, a batch of 160 cells could come
back as a `(0, 164)` matrix with all 160 in a `skipped` list, and nothing in the return value
distinguished that from a cohort of genuinely unusable cells -- the caller had rows and ids and no way
to ask "which graph did this one come from?" other than joining strings. An aligned list answers it
positionally, so a missing row is visible at the index it is missing from.

What is worth pinning here, beyond the length: that a skipped element carries `X is None` and a reason
rather than a zero row; that the reason survives into `skips` paired with the right label, including
the `<graph i>` fallback for a graph with no id at all; that `design_matrix` reports the batch width
even when nothing survived, which is what lets a caller concatenate blocks; that `concat_features`
joins two aligned batches into one block without moving a column, which is how `radial_hist` and
`radial_sholl` still add up to the retired `radial_arbors` vector; and that *every* extractor reports
its width from an all-skipped batch, because every one of them derives it from its own constants
rather than from the first cell that happened to succeed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from arborist.skeleton_graph import SkeletonGraph

from arborist_experiments.features import (
    MORPHOMETRIC_FEATURES,
    FeatureResult,
    batch_names,
    concat_features,
    density_feature_names,
    design_matrix,
    drop_zero_variance_and_impute,
    extract_density,
    extract_morphometrics,
    extract_persistence_image,
    extract_radial_hist,
    extract_radial_sholl,
    extract_sholl,
    persistence_feature_names,
    skips,
)
from arborist_experiments.features import features as morphopy_features
from arborist_experiments.utils.cells import graph_label, graph_labels, make_cell

from .test_requirements_and_normalization import BINNING, PARENTS, RADIUS, TYPES, XYZ

# `(graphs, modality) -> list[FeatureResult]`, each entry binding the constants its own extractor
# declares. The parametrized tests below only need one call shape; that is this table's business to
# arrange, not a reason for an extractor to accept a whole `BinningConfig`. `morphometric_statistics`
# and `radial_sholl` need no binding because they read no cohort constant.
EXTRACTORS = {
    "density_xy": lambda g, m: extract_density(g, m, "xy", **BINNING.density_kwargs()),
    "density_xz": lambda g, m: extract_density(g, m, "xz", **BINNING.density_kwargs()),
    "density_yz": lambda g, m: extract_density(g, m, "yz", **BINNING.density_kwargs()),
    "persistence_image": lambda g, m: extract_persistence_image(
        g, m, **BINNING.persistence_kwargs()),
    "sholl": lambda g, m: extract_sholl(g, m, sholl_radii=BINNING.sholl_radii),
    "morphometric_statistics": extract_morphometrics,
    "radial_hist": lambda g, m: extract_radial_hist(
        g, m, r_edges=BINNING.r_edges, z_edges=BINNING.z_edges),
    "radial_sholl": extract_radial_sholl,
}

# Every extractor's width on the `BINNING` config, none of it read off a cell: density from the
# config's ranges over `bin_size` (F3.2), persistence from `PersistenceImager.resolution` (F4.2),
# morphometrics from morphopy's own declared column list (F5.2), sholl from `sholl_radii`, and the two
# radial ones from `r_edges`/`z_edges` and module constants no cohort varies.
WIDTHS = {
    "density_xy": 1440,          # x 600/20 = 30 bins by y 950/20 = 48
    "density_xz": 300,           # by z 200/20 = 10
    "density_yz": 480,
    "persistence_image": 400,    # 20 x 20 pixels from birth_range/pers_range/pixel_size
    "sholl": len(BINNING.sholl_radii),
    "morphometric_statistics": len(MORPHOMETRIC_FEATURES),
    "radial_hist": 144,          # the `dendrite` modality: one structure group, not `full`'s two
    "radial_sholl": 20,
}


def _good(sid: str = "cell_a") -> SkeletonGraph:
    return make_cell(sid, TYPES, PARENTS, XYZ, RADIUS, np.zeros(len(TYPES), dtype=bool))


def _no_type(sid: str = "cell_b") -> SkeletonGraph:
    """A graph missing the one measurement nothing can derive, so `check` rejects it in the parent."""
    graph = _good(sid)
    del graph.node_feats["type"]
    return graph


def _no_id() -> SkeletonGraph:
    """Unusable and unnamed: no `type`, and no id for a reason to be attributed to."""
    graph = _no_type("unused")
    graph.component_id_to_swc_id.clear()
    return graph


@pytest.mark.parametrize("rep", list(EXTRACTORS))
def test_one_result_per_graph_with_a_reason_where_there_are_no_features(rep):
    graphs = [_good("a"), _no_type("b"), _good("c")]
    results = EXTRACTORS[rep](graphs, "dendrite")

    assert len(results) == len(graphs)
    for r in (results[0], results[2]):
        assert r.skipped is None
        assert r.X.ndim == 1 and r.X.dtype == np.float64
    assert results[1].X is None
    assert "'type'" in results[1].skipped


@pytest.mark.parametrize("rep", list(EXTRACTORS))
def test_design_matrix_and_skips_partition_the_batch_in_input_order(rep):
    graphs = [_good("a"), _no_type("b"), _good("c")]
    labels = graph_labels(graphs)
    results = EXTRACTORS[rep](graphs, "dendrite")

    X, kept = design_matrix(results, labels)
    reasons = skips(results, labels)
    assert kept == ["a", "c"]
    assert X.shape == (2, len(batch_names(results)))
    assert [lab for lab, _r in reasons] == ["b"]
    # Together they account for every graph exactly once, which is what makes the pair safe to log
    # against a cohort count.
    assert len(kept) + len(reasons) == len(graphs)


@pytest.mark.parametrize("rep", list(EXTRACTORS))
def test_a_graph_with_no_id_still_gets_its_reason_attributed(rep):
    graphs = [_good("a"), _no_id()]
    labels = graph_labels(graphs)
    assert labels == ["a", "<graph 1>"]
    reasons = skips(EXTRACTORS[rep](graphs, "dendrite"), labels)
    assert [lab for lab, _r in reasons] == ["<graph 1>"]


def test_graph_label_falls_back_to_the_position():
    assert graph_label(_good("a"), 7) == "a"
    assert graph_label(_no_id(), 7) == "<graph 7>"


@pytest.mark.parametrize("rep", list(EXTRACTORS))
def test_an_all_skipped_batch_still_reports_the_declared_width(rep):
    graphs = [_no_type("a"), _no_type("b")]
    results = EXTRACTORS[rep](graphs, "dendrite")
    X, kept = design_matrix(results, graph_labels(graphs))

    assert kept == []
    assert X.shape == (0, WIDTHS[rep])
    # The names live on the skipped elements themselves, which is the only place they can be.
    assert all(r.feature_names for r in results)


def test_radial_and_sholl_widths_are_the_published_ones():
    graphs = [_no_type("a")]
    hist = extract_radial_hist(graphs, "full", r_edges=BINNING.r_edges, z_edges=BINNING.z_edges)
    radial_sholl = extract_radial_sholl(graphs, "full")
    assert design_matrix(hist, graph_labels(graphs))[0].shape == (0, 288)
    assert design_matrix(radial_sholl, graph_labels(graphs))[0].shape == (0, 40)
    # 288 + 40 is the retired `radial_arbors` width, which is the sense in which the split moved no
    # column: two blocks of the same vector rather than two new features.
    both = concat_features(hist, radial_sholl)
    assert design_matrix(both, graph_labels(graphs))[0].shape == (0, 328)
    sholl = extract_sholl(graphs, "full", sholl_radii=BINNING.sholl_radii)
    assert design_matrix(sholl, graph_labels(graphs))[0].shape == (0, len(BINNING.sholl_radii))


@pytest.mark.parametrize("rep", list(EXTRACTORS))
def test_the_declared_width_is_the_one_the_library_produces(rep):
    """A width declared from a config is only worth having if it agrees with what morphopy returns.

    Which is the whole risk in closing F3.2/F4.2/F5.2: the alternative to reading the width off the
    first successful cell is a rule that can drift from the library's, so the rule is checked against
    a real extraction here and on every row in the extractors themselves.
    """
    results = EXTRACTORS[rep]([_good("a")], "dendrite")
    assert results[0].skipped is None
    assert len(results[0].X) == WIDTHS[rep] == len(results[0].feature_names)


# Both names functions take the geometry only -- the value-only parameters do not move a width, so
# `**BINNING.density_kwargs()` and `**BINNING.persistence_kwargs()` are deliberately not splattable
# into them.
GEOMETRY = {k: v for k, v in BINNING.density_kwargs().items()
            if k.endswith("_range") or k == "bin_size"}
PERSIM_GRID = {k: v for k, v in BINNING.persistence_kwargs().items()
               if k in ("birth_range", "pers_range", "pixel_size")}


def test_the_density_width_comes_from_the_ranges_and_persistence_from_persim():
    # 30 x-bins by 48 y-bins on the Patch-seq envelope; the projection picks which pair.
    assert len(density_feature_names("xy", **GEOMETRY)) == 1440
    assert len(density_feature_names("xz", **GEOMETRY)) == 300
    assert density_feature_names("xy", **GEOMETRY)[:2] == ["density_xy_0", "density_xy_1"]
    assert len(persistence_feature_names(**PERSIM_GRID)) == 400
    with pytest.raises(ValueError, match="unknown projection"):
        density_feature_names("zz", **GEOMETRY)


def test_n_bins_overrides_the_count_but_not_the_need_for_a_range():
    assert len(density_feature_names("xy", **GEOMETRY, n_bins=(3, 5))) == 15
    # The range is what makes two cells comparable; `n_bins` only says how finely to cut it.
    with pytest.raises(ValueError, match="needs y_range"):
        density_feature_names("xy", x_range=(-300.0, 300.0), n_bins=(3, 5))


def test_a_projection_without_its_ranges_has_no_declarable_width():
    # morphopy would then bin that axis over each *cell's* own extent, so the width is a property of
    # the cell and no names function can exist. Saying so beats declaring a box nobody chose.
    with pytest.raises(ValueError, match="needs y_range"):
        density_feature_names("xy", x_range=(-300.0, 300.0))
    with pytest.raises(ValueError, match="needs x_range"):
        density_feature_names("xz", z_range=(-50.0, 150.0))
    # `yz` does not read x, so dropping x_range is not an error -- it is just unused.
    assert len(density_feature_names("yz", y_range=(-350.0, 600.0), z_range=(-50.0, 150.0))) == 480


def test_the_persistence_grid_is_persims_own_truncating_arithmetic():
    # 500/25 = 20 pixels square. persim truncates rather than rounds, so a range that is not a whole
    # number of pixels loses the remainder -- int(500/30) = 16, not 17, and the image is 256 wide.
    assert len(persistence_feature_names(**PERSIM_GRID)) == 400
    assert len(persistence_feature_names(
        birth_range=(0.0, 500.0), pers_range=(0.0, 500.0), pixel_size=30.0)) == 256


def test_the_value_only_persim_parameters_reach_persim_without_moving_the_width():
    # The pass-through is what makes them worth naming in the signature: `n=0` weights every point of
    # the diagram equally instead of by its death, which is a different image of the same size.
    graphs = [_good("a")]
    default = extract_persistence_image(graphs, "dendrite", **BINNING.persistence_kwargs())
    flat = extract_persistence_image(graphs, "dendrite", **BINNING.persistence_kwargs(),
                                     weight_params={"n": 0.0})
    assert len(default[0].X) == len(flat[0].X) == 400
    assert not np.array_equal(default[0].X, flat[0].X)


def test_persistence_kwargs_rejects_a_config_key_persim_does_not_take():
    extra = replace(BINNING, persim_kwargs={**BINNING.persim_kwargs, "resolution": (20, 20)})
    with pytest.raises(ValueError, match="not PersistenceImager parameters"):
        extra.persistence_kwargs()
    # persim's own defaults are a 1 um image, which is not a box any cohort means.
    without = {k: v for k, v in BINNING.persim_kwargs.items() if k != "pixel_size"}
    with pytest.raises(ValueError, match="missing"):
        replace(BINNING, persim_kwargs=without).persistence_kwargs()


def test_density_kwargs_rejects_a_config_key_that_maps_to_no_parameter():
    # morphopy read its config through `.get(name, default)`, so a typo was a silent default. The
    # dict is still the stored form -- `constants_digest` hashes it -- but the translation now raises.
    typo = replace(BINNING, density_config={**BINNING.density_config, "bin_sizes": 25})
    with pytest.raises(ValueError, match="maps to no extract_density parameter"):
        typo.density_kwargs()
    half = {k: v for k, v in BINNING.density_config.items() if k != "r_max_y"}
    with pytest.raises(ValueError, match="must be given together"):
        replace(BINNING, density_config=half).density_kwargs()


def test_a_declared_width_that_disagrees_with_morphopy_raises(monkeypatch):
    # The failure the per-row check exists for. Absorbed as a per-graph reason it would empty the
    # cohort one cell at a time; absorbed as a `design_matrix` width disagreement it would drop every
    # row. Both are the silent wrongness this family was reshaped to remove, so it raises.
    monkeypatch.setattr(morphopy_features, "_n_bins", lambda *a: (7, 1))
    with pytest.raises(ValueError, match="have diverged"):
        extract_density([_good("a")], "dendrite", "xy", **BINNING.density_kwargs())


def test_morphometrics_columns_are_declared_rather_than_discovered():
    results = extract_morphometrics([_good("a"), _no_type("b")], "dendrite")
    assert batch_names(results) == MORPHOMETRIC_FEATURES
    # Including on the skipped element, where nothing could have been discovered from.
    assert results[1].feature_names == MORPHOMETRIC_FEATURES
    # morphopy's own order, so the declaration is a claim about the library and is checked per row:
    # a renamed or reordered statistic is a raise, not a silently relabelled matrix.
    assert MORPHOMETRIC_FEATURES[:3] == ["branch_points", "width", "depth"]
    assert MORPHOMETRIC_FEATURES[-1] == "tree_asymmetry"


def test_concat_features_joins_two_batches_column_wise_in_order():
    graphs = [_good("a"), _no_type("b")]
    hist = extract_radial_hist(graphs, "dendrite", r_edges=BINNING.r_edges,
                               z_edges=BINNING.z_edges)
    sholl = extract_radial_sholl(graphs, "dendrite")
    both = concat_features(hist, sholl)

    assert len(both) == 2
    assert batch_names(both) == batch_names(hist) + batch_names(sholl)
    assert np.array_equal(both[0].X, np.concatenate([hist[0].X, sholl[0].X]))
    # A block cannot be half a row: skipped in either part is skipped in the join, with the width
    # still readable from the names both parts declare.
    assert both[1].X is None and "'type'" in both[1].skipped
    assert len(both[1].feature_names) == len(hist[1].feature_names) + len(sholl[1].feature_names)


def test_concat_features_refuses_misaligned_batches():
    graphs = [_good("a"), _good("c")]
    sholl = extract_radial_sholl(graphs, "dendrite")
    with pytest.raises(ValueError, match="not aligned"):
        concat_features(sholl, sholl[:1])


# --- the cohort-level morphometrics step ------------------------------------------------------

def _hand_built():
    """Three cells, a constant column, an all-NaN column and one missing value to impute."""
    names = ["varies", "constant", "all_nan", "has_nan"]
    rows = [
        np.array([1.0, 7.0, np.nan, 2.0]),
        np.array([2.0, 7.0, np.nan, 4.0]),
        np.array([3.0, 7.0, np.nan, np.nan]),
    ]
    return [FeatureResult(X=r, feature_names=names, skipped=None) for r in rows]


def test_drop_zero_variance_and_impute_keeps_the_batch_aligned():
    results = _hand_built()
    results.insert(1, FeatureResult(X=None, feature_names=results[0].feature_names,
                                   skipped="morphometrics: nope"))
    out = drop_zero_variance_and_impute(results)

    assert len(out) == len(results)
    assert batch_names(out) == ["varies", "has_nan"]
    assert out[1].X is None and out[1].skipped == "morphometrics: nope"
    # A skipped element carries the surviving column set too, so the batch width is readable from it.
    assert out[1].feature_names == ["varies", "has_nan"]

    X, kept = design_matrix(out, ["a", "b", "c", "d"])
    assert kept == ["a", "c", "d"]
    # The mean is over the surviving columns' present values: (2 + 4) / 2.
    assert np.array_equal(X, np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 3.0]]))


def test_drop_zero_variance_and_impute_is_a_no_op_without_survivors():
    results = [FeatureResult(X=None, feature_names=["a"], skipped="nope")]
    assert drop_zero_variance_and_impute(results) == results
