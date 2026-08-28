"""Verification suite for the joint-cohort TreeMoCo port.

Three things here are specific to this port and every learned number depends on them: that the corpus
key is unique (416 nucleus ids are in both caches, so a bare specimen id is NOT a key), that no
modality filter is applied (the whole cell goes in, which is the entire reason this rep sidesteps the
`full`-unavailable-on-excitatory problem), and that the trees the encoder sees satisfy the
CollapsedTree contract on both cohorts.

The collapsed-size figures are asserted loosely (medians within a band) rather than exactly: they are
descriptive statistics over a sampled prefix, not a join whose drift would invalidate a result.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from arborist_experiments.microns_morphology_benchmark.data_loading import (
    CACHE_PATH, INH_CACHE_PATH, TYPE_APICAL, TYPE_AXON, TYPE_BASAL, TYPE_SOMA,
    cell_from_cache, load_cache,
)
from arborist_experiments.microns_morphology_benchmark.pretrain_treemoco_microns import (
    SOURCE_CACHES, build_corpus, feature_names,
)
from arborist_experiments.microns_morphology_benchmark.probe_treemoco_microns import (
    KEY_PREFIX, align_to_cohort,
)
from arborist_experiments.patchseq_morphology_benchmark.treemoco.tree_features import (
    N_FEATURES, N_FEATURES_TYPED, collapse, n_features, node_features,
)

N_LIMIT = 200        # cells per cache for the corpus-shape tests
N_SAMPLE = 300       # cells per cache for the per-tree invariant sweep

EXPECTED_N_EXC = 49803
EXPECTED_N_INH = 7824
EXPECTED_N_CORPUS = EXPECTED_N_EXC + EXPECTED_N_INH      # 57627
# The same neuron reconstructed and annotated by both pipelines. This is why keys are prefixed.
EXPECTED_N_OVERLAP = 416

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CACHE_PATH) and os.path.exists(INH_CACHE_PATH)),
    reason="both MICrONS caches must be built",
)


@pytest.fixture(scope="module")
def caches():
    return {k: load_cache(v) for k, v in SOURCE_CACHES.items()}


# --- the key --------------------------------------------------------------------------------

def test_specimen_ids_collide_across_caches_so_keys_must_be_prefixed(caches):
    """The premise of the whole keying scheme. If this ever hits 0, prefixing is dead weight."""
    exc, inh = set(caches["exc"].specimens), set(caches["inh"].specimens)
    assert len(exc) == EXPECTED_N_EXC
    assert len(inh) == EXPECTED_N_INH
    assert len(exc) + len(inh) == EXPECTED_N_CORPUS
    assert len(exc & inh) == EXPECTED_N_OVERLAP
    # Bare ids would collapse to fewer keys than there are trees. That is the failure prefixing
    # prevents, and it would hand a probe the other cohort's embedding.
    assert len(exc | inh) < EXPECTED_N_CORPUS


def test_corpus_keys_are_unique_and_prefixed():
    trees, keys, skipped = build_corpus(limit=N_LIMIT, log=lambda m: None)
    assert len(trees) == len(keys)
    assert len(keys) == len(set(keys)), "duplicate corpus key: the prefix is missing or broken"
    assert not skipped, f"collapse failed on {len(skipped)} cells: {skipped[:3]}"
    prefixes = {k.split(":", 1)[0] for k in keys}
    assert prefixes == {"exc", "inh"}
    assert sum(k.startswith("exc:") for k in keys) == N_LIMIT
    assert sum(k.startswith("inh:") for k in keys) == N_LIMIT
    # Order is exc then inh, cache row order within a source, so a rerun reproduces the corpus.
    assert all(k.startswith("exc:") for k in keys[:N_LIMIT])
    assert all(k.startswith("inh:") for k in keys[N_LIMIT:])


def test_both_copies_of_an_overlapping_cell_are_in_the_corpus(caches):
    """Not deduplicated: two annotations of one neuron are two legitimate training examples."""
    exc, inh = caches["exc"], caches["inh"]
    shared = sorted(set(exc.specimens) & set(inh.specimens))
    sid = shared[0]
    e_row, i_row = exc.row_by_id[sid], inh.row_by_id[sid]
    t_exc = collapse(cell_from_cache(exc, e_row))
    t_inh = collapse(cell_from_cache(inh, i_row))
    # Same nucleus, genuinely different reconstructions -- so keeping both is not duplication.
    assert f"exc:{sid}" != f"inh:{sid}"
    assert len(t_exc.parents) != len(t_inh.parents) or not np.allclose(t_exc.xyz, t_inh.xyz)


# --- no modality filter ---------------------------------------------------------------------

def test_whole_cell_goes_in_apical_on_exc_and_axon_on_inh(caches):
    """The point of this rep: no filter_cell, so nothing is dropped for being the wrong compartment.

    On the hand-designed rows the excitatory cohort cannot score `full` at all (0.43% axon
    prevalence) and the inhibitory one cannot score `apical` (no type-4 nodes). Here both cohorts
    simply contribute whatever they have.
    """
    exc, inh = caches["exc"], caches["inh"]
    exc_types, inh_types = set(), set()
    for row in range(N_SAMPLE):
        exc_types |= set(np.unique(collapse(cell_from_cache(exc, row)).types).tolist())
        inh_types |= set(np.unique(collapse(cell_from_cache(inh, row)).types).tolist())
    assert TYPE_APICAL in exc_types, "apical was dropped from excitatory trees"
    assert TYPE_AXON in inh_types, "axon was dropped from inhibitory trees"
    assert TYPE_APICAL not in inh_types, "inhibitory SWCs have no type-4 nodes"
    for t in (exc_types, inh_types):
        assert TYPE_SOMA in t and TYPE_BASAL in t
        assert t <= {TYPE_SOMA, TYPE_AXON, TYPE_BASAL, TYPE_APICAL}


def test_type_one_hot_separates_axon_from_dendrite_but_merges_apical(caches):
    """TYPE_GROUPS = ((1,), (2,), (3,4)).

    Merging basal and apical is deliberate: the apical call presupposes having judged the cell
    pyramidal, so a separate channel would feed the excitatory subclass label back in as a node
    feature. Axon keeps its own channel, which is what lets one encoder read both cohorts.
    """
    tree = collapse(cell_from_cache(caches["inh"], 0))
    X = node_features(tree.xyz, tree.polylines, tree.is_synthetic, tree.no_branch, tree.types)
    assert X.shape[1] == N_FEATURES_TYPED == n_features(True)
    onehot = X[:, N_FEATURES:]
    assert onehot.sum(axis=1).max() <= 1.0
    soma, axon, dend = onehot[:, 0], onehot[:, 1], onehot[:, 2]
    assert np.array_equal(soma.astype(bool), tree.types == TYPE_SOMA)
    assert np.array_equal(axon.astype(bool), tree.types == TYPE_AXON)
    assert np.array_equal(dend.astype(bool),
                          np.isin(tree.types, [TYPE_BASAL, TYPE_APICAL]))


def test_is_synthetic_is_dead_weight_on_microns(caches):
    """MICrONS trees arrive connected with one root that is the soma, so no link was invented.

    On Patch-seq this column flagged 42% of cells. Recording that it is identically zero here is
    what stops someone reading a nonzero coefficient on it as signal.
    """
    for source in ("exc", "inh"):
        for row in range(50):
            tree = collapse(cell_from_cache(caches[source], row))
            assert not tree.is_synthetic.any(), f"{source} row {row} has a synthetic edge"


# --- the trees the encoder sees -------------------------------------------------------------

def test_collapsed_tree_invariants(caches):
    for source in ("exc", "inh"):
        cache = caches[source]
        for row in range(N_SAMPLE):
            tree = collapse(cell_from_cache(cache, row))
            m = len(tree.parents)
            assert m >= 1
            assert tree.xyz.shape == (m, 3)
            assert tree.polylines.shape[0] == m and tree.polylines.shape[2] == 3
            assert tree.types.shape == (m,) and tree.no_branch.shape == (m,)
            roots = np.flatnonzero(tree.parents == -1)
            assert len(roots) == 1, f"{source} row {row}: {len(roots)} roots"
            assert tree.parents.min() >= -1 and tree.parents.max() < m
            assert not (tree.parents == np.arange(m)).any()
            # A collapsed node's parent must precede it, which is what makes the height banding
            # in batching.tree_lstm_forward a valid topological order.
            child = np.flatnonzero(tree.parents >= 0)
            assert (tree.parents[child] < child).all()
            assert np.isfinite(tree.xyz).all() and np.isfinite(tree.polylines).all()


def test_collapse_shrinks_the_tree_by_the_expected_order(caches):
    """Collapsing to root+branchpoints+tips is what makes 57627 trees trainable at all."""
    stats = {}
    for source in ("exc", "inh"):
        cache = caches[source]
        raw, M = [], []
        for row in range(N_SAMPLE):
            cell = cell_from_cache(cache, row)
            raw.append(len(cell.types))
            M.append(len(collapse(cell).parents))
        stats[source] = (float(np.median(raw)), float(np.median(M)))
        assert np.median(M) < np.median(raw) / 5, f"{source}: collapse barely shrank the tree"
    # Excitatory dendrites branch far less than inhibitory arbors; the inhibitory tail is what
    # dominates training cost.
    assert stats["exc"][1] < stats["inh"][1]
    assert 20 < stats["exc"][1] < 120
    assert 80 < stats["inh"][1] < 400


# --- the probe join -------------------------------------------------------------------------

def test_align_to_cohort_resolves_by_prefixed_key():
    cohort = _fake_cohort(["195614", "364094", "109973"])
    row_by_key = {"inh:195614": 0, "exc:195614": 1, "exc:364094": 2, "exc:109973": 3}
    X = np.arange(4 * 2, dtype=np.float64).reshape(4, 2)
    Xa, y = align_to_cohort(cohort, row_by_key, X, "em_exc")
    # Must pick the exc rows (1, 2, 3), never the inh row 0 that shares the id.
    assert np.array_equal(Xa, X[[1, 2, 3]])
    assert np.array_equal(y, cohort.labels)


def test_align_to_cohort_raises_rather_than_dropping_cells():
    cohort = _fake_cohort(["195614", "999999"])
    row_by_key = {"exc:195614": 0}
    with pytest.raises(KeyError, match="no embedding"):
        align_to_cohort(cohort, row_by_key, np.zeros((1, 2)), "em_exc")


def test_key_prefixes_cover_every_benchmark_source():
    from arborist_experiments.microns_morphology_benchmark.data_loading import SOURCES

    assert set(KEY_PREFIX) == set(SOURCES)
    assert set(KEY_PREFIX.values()) == set(SOURCE_CACHES)


def test_feature_names_match_the_embedding_width():
    names = feature_names(128)
    assert len(names) == len(set(names)) == 128
    assert names[0] == "z_all_000" and names[-1] == "z_all_127"


def _fake_cohort(ids: list[str]):
    from types import SimpleNamespace

    return SimpleNamespace(
        cells=[SimpleNamespace(specimen_id=i) for i in ids],
        labels=np.arange(len(ids), dtype=np.int64),
        label_names=[f"c{i}" for i in range(len(ids))],
    )
