"""What the persistence layer has to guarantee before anything is trusted to it.

Two properties matter and neither is obvious from reading the code. First, a cached matrix must
reproduce a fresh extraction exactly -- if it does not, the cache silently changes results instead
of saving time, which is worse than no cache. Second, the out-of-fold record must be a sufficient
statistic: every number the run reports has to be recomputable from `(proba, y, fold)` alone, or
the next question still costs a rerun.

The key tests are the ones that would fire on a real mistake: a key that ignores something the
features depend on (so two different cohorts collide), and an OOF matrix whose rows do not line up
with the specimen ids they are joined by.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from arborist_experiments.global_classification.cv import persistence
from arborist_experiments.global_classification.cv.classifier import (
    GRID,
    CVResult,
    boundary_hits,
    run_cv,
)
from arborist_experiments.global_classification.extract import FeatureResult
from arborist_experiments.global_classification.microns import feature_cache
from arborist_experiments.global_classification.microns.binning import EM_EXC


def _cohort(ids):
    return SimpleNamespace(cells=[SimpleNamespace(specimen_id=i) for i in ids])


def _fr(ids, n_features=4):
    rng = np.random.default_rng(0)
    return FeatureResult(
        specimen_ids=list(ids),
        X=rng.normal(size=(len(ids), n_features)),
        feature_names=[f"f{i}" for i in range(n_features)],
        skipped=[],
    )


# --- the cache key --------------------------------------------------------------------------

def test_key_changes_with_membership_radius_and_constants():
    a = _cohort(["1", "2", "3"])
    b = _cohort(["1", "2", "4"])
    k = feature_cache.cohort_key(a, "em_exc", False, EM_EXC)
    assert feature_cache.cohort_key(b, "em_exc", False, EM_EXC) != k, "membership must be in key"
    assert feature_cache.cohort_key(a, "microns_inh", False, EM_EXC) != k
    # real_radius changes feature *values* without changing membership, so it cannot be implied.
    assert feature_cache.cohort_key(a, "em_exc", True, EM_EXC) != k

    coarser = replace(EM_EXC, density_config={**EM_EXC.density_config, "bin_size": 25})
    assert feature_cache.cohort_key(a, "em_exc", False, coarser) != k, "constants must be in key"


def test_key_is_order_independent():
    assert (feature_cache.cohort_key(_cohort(["3", "1", "2"]), "em_exc", False, EM_EXC)
            == feature_cache.cohort_key(_cohort(["1", "2", "3"]), "em_exc", False, EM_EXC))


# --- round trip -----------------------------------------------------------------------------

def test_round_trip_is_bit_exact(tmp_path):
    cohort = _cohort(["10", "11", "12"])
    key = feature_cache.cohort_key(cohort, "em_exc", False, EM_EXC)
    fr = _fr(["10", "11", "12"])
    fr.skipped.append(("13", "sholl: ValueError: boom"))
    feature_cache.save(key, "sholl", "dendrite", fr, {"source": "em_exc"}, root=str(tmp_path))
    got = feature_cache.load(key, "sholl", "dendrite", cohort=cohort, root=str(tmp_path))
    assert np.array_equal(got.X, fr.X)              # bit-exact, not allclose
    assert got.X.dtype == fr.X.dtype == np.float64
    assert got.specimen_ids == fr.specimen_ids
    assert got.feature_names == fr.feature_names
    assert got.skipped == fr.skipped


def test_miss_returns_none_rather_than_raising(tmp_path):
    key = feature_cache.cohort_key(_cohort(["1"]), "em_exc", False, EM_EXC)
    assert feature_cache.load(key, "sholl", "dendrite", root=str(tmp_path)) is None


def test_load_rejects_an_entry_holding_foreign_cells(tmp_path):
    """A key collision must fail loudly; features silently belonging to other cells is the worst
    outcome the cache can produce."""
    key = "handwritten_key"
    feature_cache.save(key, "sholl", "dendrite", _fr(["1", "2"]), None, root=str(tmp_path))
    with pytest.raises(ValueError, match="not in this cohort"):
        feature_cache.load(key, "sholl", "dendrite", cohort=_cohort(["1", "9"]),
                           root=str(tmp_path))


def test_no_partial_file_is_left_behind(tmp_path):
    key = feature_cache.cohort_key(_cohort(["1"]), "em_exc", False, EM_EXC)
    feature_cache.save(key, "sholl", "dendrite", _fr(["1"]), None, root=str(tmp_path))
    d = feature_cache.cache_dir(key, root=str(tmp_path))
    import os
    assert not [n for n in os.listdir(d) if "partial" in n]


# --- the out-of-fold record ------------------------------------------------------------------

@pytest.fixture(scope="module")
def cv_and_data():
    rng = np.random.default_rng(0)
    n, k = 150, 3
    y = np.repeat(np.arange(k), n // k)
    X = rng.normal(size=(n, 6)) + y[:, None]
    cv = run_cv(X, y, class_names=[f"c{i}" for i in range(k)], outer_k=3, inner_k=2,
                seed=0, solver="saga")
    return cv, X, y


def test_run_cv_returns_a_complete_oof_record(cv_and_data):
    cv, _X, y = cv_and_data
    assert isinstance(cv, CVResult)
    assert cv.oof_proba.shape == (len(y), cv.n_classes)
    # Outer test folds are disjoint and cover every sample, so nothing may stay unassigned.
    assert (cv.oof_fold >= 0).all() and not np.isnan(cv.oof_proba).any()
    assert np.bincount(cv.oof_fold).tolist() == [50, 50, 50]
    assert np.allclose(cv.oof_proba.sum(axis=1), 1.0)
    assert np.array_equal(cv.oof_pred, cv.oof_proba.argmax(axis=1))


def test_every_reported_number_is_recomputable_from_the_oof(cv_and_data):
    """The point of saving the OOF: the metrics row and both confusion views follow from it."""
    cv, _X, y = cv_and_data
    for k in range(3):
        m = cv.oof_fold == k
        assert cv.metrics["balanced_accuracy"][k] == pytest.approx(
            balanced_accuracy_score(y[m], cv.oof_pred[m]))
    assert np.array_equal(cv.confusion_per_fold.sum(axis=0), cv.confusion)
    for k in range(3):
        m = cv.oof_fold == k
        assert np.array_equal(cv.confusion_per_fold[k],
                              confusion_matrix(y[m], cv.oof_pred[m],
                                               labels=np.arange(cv.n_classes)))


def test_write_oof_round_trips_and_stays_joinable(cv_and_data, tmp_path):
    cv, _X, y = cv_and_data
    ids = [f"cell{i}" for i in range(len(y))]
    path = str(tmp_path / "oof.npz")
    persistence.write_oof(path, cv.oof_proba, y, cv.oof_fold, ids, cv.class_names)
    with np.load(path) as z:
        assert [str(s) for s in z["specimen_ids"]] == ids
        assert [str(s) for s in z["class_names"]] == cv.class_names
        assert np.array_equal(z["proba"], cv.oof_proba)
        assert np.array_equal(z["y"], y) and np.array_equal(z["fold"], cv.oof_fold)


def test_write_oof_refuses_a_misaligned_id_list(cv_and_data, tmp_path):
    cv, _X, y = cv_and_data
    with pytest.raises(ValueError, match="length mismatch"):
        persistence.write_oof(str(tmp_path / "o.npz"), cv.oof_proba, y, cv.oof_fold,
                              ["only_one"], cv.class_names)
    with pytest.raises(ValueError, match="class names"):
        persistence.write_oof(str(tmp_path / "o.npz"), cv.oof_proba, y, cv.oof_fold,
                              [f"c{i}" for i in range(len(y))], ["too", "few"])


def test_write_confusion_keeps_the_folds_apart(cv_and_data, tmp_path):
    cv, _X, _y = cv_and_data
    path = str(tmp_path / "conf.csv")
    persistence.write_confusion(path, cv.confusion, cv.class_names,
                                per_fold=cv.confusion_per_fold)
    rows = list(csv.reader(open(path)))
    assert rows[0] == ["fold", "true_label"] + [f"pred_{c}" for c in cv.class_names]
    body = rows[1:]
    k, n = cv.confusion_per_fold.shape[0], cv.n_classes
    assert len(body) == n * (1 + k)
    agg = np.array([[int(v) for v in r[2:]] for r in body if r[0] == "all"])
    assert np.array_equal(agg, cv.confusion)
    for f in range(k):
        got = np.array([[int(v) for v in r[2:]] for r in body if r[0] == str(f)])
        assert np.array_equal(got, cv.confusion_per_fold[f])


def test_write_confusion_omits_the_fold_rows_when_there_are_none(tmp_path):
    """The gaussian_arbor row comes from the untouched Patch-seq run_gaussian_cv, which has no
    per-fold matrices; that must degrade to an aggregate-only file, not crash."""
    path = str(tmp_path / "conf.csv")
    persistence.write_confusion(path, np.eye(2, dtype=int), ["a", "b"])
    rows = list(csv.reader(open(path)))
    assert len(rows) == 3 and {r[0] for r in rows[1:]} == {"all"}


# --- the grid-boundary tripwire --------------------------------------------------------------

def test_boundary_hits_names_the_fold_the_parameter_and_the_edge():
    hits = boundary_hits([{"clf__C": 10.0, "clf__l1_ratio": 0.5},
                          {"clf__C": 1.0, "clf__l1_ratio": 0.0}])
    assert hits == ["fold0:clf__C=10.0@max", "fold1:clf__l1_ratio=0.0@min"]


def test_boundary_hits_is_silent_on_foreign_best_params():
    """The finetune probe records {"mode": ..., "epochs": ...}; there is no grid to sit on."""
    assert boundary_hits([{"mode": "finetune", "epochs": 40}]) == []


def test_boundary_hits_defaults_to_the_shipped_grid():
    mid = {"clf__C": 1.0, "clf__l1_ratio": 0.5}
    assert boundary_hits([mid]) == []
    assert boundary_hits([mid], grid={"clf__C": [1.0, 2.0]}) == ["fold0:clf__C=1.0@min"]
    assert GRID["clf__C"] == [0.1, 1.0, 10.0]
