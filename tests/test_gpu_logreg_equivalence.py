"""The GPU FISTA solver must minimize sklearn's objective, not merely a similar one.

If it does not, every MICrONS number stops being comparable to the Patch-seq baseline, so
these tests gate the large runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from arborist_experiments.global_classification.cv.classifier import GRID
from arborist_experiments.global_classification.cv.gpu_logreg import (
    TorchElasticNetLogReg,
    elasticnet_objective,
)

GRID_POINTS = [(c, l) for c in GRID["clf__C"] for l in GRID["clf__l1_ratio"]]


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=600, n_features=40, n_informative=25, n_redundant=5,
        n_classes=8, n_clusters_per_class=1, class_sep=1.5, random_state=0,
    )
    X = (X - X.mean(0)) / X.std(0)
    return X, y


def _saga(X, y, C, l1_ratio):
    m = LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio,
                           max_iter=20000, tol=1e-8, random_state=0)
    return m.fit(X, y)


@pytest.mark.parametrize("C,l1_ratio", GRID_POINTS)
def test_objective_at_least_as_good_as_saga(data, C, l1_ratio):
    X, y = data
    ref = _saga(X, y, C, l1_ratio)
    gpu = TorchElasticNetLogReg(C=C, l1_ratio=l1_ratio, dtype="float64",
                                device="cpu", random_state=0).fit(X, y)

    f_ref = elasticnet_objective(X, y, ref.coef_, ref.intercept_, C, l1_ratio)
    f_gpu = elasticnet_objective(X, y, gpu.coef_, gpu.intercept_, C, l1_ratio)

    # FISTA must not be worse than saga at the same objective, up to a relative slack.
    assert f_gpu <= f_ref + 1e-6 * abs(f_ref), f"FISTA {f_gpu} worse than saga {f_ref}"
    assert gpu.converged_, f"did not converge in {gpu.n_iter_} iters"
    assert np.isclose(f_gpu, gpu.objective_, rtol=1e-9)


@pytest.mark.parametrize("C,l1_ratio", GRID_POINTS)
def test_predictions_and_logloss_match_saga(data, C, l1_ratio):
    X, y = data
    ref = _saga(X, y, C, l1_ratio)
    gpu = TorchElasticNetLogReg(C=C, l1_ratio=l1_ratio, dtype="float64",
                                device="cpu", random_state=0).fit(X, y)

    assert abs(log_loss(y, ref.predict_proba(X)) - log_loss(y, gpu.predict_proba(X))) < 1e-4
    assert (ref.predict(X) == gpu.predict(X)).mean() > 0.995


@pytest.mark.parametrize("C,l1_ratio", [(c, l) for c, l in GRID_POINTS if l < 1.0])
def test_coefficients_match_saga_where_minimizer_is_unique(data, C, l1_ratio):
    """Coefficients are only well-defined to compare when l1_ratio < 1.

    With l1_ratio == 1 the objective is convex but not strictly convex, so the minimizer is a
    set rather than a point: at C=0.1 saga and FISTA agree on the objective to 3.6e-10 and on
    every single prediction, while one coefficient differs by 0.024. Asserting coefficient
    equality there would be testing an ill-posed quantity. The (1-l1_ratio)*||W||^2 term makes
    the objective strictly convex for l1_ratio < 1, so here the minimizer is unique.
    """
    X, y = data
    ref = _saga(X, y, C, l1_ratio)
    gpu = TorchElasticNetLogReg(C=C, l1_ratio=l1_ratio, dtype="float64",
                                device="cpu", random_state=0).fit(X, y)
    assert np.max(np.abs(ref.coef_ - gpu.coef_)) < 1e-3


@pytest.mark.parametrize("C,l1_ratio", GRID_POINTS)
def test_l1_support_matches_saga(data, C, l1_ratio):
    """A solver that gets the objective right but the sparsity pattern wrong is still wrong."""
    X, y = data
    ref = _saga(X, y, C, l1_ratio)
    gpu = TorchElasticNetLogReg(C=C, l1_ratio=l1_ratio, dtype="float64",
                                device="cpu", random_state=0).fit(X, y)
    tol = 1e-6
    agree = ((np.abs(ref.coef_) > tol) == (np.abs(gpu.coef_) > tol)).mean()
    assert agree > 0.97, f"support agreement only {agree:.3f}"


@pytest.mark.parametrize("C,l1_ratio", [(0.1, 0.5), (10.0, 0.5), (1.0, 1.0)])
def test_fp32_matches_fp64(data, C, l1_ratio):
    X, y = data
    f64 = TorchElasticNetLogReg(C=C, l1_ratio=l1_ratio, dtype="float64",
                                device="cpu", random_state=0).fit(X, y)
    f32 = TorchElasticNetLogReg(C=C, l1_ratio=l1_ratio, dtype="float32",
                                device="cpu", random_state=0).fit(X, y)
    f_64 = elasticnet_objective(X, y, f64.coef_, f64.intercept_, C, l1_ratio)
    f_32 = elasticnet_objective(X, y, f32.coef_, f32.intercept_, C, l1_ratio)
    assert abs(f_32 - f_64) <= 1e-5 * abs(f_64), f"fp32 {f_32} vs fp64 {f_64}"
    assert (f64.predict(X) == f32.predict(X)).mean() > 0.995


def test_sklearn_api_compatibility(data):
    import pickle

    from sklearn.base import clone

    X, y = data
    est = TorchElasticNetLogReg(C=0.1, l1_ratio=0.5, dtype="float32", device="cpu")
    assert clone(est).get_params() == est.get_params()
    est.fit(X, y)
    round_trip = pickle.loads(pickle.dumps(est))
    assert np.allclose(round_trip.predict_proba(X), est.predict_proba(X))
    p = est.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert est.coef_.shape == (8, X.shape[1]) and est.intercept_.shape == (8,)


@pytest.mark.parametrize("C,l1_ratio", GRID_POINTS)
def test_stops_early_and_matches_the_full_iteration_budget(data, C, l1_ratio):
    """The solver must terminate on its own, not on max_iter.

    The proximal residual has an fp32 noise floor that varies by problem (measured between
    1e-4 and below 1e-5 relative), so tol alone never fires and every fit ran the full 20000
    iterations -- 25x slower for an answer identical to 8 significant figures. This asserts
    both halves: that it stops early, and that stopping early costs nothing.
    """
    X, y = data
    ref = TorchElasticNetLogReg(C=C, l1_ratio=l1_ratio, random_state=0,
                                obj_tol=0.0, device="cpu").fit(X, y)
    est = TorchElasticNetLogReg(C=C, l1_ratio=l1_ratio, random_state=0, device="cpu").fit(X, y)

    assert est.stop_reason_ != "max_iter"
    assert est.converged_
    assert est.n_iter_ < ref.n_iter_
    rel = (est.objective_ - ref.objective_) / abs(ref.objective_)
    assert rel <= 1e-7, f"early stop cost {rel:.2e} of the objective"
    assert (est.predict(X) == ref.predict(X)).all()
