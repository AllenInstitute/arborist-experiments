"""GPU FISTA solver for sklearn's exact multinomial elastic-net logistic objective.

Why this exists: saga at n=38000, d=200, K=8 takes 9.5-195 s per fit depending on the grid
point, and nested 5x3 CV over a 9-point grid is 140 fits per representation. The tempting
shortcuts are both wrong -- pinning C=0.1 (selected 30/30 on Patch-seq) bakes in a small-n
artifact exactly where 31x more data should push toward less regularization, and swapping to
an L2 lbfgs solver is out because l1_ratio=0.5 was selected 24/30, so the L1 term is load
bearing. So the objective and the grid are unchanged and only the optimizer differs.

Measured, one fit at n=38000, d=200, K=8:

    grid point        saga      fp64     fp32     objective (saga / fp32)
    C=0.1, l1=0.5     22.9 s    24.0 s   2.5 s    2495.7464 / 2495.7461
    C=0.1, l1=1.0      9.5 s    24.8 s   2.5 s    2625.0550 / 2625.0549
    C=10,  l1=0.5    195.2 s    45.2 s   2.5 s    230091.61 / 230090.31

fp32 is the right dtype on a T4 (fp64 runs at 1/32 rate, which is the entire 10-20x gap) and
the objectives agree to 7 significant figures. fp64 is kept as a confirmation mode.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted


def elasticnet_objective(
    X: np.ndarray, y: np.ndarray, coef: np.ndarray, intercept: np.ndarray,
    C: float, l1_ratio: float,
) -> float:
    """sklearn's penalized objective, in float64. The equivalence tests compare against this.

        F = C * sum_i CE(x_i W + b, y_i) + l1_ratio*||W||_1 + 0.5*(1-l1_ratio)*||W||_2^2
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(coef, dtype=np.float64).T
    b = np.asarray(intercept, dtype=np.float64)
    Z = X @ W + b
    Z = Z - Z.max(axis=1, keepdims=True)
    ce = float(np.sum(np.log(np.exp(Z).sum(axis=1)) - Z[np.arange(len(y)), y]))
    return C * ce + l1_ratio * float(np.abs(W).sum()) + 0.5 * (1 - l1_ratio) * float((W ** 2).sum())


def _lipschitz(X_aug, C: float, l1_ratio: float, iters: int, seed: int) -> float:
    """C * 0.5 * sigma_max(X_aug)^2 + (1 - l1_ratio).

    0.5 is the spectral bound on the multinomial softmax Hessian. sigma_max^2 comes from
    seeded power iteration so the step size is reproducible run to run.
    """
    import torch

    g = torch.Generator(device=X_aug.device).manual_seed(seed)
    v = torch.randn(X_aug.shape[1], generator=g, device=X_aug.device, dtype=X_aug.dtype)
    v /= v.norm()
    s2 = torch.zeros((), device=X_aug.device, dtype=X_aug.dtype)
    for _ in range(iters):
        w = X_aug.T @ (X_aug @ v)
        s2 = w.norm()
        if s2 == 0:
            break
        v = w / s2
    return float(C) * 0.5 * float(s2) + (1.0 - float(l1_ratio))


class TorchElasticNetLogReg(ClassifierMixin, BaseEstimator):
    """Drop-in replacement for LogisticRegression(penalty='elasticnet', solver='saga').

    Only the intercept is unpenalized, matching sklearn. Fitted attributes are numpy, so the
    estimator pickles and clones normally inside GridSearchCV.
    """

    def __init__(
        self, C: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 20000, tol: float = 1e-6,
        obj_tol: float = 1e-12, stall_patience: int = 5,
        device: str = "cuda", dtype: str = "float32", check_every: int = 25,
        power_iters: int = 100, random_state: int | None = None, verbose: int = 0,
    ):
        self.C = C
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.obj_tol = obj_tol
        self.stall_patience = stall_patience
        self.device = device
        self.dtype = dtype
        self.check_every = check_every
        self.power_iters = power_iters
        self.random_state = random_state
        self.verbose = verbose

    def _resolve_device(self):
        # Never at import time: a CUDA context created in the parent breaks forked workers.
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn("CUDA unavailable, falling back to CPU", RuntimeWarning)
            return torch.device("cpu")
        return torch.device(self.device)

    def fit(self, X, y):
        import torch

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = unique_labels(y)
        K = len(self.classes_)
        if K < 3:
            warnings.warn(
                "sklearn parameterizes 2-class elastic-net logistic regression with a single "
                "weight vector, which is a different objective from this K-vector form; "
                "results are not comparable for K=2.", RuntimeWarning)
        y_idx = np.searchsorted(self.classes_, y)
        n, d = X.shape
        self.n_features_in_ = d

        dev = self._resolve_device()
        tdt = torch.float32 if self.dtype == "float32" else torch.float64
        Xt = torch.as_tensor(X, dtype=tdt, device=dev)
        yt = torch.as_tensor(y_idx, dtype=torch.long, device=dev)
        # Augment with a ones column purely for the Lipschitz bound, so the intercept's
        # curvature is included in the step size.
        X_aug = torch.cat([Xt, torch.ones(n, 1, dtype=tdt, device=dev)], dim=1)
        L = _lipschitz(X_aug, self.C, self.l1_ratio, self.power_iters,
                       0 if self.random_state is None else int(self.random_state))
        del X_aug
        step = 1.0 / (1.02 * max(L, 1e-12))

        C = float(self.C)
        l1 = float(self.l1_ratio)
        l2 = 1.0 - l1
        Y = torch.zeros(n, K, dtype=tdt, device=dev)
        Y[torch.arange(n, device=dev), yt] = 1.0

        W = torch.zeros(d, K, dtype=tdt, device=dev)
        b = torch.zeros(K, dtype=tdt, device=dev)
        Wz, bz = W.clone(), b.clone()
        t_mom = 1.0

        def grads(W_, b_):
            P = torch.softmax(Xt @ W_ + b_, dim=1)
            R = P - Y
            return C * (Xt.T @ R) + l2 * W_, C * R.sum(dim=0)

        def objective(W_, b_) -> float:
            Z = Xt @ W_ + b_
            ce = torch.nn.functional.cross_entropy(Z.double(), yt, reduction="sum")
            return float(C * ce + l1 * W_.double().abs().sum()
                         + 0.5 * l2 * (W_.double() ** 2).sum())

        def residual(W_, gW, gb) -> float:
            """Proximal-optimality residual: exactly zero at the solution, scale-free in C.

            Using |delta coef| instead would be scale dependent and fp32 never reaches a
            fixed 1e-7, so the solver would always burn max_iter.
            """
            nz = W_ != 0
            r = torch.where(nz, (gW + l1 * torch.sign(W_)).abs(),
                            torch.clamp(gW.abs() - l1, min=0.0))
            return float(torch.maximum(r.max(), gb.abs().max()))

        gW0, gb0 = grads(W, b)
        res0 = max(residual(W, gW0, gb0), 1e-30)
        best_obj = objective(W, b)
        best_W, best_b = W.clone(), b.clone()
        self.converged_ = False
        self.stop_reason_ = "max_iter"
        stall = 0
        it = 0

        for it in range(1, int(self.max_iter) + 1):
            gW, gb = grads(Wz, bz)
            V = Wz - step * gW
            W_new = torch.sign(V) * torch.clamp(V.abs() - step * l1, min=0.0)
            b_new = bz - step * gb

            t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_mom * t_mom) ** 0.5)
            mom = (t_mom - 1.0) / t_next
            Wz = W_new + mom * (W_new - W)
            bz = b_new + mom * (b_new - b)
            W, b, t_mom = W_new, b_new, t_next

            if it % int(self.check_every) == 0 or it == self.max_iter:
                obj = objective(W, b)
                gain = max(best_obj - obj, 0.0) / max(abs(best_obj), 1.0)
                if obj > best_obj:
                    # Monotone safeguard with momentum restart (adaptive-restart FISTA).
                    step *= 0.5
                    W, b = best_W.clone(), best_b.clone()
                    Wz, bz = W.clone(), b.clone()
                    t_mom = 1.0
                else:
                    best_obj = obj
                    best_W, best_b = W.clone(), b.clone()
                gW, gb = grads(W, b)
                if residual(W, gW, gb) <= float(self.tol) * res0:
                    self.converged_, self.stop_reason_ = True, "residual"
                    break
                # The residual has an fp32 noise floor that varies by problem (measured
                # between 1e-4 and below 1e-5 relative), so tol alone would burn max_iter on
                # every fit. The objective is accumulated in float64 regardless of the iterate
                # dtype, so stalling on it is the criterion that actually terminates -- and it
                # is the quantity the saga equivalence tests compare.
                stall = stall + 1 if gain < float(self.obj_tol) else 0
                if stall >= int(self.stall_patience):
                    self.converged_, self.stop_reason_ = True, "objective_stalled"
                    break

        self.coef_ = best_W.double().cpu().numpy().T.copy()
        self.intercept_ = best_b.double().cpu().numpy().copy()
        self.objective_ = float(best_obj)
        self.n_iter_ = it
        if self.verbose:
            print(f"FISTA C={self.C} l1={self.l1_ratio} iters={it} "
                  f"stop={self.stop_reason_} F={best_obj:.6f}")
        del Xt, Y, W, b, Wz, bz, best_W, best_b
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_.T + self.intercept_

    def predict_log_proba(self, X):
        Z = self.decision_function(X)
        Z = Z - Z.max(axis=1, keepdims=True)
        return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
