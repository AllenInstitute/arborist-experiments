"""On-disk cache of extracted design matrices.

Extraction is the expensive half of a benchmark run and the CV that consumes it is the cheap half
(sholl alone is ~45 min at 48k cells against 5-10 min for the whole GPU nested CV). Every question
asked of a finished run afterwards -- a per-class recall, a coefficient, a cohort split -- needed
the matrix back, and the matrix was never written down, so re-asking cost another extraction pass.

The key is the cell membership itself (a digest of the sorted specimen ids) together with the
geometry constants and `real_radius`, not the cohort-building arguments that produced them: two
different routes to the same cell set yield the same features, and `label_column` / `permute_labels`
do not enter the features at all. Constants are digested as one blob, so changing any of them
invalidates every entry -- they are frozen in practice, and a partially-stale cache is not worth
the bookkeeping.

Subsampled runs get their own entry rather than reusing a slice of the full one. That is deliberate:
`extract_morphometrics` drops zero-variance columns and imputes NaNs with cohort column means, so a
subset of a cached matrix is not the matrix that subset cohort would have produced.

Stored float64, uncompressed. float32 would halve the ~700 MB but the pipeline's scaler and PCA
would then see different inputs than a fresh run, and bit-exact reproduction is what makes the
cache safe to trust.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from arborist_experiments.global_classification.extract import BinningConfig, FeatureResult


CACHE_ROOT = "/root/capsule/scratch/features"


def constants_digest(cfg: BinningConfig) -> str:
    """Digest of every binning constant a feature value depends on.

    The payload is built exactly as it was when it read live module globals -- same five keys,
    same coercions, same sort -- because the published `em_exc` cache directory is named after
    its output. `em_exc` must digest to 2a91b6b7; `tests/test_binning_configs.py` pins that.

    `cfg.name` is not in the payload: it is a label, not a constant a feature value depends on.
    """
    payload = json.dumps({
        "density_config": cfg.density_config,
        "sholl_radii": [float(r) for r in cfg.sholl_radii],
        "persim_kwargs": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in cfg.persim_kwargs.items()},
        "r_edges": cfg.r_edges.tolist(),
        "z_edges": cfg.z_edges.tolist(),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def cohort_key(cohort, source: str, real_radius: bool, cfg: BinningConfig) -> str:
    ids = sorted(c.specimen_id for c in cohort.cells)
    id_hash = hashlib.sha1("\n".join(ids).encode()).hexdigest()[:12]
    return (f"{source}_n{len(ids)}_{id_hash}"
            f"_r{int(bool(real_radius))}_c{constants_digest(cfg)}")


def cache_dir(key: str, root: str = CACHE_ROOT) -> str:
    return os.path.join(root, key)


def entry_path(key: str, rep: str, mod: str, root: str = CACHE_ROOT) -> str:
    return os.path.join(cache_dir(key, root), f"{rep}_{mod}.npz")


def load(key: str, rep: str, mod: str, cohort=None, root: str = CACHE_ROOT):
    """The cached FeatureResult, or None if absent. Raises on a corrupt or foreign entry."""
    path = entry_path(key, rep, mod, root)
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        fr = FeatureResult(
            specimen_ids=[str(s) for s in z["specimen_ids"]],
            X=z["X"],
            feature_names=[str(s) for s in z["feature_names"]],
            skipped=[(str(a), str(b)) for a, b in z["skipped"].reshape(-1, 2)],
        )
    if fr.X.shape != (len(fr.specimen_ids), len(fr.feature_names)):
        raise ValueError(f"corrupt feature cache {path}: X {fr.X.shape} vs "
                         f"{len(fr.specimen_ids)} ids x {len(fr.feature_names)} names")
    if cohort is not None:
        foreign = set(fr.specimen_ids) - {c.specimen_id for c in cohort.cells}
        if foreign:
            raise ValueError(f"feature cache {path} holds {len(foreign)} cells not in this "
                             f"cohort (e.g. {sorted(foreign)[:3]}) -- key collision")
    return fr


def save(key: str, rep: str, mod: str, fr: FeatureResult, meta: dict | None = None,
         root: str = CACHE_ROOT) -> str:
    d = cache_dir(key, root)
    os.makedirs(d, exist_ok=True)
    manifest = os.path.join(d, "cohort.json")
    if meta is not None and not os.path.exists(manifest):
        with open(manifest, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True, default=str)
    path = entry_path(key, rep, mod, root)
    tmp = os.path.join(d, f".{rep}_{mod}.partial.npz")
    np.savez(tmp,
             X=np.ascontiguousarray(fr.X),
             specimen_ids=np.array(fr.specimen_ids, dtype=str),
             feature_names=np.array(fr.feature_names, dtype=str),
             skipped=np.array(fr.skipped, dtype=str).reshape(-1, 2))
    os.replace(tmp, path)
    return path
