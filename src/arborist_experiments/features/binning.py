"""The per-cohort binning constants, as a value rather than as module state.

Every handcrafted feature in this package is a histogram of something, and every histogram needs
edges derived from the cohort it bins: a thin tissue slice, a laterally compact volume and a
population with long-range axons each need a different envelope. Those numbers are passed in, so one
copy of the code serves every cohort.

A config is an ordinary argument rather than module state: `utils.cells.run_jobs` passes its
`*job_args` through `ex.map`, which pickles them to the workers, so a config reaches them the same
way `modality` and `axis` do, and two cohorts can be extracted in one process.

`sholl_max_radius`, `sholl_num_steps`, `resample_step` and `cortical_axis` are deliberately NOT
here: no cohort varies them, and they are outside the set `feature_cache.constants_digest` hashes,
so adding them would change every existing cache's identity.

This is a COHORT descriptor, not a feature's parameter list. No extractor takes one: each declares
the constants it actually reads, and a driver holding one of these unpacks it at the call site. The
bundle exists for the digest -- cache identity is `constants_digest` of exactly these five fields, so
having one object that *is* the set of value-affecting constants is what stops a call site from
varying a constant the digest does not cover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinningConfig:
    """Everything a feature value depends on that varies by cohort.

    `name` is for logs only -- it is not part of the digest, so renaming a cohort
    does not invalidate its cached matrices.
    """

    name: str
    density_config: dict
    sholl_radii: list[float]
    persim_kwargs: dict
    r_edges: np.ndarray
    z_edges: np.ndarray

    def density_kwargs(self) -> dict:
        """`density_config` as `extract_density`'s parameters: `**cfg.density_kwargs()`.

        The dict stays the stored form because `feature_cache.constants_digest` hashes it, so its
        keys are part of cache identity and not free to change. `extract_density` takes explicit
        parameters instead, and this is the one place the two spellings meet.

        Unknown keys raise. morphopy reaches every key through `config_params.get(name, default)`, so
        a misspelled key such as `bin_sizes` passed straight through would become a silent default;
        here a cohort config that carries a key nothing maps is a startup error.
        """
        renamed = {"bin_size": "bin_size", "density": "density", "smooth": "smooth",
                   "sigma": "sigma", "distance": "resample_distance"}
        out = {}
        for ax in ("x", "y", "z"):
            lo, hi = self.density_config.get(f"r_min_{ax}"), self.density_config.get(f"r_max_{ax}")
            if (lo is None) != (hi is None):
                raise ValueError(f"{self.name}: r_min_{ax}/r_max_{ax} must be given together")
            if lo is not None:
                out[f"{ax}_range"] = (float(lo), float(hi))
        for key, value in self.density_config.items():
            if key in renamed:
                out[renamed[key]] = value
            elif not key.startswith(("r_min_", "r_max_", "n_bins_")):
                raise ValueError(
                    f"{self.name}: density_config key {key!r} maps to no extract_density parameter")
        # Per-axis in the dict, per-projection in the signature, so it cannot be forwarded blindly.
        if any(k.startswith("n_bins_") for k in self.density_config):
            raise NotImplementedError(
                f"{self.name}: n_bins_* is a per-axis override and extract_density takes n_bins as the "
                f"projection's pair -- pass it at the call site. No cohort config uses one.")
        return out

    def persistence_kwargs(self) -> dict:
        """`persim_kwargs` as `extract_persistence_image`'s parameters.

        A rename-free translation, unlike `density_kwargs`: the stored keys already *are*
        `PersistenceImager`'s parameter names, so this only checks them. It is still worth having,
        because it is where a config carrying a key persim does not take fails at startup rather than
        inside a worker partway through an extraction, and because it keeps
        `**cfg.persistence_kwargs()` the one call shape a driver writes for either extractor.

        The dict stays the stored form for the same reason `density_config` does --
        `feature_cache.constants_digest` hashes it, tuple-to-list coerced, into the cache identity.
        """
        allowed = ("birth_range", "pers_range", "pixel_size",
                   "weight", "weight_params", "kernel", "kernel_params")
        unknown = sorted(set(self.persim_kwargs) - set(allowed))
        if unknown:
            raise ValueError(
                f"{self.name}: persim_kwargs {unknown} are not PersistenceImager parameters, so "
                f"extract_persistence_image has none to map them to")
        missing = [k for k in ("birth_range", "pers_range", "pixel_size")
                   if k not in self.persim_kwargs]
        if missing:
            raise ValueError(
                f"{self.name}: persim_kwargs is missing {missing} -- persim's own defaults are a "
                f"1 um image, which no cohort means")
        return dict(self.persim_kwargs)
