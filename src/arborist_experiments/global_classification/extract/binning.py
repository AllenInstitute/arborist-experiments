"""The per-cohort binning constants, as a value rather than as module state.

Every handcrafted feature in this package is a histogram of something, and every histogram needs
edges derived from the cohort it bins: the Patch-seq envelope is a ~200 um slice, MICrONS
excitatory is a laterally compact volume, MICrONS inhibitory has axons reaching 350 um. Those
three sets of numbers are the *only* thing that ever differed between the two copies of
`features.py` and `radial_arbor_features.py` this package used to carry, so they are what gets
passed in and the code is what gets shared.

The predecessor of this module rebound `features.DENSITY_CONFIG` and
`radial_arbor_features.R_EDGES` in place, on the argument that the extractors run inside
`ProcessPoolExecutor` workers and only fork-inherited globals reach them. That argument was
wrong: `utils.cells.run_jobs` passes its `*job_args` through `ex.map`, which pickles them to the
workers, so a config travels the same route `modality` and `axis` always did. What the rebinding
cost was real -- a `constants_digest()` that had to read live modules, an
`assert_exc_constants()` to catch a stray import-time activation, and two cohorts that could not
be extracted in one process.

`sholl_max_radius`, `sholl_num_steps`, `resample_step` and `cortical_axis` are deliberately NOT
here: they are identical across all three cohorts, and they are outside the set
`feature_cache.constants_digest` hashes. Moving them in would change that digest and orphan the
published `em_exc` cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BinningConfig:
    """Everything a feature value depends on that varies by cohort.

    `name` is for logs and desc.txt only -- it is not part of the digest, so renaming a cohort
    does not invalidate its cached matrices.
    """

    name: str
    density_config: dict
    sholl_radii: list[float]
    persim_kwargs: dict
    r_edges: np.ndarray
    z_edges: np.ndarray
    disk_areas: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # Derived, not passed: the radial histogram divides by annulus area to read as a density,
        # and a caller who set r_edges but forgot disk_areas would get a histogram normalized by
        # another cohort's discs. That was a real failure mode of the rebinding version.
        areas = np.pi * (self.r_edges[1:] ** 2 - self.r_edges[:-1] ** 2)
        object.__setattr__(self, "disk_areas", areas / areas.max())
