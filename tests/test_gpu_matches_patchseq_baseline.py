"""The one check whose correct answer is already known.

The Patch-seq benchmark's `radial_arbors x full` row is 0.686 balanced accuracy at n=1556,
13 classes, seed 0. This runs the *identical* cohort and features through the MICrONS GPU
`run_cv` and demands the same number. Synthetic-data equivalence tests can miss a mistake in
the CV plumbing (fold seeds, PCA cap, inner scoring); this cannot.

Slow (~4 min) and needs the Patch-seq data plus a GPU, so it is skipped when either is absent.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

MET_PATH = "/root/capsule/data/patchseq_MET/patchseq_MET.hdf5"
SWC_PATH = "/root/capsule/data/patchseq_SWC/patchseq_SWC.hdf5"

PUBLISHED_BA = 0.6865        # scratch/benchmarks/v4, reproduced with the untouched driver
TOLERANCE = 0.01

pytestmark = pytest.mark.skipif(
    not (os.path.exists(MET_PATH) and os.path.exists(SWC_PATH)),
    reason="Patch-seq data not mounted",
)


@pytest.mark.parametrize("solver", ["gpu"])
def test_gpu_run_cv_reproduces_patchseq_radial_arbors_full(solver):
    from arborist_experiments.global_classification.cv.classifier import run_cv
    from arborist_experiments.global_classification.extract import extract_radial_arbors
    from arborist_experiments.global_classification.patchseq.binning import PATCHSEQ
    from arborist_experiments.global_classification.patchseq.data_loading import (
        build_cohort,
    )

    cohort = build_cohort(met_path=MET_PATH, swc_path=SWC_PATH, min_class_size=25)
    fr = extract_radial_arbors(cohort, "full", PATCHSEQ)
    assert len(fr.specimen_ids) == 1556 and fr.X.shape[1] == 328

    label_by_id = {c.specimen_id: int(l) for c, l in zip(cohort.cells, cohort.labels)}
    y = np.array([label_by_id[s] for s in fr.specimen_ids], dtype=np.int64)
    present = np.unique(y)
    y = np.searchsorted(present, y)
    class_names = [cohort.label_names[i] for i in present]

    cv = run_cv(fr.X, y, class_names=class_names, outer_k=5, inner_k=3, seed=0,
                solver=solver, dtype="float32")
    ba = float(np.mean(cv.metrics["balanced_accuracy"]))
    assert cv.n_samples == 1556 and cv.n_classes == 13
    assert abs(ba - PUBLISHED_BA) <= TOLERANCE, (
        f"{solver} gives {ba:.4f}, Patch-seq baseline is {PUBLISHED_BA:.4f}"
    )
