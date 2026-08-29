"""Signal probe for a path-encoding artifact.

Runs the benchmark's own classifier (`classifier.run_cv`, nested stratified CV, multinomial
elastic net) over the pooled per-cell latents in a `path_encodings.py` artifact, for each of the
three benchmark modalities. Emits `probe_metrics.csv` in the same schema as
`benchmark_metrics.csv` so rows drop straight into a comparison, plus confusion matrices and
UMAP embeddings.

This is a probe, not a benchmark row: `run_benchmark.py` and its EXTRACTORS table are
deliberately untouched. The encoding artifact it reads is treated as immutable -- outputs go to
a fresh version directory of their own.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import h5py
import numpy as np

from arborist_experiments.morphology_cv import classifier
from arborist_experiments.patchseq_morphology_benchmark import features
from arborist_experiments.morphology_cv.classifier import run_cv
from arborist_experiments.patchseq_morphology_benchmark.data_loading import build_cohort
from arborist_experiments.cells.plots import plot_confusion, plot_embedding
from arborist_experiments.experiment_harness import (
    log as _log,
    mirror_dir,
    next_version_dir as _next_version_dir,
    write_rows,
)


def _mirror_probe(src_dir: str, dst_dir: str) -> None:
    """Mirror into results/, prefixing the generic filenames.

    results/ is flat and shared, so an unprefixed desc.txt / run_log.txt / run_config.json
    would silently overwrite the encoding run's own. The scratch copy keeps the plain names
    that the capsule's versioning convention expects.
    """
    mirror_dir(src_dir, dst_dir, prefix=f"{REP_NAME}_probe_",
               generic_names=GENERIC_NAMES)


def load_pooled(artifact: str):
    with h5py.File(artifact, "r") as f:
        X = f["pooled/X"][:]
        ids = list(f["pooled/specimen_ids"].asstr()[:])
        names = list(f["pooled/feature_names"].asstr()[:])
        attrs = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in f.attrs.items()}
    return X, ids, names, attrs


def modality_columns(names: list[str], modality: str) -> np.ndarray:
    """Column indices for a benchmark modality.

    "full" is every latent statistic; "axon"/"dendrite" keep only that arbor group's
    mean+std block, matching how the benchmark's extractors restrict by structure type.
    """
    if modality == "full":
        return np.arange(len(names))
    return np.flatnonzero(np.array([f"_{modality}_" in n for n in names]))


def align_to_cohort(X, ids, cohort):
    """Rows in cohort order, plus the remapped label vector.

    Rows are matched by specimen_id rather than trusting position, so a probe still lines up
    if the artifact was written from a different cohort ordering.
    """
    row_of = {sid: i for i, sid in enumerate(ids)}
    keep = [c.specimen_id for c in cohort.cells if c.specimen_id in row_of]
    missing = [c.specimen_id for c in cohort.cells if c.specimen_id not in row_of]
    idx = np.array([row_of[s] for s in keep], dtype=np.int64)
    id_to_label = {c.specimen_id: int(l) for c, l in zip(cohort.cells, cohort.labels)}
    y = np.array([id_to_label[s] for s in keep], dtype=np.int64)
    return X[idx], y, keep, missing


def main(args):
    out_dir = _next_version_dir(ARCHIVE_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fh = open(os.path.join(out_dir, "run_log.txt"), "w")
    _log(fh, f"Durable output dir: {out_dir} (mirrored to {RESULTS_DIR})")
    _log(fh, f"Cores: {os.cpu_count()}  extract workers: {features.N_WORKERS}  "
             f"cv jobs: {classifier.N_JOBS}")
    _log(fh, f"Artifact: {args.artifact}")

    X_all, ids, names, attrs = load_pooled(args.artifact)
    _log(fh, f"  pooled X {X_all.shape}, {len(ids)} cells, {len(names)} features")
    _log(fh, f"  encoder: {os.path.basename(str(attrs.get('checkpoint', '?')))}")
    _log(fh, f"  node_spacing={attrs.get('node_spacing')} "
             f"segment_len={attrs.get('segment_len')} min_points={attrs.get('min_points')}")

    cohort = build_cohort(met_path=MET_PATH, swc_path=SWC_PATH, min_class_size=MIN_CLASS_SIZE)
    X_all, y, kept_ids, missing = align_to_cohort(X_all, ids, cohort)
    _log(fh, f"Cohort: n_cells={len(cohort)} n_classes={len(cohort.label_names)}")
    _log(fh, f"  matched to artifact: {len(kept_ids)}; missing from artifact: {len(missing)}")
    for sid in missing[:10]:
        _log(fh, f"    missing {sid}")
    if len(missing) > 10:
        _log(fh, f"    ...and {len(missing) - 10} more.")

    classes_in_y = np.unique(y)
    class_names = [cohort.label_names[i] for i in classes_in_y]
    y = np.array([np.where(classes_in_y == v)[0][0] for v in y], dtype=np.int64)
    counts = np.bincount(y, minlength=len(class_names))
    for name, n in zip(class_names, counts):
        _log(fh, f"  class {name}: {n}")
    _log(fh, f"  majority-class floor: {counts.max() / counts.sum():.4f}")

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({
            "artifact": os.path.abspath(args.artifact),
            "artifact_attrs": {k: str(v) for k, v in attrs.items()},
            "met_path": MET_PATH, "swc_path": SWC_PATH,
            "min_class_size": MIN_CLASS_SIZE, "seed": SEED,
            "outer_k": OUTER_K, "inner_k": INNER_K,
            "modalities": MODALITIES, "n_cells": len(kept_ids),
            "classes": class_names, "label": "MET.meta/subclass",
            "coordinates": "swc_upright",
        }, f, indent=2, sort_keys=True)
    _mirror_probe(out_dir, RESULTS_DIR)

    csv_path = os.path.join(out_dir, "probe_metrics.csv")
    summary = {}
    with open(csv_path, "w", newline="") as csv_fh:
        writer = csv.writer(csv_fh)
        writer.writerow(["feature_rep", "modality", "metric", "mean", "std",
                         "n_cells", "n_classes", "per_fold"])
        for mod in MODALITIES:
            cols = modality_columns(names, mod)
            X = X_all[:, cols]
            n_zero_rows = int((~X.any(axis=1)).sum())
            _log(fh, f"--- {REP_NAME} x {mod} ---")
            _log(fh, f"  {X.shape[1]} features, {n_zero_rows} cells with an all-zero block "
                     f"(kept as true zeros, matching radial_arbors)")

            t0 = time.time()
            cv = run_cv(X, y, class_names=class_names,
                        outer_k=OUTER_K, inner_k=INNER_K, seed=SEED)
            _log(fh, f"  cv: outer={OUTER_K} inner={INNER_K} in {time.time() - t0:.1f}s")
            for metric, vals in cv.metrics.items():
                _log(fh, f"    {metric:9s} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
            _log(fh, f"  best_params: {cv.best_params}")
            summary[mod] = {k: (float(np.mean(v)), float(np.std(v)))
                            for k, v in cv.metrics.items()}

            write_rows(writer, REP_NAME, mod, cv, csv_fh=csv_fh)
            stem = f"{REP_NAME}_{mod}"
            plot_confusion(cv.confusion, class_names,
                           os.path.join(out_dir, f"confusion_{stem}.png"),
                           title=f"{REP_NAME} / {mod} (n={cv.n_samples})")
            try:
                plot_embedding(X, y, class_names,
                               os.path.join(out_dir, f"embedding_{stem}.png"),
                               method="umap", title=f"{REP_NAME} / {mod}")
            except Exception as e:
                _log(fh, f"  embedding failed: {type(e).__name__}: {e}")
            _mirror_probe(out_dir, RESULTS_DIR)

    lines = [f"Signal probe of {os.path.abspath(args.artifact)}", ""]
    lines += ["* Probe parameters",
              f"    representation = {REP_NAME!r}",
              f"    modalities = {MODALITIES!r}",
              f"    outer_k = {OUTER_K}", f"    inner_k = {INNER_K}",
              f"    seed = {SEED}", f"    min_class_size = {MIN_CLASS_SIZE}",
              f"    n_cells = {len(kept_ids)}", f"    n_classes = {len(class_names)}",
              f"    majority_class_floor = {counts.max() / counts.sum():.4f}", ""]
    lines += ["* Encoder (from artifact attrs)"]
    lines += [f"    {k} = {v}" for k, v in sorted(attrs.items())]
    lines += ["", "* Results (mean +/- std over outer folds)"]
    for mod, mets in summary.items():
        lines.append(f"    {mod}:")
        for metric, (m, s) in mets.items():
            lines.append(f"        {metric:9s} {m:.4f} +/- {s:.4f}")
    with open(os.path.join(out_dir, "desc.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    _log(fh, f"Done. Durable copy: {out_dir}")
    fh.close()
    _mirror_probe(out_dir, RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe a path-encoding artifact with the "
                                                 "benchmark classifier.")
    parser.add_argument("--artifact", default="/root/capsule/scratch/path_encodings/v1/latents.h5",
                        help="Path to a latents.h5 written by path_encodings.py.")
    args = parser.parse_args()

    MET_PATH = "/root/capsule/data/patchseq_MET/patchseq_MET.hdf5"
    SWC_PATH = "/root/capsule/data/patchseq_SWC/patchseq_SWC.hdf5"
    RESULTS_DIR = "/root/capsule/results"
    ARCHIVE_DIR = "/root/capsule/scratch/path_encoding_probes"

    MIN_CLASS_SIZE = 25
    SEED = 0
    OUTER_K = 5
    INNER_K = 3

    REP_NAME = "path_encodings"
    MODALITIES = ["full", "axon", "dendrite"]
    GENERIC_NAMES = {"desc.txt", "run_log.txt", "run_config.json"}

    main(args)
