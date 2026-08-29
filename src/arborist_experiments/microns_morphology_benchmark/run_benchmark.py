"""MICrONS morphology benchmark driver.

Ports the Patch-seq feature-representation benchmark to the MICrONS EM cells, with the same
multinomial elastic-net logistic regression and the same nested stratified CV. Two cohorts,
selected by `--source` and never combined (their compartment sets almost perfectly separate
them, so a joint classifier would be reading the acquisition pipeline):

  em_exc       48042 cells, 8 coarse classes, dendrite only  (Phase A)
  microns_inh   7094 cells, 11 coarse classes, dendrite/axon/full  (Phase B)

Outputs go to a fresh version directory under /root/capsule/scratch/ and are mirrored into
/root/capsule/results/ after every rep x modality. results/ is wiped between sessions, so it
can never be the only copy.

One invocation per configuration -- the learning curve is separate --n_cells runs, not a loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from arborist_experiments.morphology_cv import persistence
from arborist_experiments.morphology_cv.classifier import GRID, run_cv
from arborist_experiments.patchseq_morphology_benchmark.gaussian_baseline import run_gaussian_cv
from arborist_experiments.cells.plots import (
    plot_confusion, plot_embedding,
)
from . import feature_cache
from . import features
from . import radial_arbor_features as raf
from .data_loading import (
    CACHE_PATH, INH_CACHE_PATH, INH_LABELS_PATH, LABELS_PATH, build_cohort_from_source,
    class_counts, soma_depth_features, subsample_per_class, subsample_proportional,
)
from .features import (
    extract_density_xy, extract_density_xz, extract_density_yz,
    extract_morphometrics, extract_persistence_image, extract_sholl,
)
from .inh_constants import apply_inh_constants
from .modality import MODALITY_TYPES, UNAVAILABLE, check_modality, patchseq_modality_types
from .radial_arbor_features import extract_radial_arbors
from arborist_experiments.experiment_harness import (
    log as _log,
    mirror_dir,
    next_version_dir as _next_version_dir,
    write_rows,
)


# Per-source defaults. `--cache`/`--labels` override, so a smoke cache can be pointed at.
SOURCE_DEFAULTS = {
    "em_exc":      {"cache": CACHE_PATH, "labels": LABELS_PATH,
                    "modalities": ["dendrite"]},
    "microns_inh": {"cache": INH_CACHE_PATH, "labels": INH_LABELS_PATH,
                    "modalities": ["dendrite", "axon", "full"]},
}


EXTRACTORS = {
    "radial_arbors":            ("morph", extract_radial_arbors),
    "density_xz":               ("morph", extract_density_xz),
    "density_yz":               ("morph", extract_density_yz),
    "density_xy":               ("morph", extract_density_xy),
    "persistence_image":        ("morph", extract_persistence_image),
    "morphometric_statistics":  ("morph", extract_morphometrics),
    "sholl":                    ("morph", extract_sholl),
    "gaussian_arbor":           ("generative", run_gaussian_cv),
    "soma_depth":               ("control", soma_depth_features),
}


def _align_features_and_labels(fr, cohort):
    id_to_label = {c.specimen_id: int(l) for c, l in zip(cohort.cells, cohort.labels)}
    y = np.array([id_to_label[sid] for sid in fr.specimen_ids], dtype=np.int64)
    return fr.X, y


def _mirror(src_dir: str, dst_dir: str) -> None:
    mirror_dir(src_dir, dst_dir, prefix="benchmark_", generic_names=GENERIC_NAMES)


def _write_run_config(out_dir: str, extra: dict) -> None:
    # Read the binning constants off the modules, not from import-time copies: an inhibitory run
    # rebinds them via apply_inh_constants, and a stale copy here would record the wrong ones.
    cfg = {
        "min_class_size": MIN_CLASS_SIZE, "seed": SEED,
        "density_config": features.DENSITY_CONFIG,
        "r_edges": raf.R_EDGES.tolist(), "z_edges": raf.Z_EDGES.tolist(),
        "grid": {k: v for k, v in GRID.items()},
    }
    cfg.update(extra)
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True, default=str)


def _write_desc(out_dir, cohort, rep_list, mod_list, outer_k, inner_k, args, n_classes) -> None:
    chance = 1.0 / n_classes
    counts = class_counts(cohort)
    axon_cells = sum(1 for c in cohort.cells if (c.types == 2).any())
    apical_cells = sum(1 for c in cohort.cells if (c.types == 4).any())
    lines = ["MICrONS EM morphology feature-representation benchmark.",
             "Port of scratch/benchmarks/v2 (Patch-seq, n=1556, 13 classes) to the MICrONS",
             f"{args.source} cohort. Same features, same classifier objective, same CV protocol.",
             ""]
    lines += ["* Run parameters",
              f"    source = {args.source}",
              f"    smoke = {bool(args.smoke)}",
              f"    representations = {rep_list!r}",
              f"    modalities = {mod_list!r}",
              f"    outer_k = {outer_k}", f"    inner_k = {inner_k}",
              f"    seed = {SEED}", f"    min_class_size = {MIN_CLASS_SIZE}",
              f"    label_column = {args.label_column}",
              f"    exc_only = {args.exc_only}   inh_only = {args.inh_only}",
              f"    real_radius = {args.real_radius}",
              f"    require_types = {args.require_types}",
              f"    permute_labels = {args.permute_labels}",
              f"    solver = {args.solver}   dtype = {args.dtype}",
              f"    n_cells requested = {args.n_cells or 'all'}",
              f"    max_per_class = {args.max_per_class or 'none'}",
              f"    n_cells realized = {len(cohort)}",
              f"    n_classes = {n_classes}   chance balanced accuracy = {chance:.4f}",
              f"    cache_path = {args.cache}",
              f"    labels_path = {args.labels}", ""]
    src = "all 49803 exc cells / 93.4M nodes" if args.source == "em_exc" else \
          "all 7824 inh cells / 36.2M nodes, union of the dendrite/axon/full envelopes"
    lines += [f"* Derived constants (derive_constants.py, {src})",
              f"    DENSITY_CONFIG = {features.DENSITY_CONFIG}",
              f"    R_EDGES = linspace({raf.R_EDGES[0]}, {raf.R_EDGES[-1]}, {len(raf.R_EDGES)})",
              f"    Z_EDGES = linspace({raf.Z_EDGES[0]}, {raf.Z_EDGES[-1]}, {len(raf.Z_EDGES)})",
              "    SHOLL_RADII, PERSIM_KWARGS unchanged from Patch-seq for comparability", ""]
    lines += ["* Modalities",
              f"    scored = {mod_list!r}",
              f"    defined = {sorted(MODALITY_TYPES)}",
              f"    unavailable on {args.source} = {sorted(UNAVAILABLE.get(args.source, {}))}",
              f"    compartment prevalence in this cohort: axon {axon_cells}/{len(cohort)} "
              f"({100 * axon_cells / max(1, len(cohort)):.2f}%), "
              f"apical {apical_cells}/{len(cohort)} "
              f"({100 * apical_cells / max(1, len(cohort)):.2f}%)"]
    for m, why in sorted(UNAVAILABLE.get(args.source, {}).items()):
        lines.append(f"    {m!r}: {why}")
    lines.append("")
    lines += ["* Classifier",
              f"    multinomial elastic-net logistic regression, solver={args.solver}",
              "    StandardScaler -> PCA(min(n/2, 200, n_features)) -> elastic-net logistic",
              f"    grid: C in {GRID['clf__C']}, l1_ratio in {GRID['clf__l1_ratio']}",
              "    inner-loop scoring: neg_log_loss",
              "    metrics: log_loss, accuracy, balanced_accuracy, f1_macro, mcc",
              "    no class_weight -- kept unset for comparability with the Patch-seq run,",
              f"    on a cohort imbalanced {max(counts.values()) / max(1, min(counts.values())):.0f}x",
              ""]
    lines += ["* Class counts"] + [f"    {k} = {v}" for k, v in sorted(counts.items())]
    lines += ["", "* Reading the results",
              "    precomputed_morphometrics (the strongest Patch-seq row, 0.830) has no MICrONS",
              "    counterpart and is absent, so the top row here is necessarily a different",
              "    representation.",
              "    Every morphology row should be read against the soma_depth control, not against",
              ("    chance: the excitatory coarse labels are laminar and soma depth alone orders"
               " them monotonically."
               if args.source == "em_exc" else
               "    chance: the inhibitory classes are only partly laminar (L1 and ChC are"
               " superficial,\n    DTC deep), so soma_depth is a weaker but still non-trivial"
               " baseline."),
              "", "* Results",
              "    benchmark_metrics.csv   one row per rep x modality x metric, with per_fold,",
              "                            best_params and grid_boundary as columns",
              "    oof_<rep>_<mod>.npz     out-of-fold proba / y / fold / specimen_ids /",
              "                            class_names -- the sufficient statistic; every metric,",
              "                            confusion matrix, per-class recall and cohort split in",
              "                            this directory is recoverable from it without a rerun",
              "    confusion_<rep>_<mod>.csv  fold=all is the aggregate the PNG shows, 0..k-1 the",
              "                            individual folds",
              f"    feature cache = {'disabled' if args.no_feature_cache else 'scratch/features/'}"]
    with open(os.path.join(out_dir, "desc.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_manifest(cohort, out_path: str) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["specimen_id", "label", "soma_to_pia_y", "soma_to_wm_y"])
        for i, (c, lbl) in enumerate(zip(cohort.cells, cohort.labels)):
            w.writerow([c.specimen_id, cohort.label_names[int(lbl)],
                        float(cohort.pia_y[i]), float(cohort.wm_y[i])])


def main(args):
    out_dir = (os.path.join(ARCHIVE_DIR, "smoke") if args.smoke
               else _next_version_dir(ARCHIVE_DIR))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fh = open(os.path.join(out_dir, "run_log.txt"), "w")
    _log(fh, f"Durable output dir: {out_dir} (mirrored to {RESULTS_DIR})")
    _log(fh, f"Source: {args.source}  cache: {args.cache}")
    _log(fh, f"Cores: {os.cpu_count()}  extract workers: {features.N_WORKERS}  "
             f"solver: {args.solver}/{args.dtype}")

    # Before any cohort is built and long before the first process pool is forked: the workers
    # inherit these module globals, so rebinding them later would silently produce a mix.
    if args.source == "microns_inh":
        apply_inh_constants(log=lambda m: _log(fh, m))

    _log(fh, "Building cohort...")
    if args.source == "em_exc":
        build_kwargs = dict(exc_only=args.exc_only)
    else:
        build_kwargs = dict(inh_only=args.inh_only,
                            require_types=tuple(args.require_types or ()) or None)
    cohort = build_cohort_from_source(
        args.source,
        cache_path=args.cache, labels_path=args.labels, label_column=args.label_column,
        min_class_size=MIN_CLASS_SIZE, real_radius=args.real_radius,
        log=lambda m: _log(fh, "  " + m), **build_kwargs,
    )

    if args.smoke:
        cohort = subsample_per_class(cohort, SMOKE_PER_CLASS, seed=SEED)
        rep_list, mod_list = SMOKE_REPS, SMOKE_MODALITIES
        outer_k, inner_k = SMOKE_OUTER_K, SMOKE_INNER_K
        _log(fh, f"Smoke cohort: {len(cohort)} cells")
    else:
        if args.max_per_class:
            # Balances the cohort instead of preserving the 26x prior -- a different question
            # from --n_cells, so the two are not meant to be combined.
            cohort = subsample_per_class(cohort, args.max_per_class, seed=SEED)
            _log(fh, f"capped at {args.max_per_class}/class: {len(cohort)} cells")
        if args.n_cells:
            cohort = subsample_proportional(cohort, args.n_cells, seed=SEED,
                                            min_class_size=MIN_CLASS_SIZE,
                                            log=lambda m: _log(fh, "  " + m))
        rep_list = args.representations or REPRESENTATIONS
        mod_list = args.modalities or SOURCE_DEFAULTS[args.source]["modalities"]
        outer_k, inner_k = OUTER_K, INNER_K

    # Fail here rather than 7000 identical times inside a pool, or -- worse -- succeeding with a
    # modality that is a duplicate of another row on this cohort.
    for mod in mod_list:
        check_modality(args.source, mod)

    if args.permute_labels:
        # Negative control: balanced accuracy must collapse to 1/K. Permuting after
        # subsampling keeps the class counts exactly as scored.
        rng = np.random.default_rng(SEED)
        cohort.labels = rng.permutation(cohort.labels)
        _log(fh, "PERMUTED LABELS -- this is a negative control run")

    n_classes = len(cohort.label_names)
    chance = 1.0 / n_classes
    _log(fh, f"Cohort: {len(cohort)} cells, {n_classes} classes, chance={chance:.4f}")
    _log(fh, f"  {class_counts(cohort)}")

    fc_key = feature_cache.cohort_key(cohort, args.source, args.real_radius)
    fc_meta = {"source": args.source, "n_cells": len(cohort), "real_radius": args.real_radius,
               "label_column": args.label_column, "min_class_size": MIN_CLASS_SIZE,
               "n_cells_requested": args.n_cells, "max_per_class": args.max_per_class,
               "smoke": bool(args.smoke), "seed": SEED,
               "constants_digest": feature_cache.constants_digest(),
               "density_config": features.DENSITY_CONFIG,
               "r_edges": raf.R_EDGES.tolist(), "z_edges": raf.Z_EDGES.tolist(),
               "cache_path": args.cache}
    _log(fh, "Feature cache: disabled (--no_feature_cache)" if args.no_feature_cache
             else f"Feature cache: {feature_cache.cache_dir(fc_key)}")

    _write_manifest(cohort, os.path.join(out_dir, "cohort_manifest.csv"))
    _write_run_config(out_dir, {
        "source": args.source, "cache_path": args.cache, "labels_path": args.labels,
        "coordinates": ("swc_upright (soma-centered on all three axes)" if args.source == "em_exc"
                        else "global rotated into upright by inh_id_map, then soma-centered"),
        "smoke": bool(args.smoke), "outer_k": outer_k, "inner_k": inner_k,
        "representations": rep_list, "modalities": mod_list,
        "n_cells_requested": args.n_cells, "n_cells": len(cohort),
        "max_per_class": args.max_per_class,
        "classes": cohort.label_names, "class_counts": class_counts(cohort),
        "label_column": args.label_column, "exc_only": args.exc_only,
        "inh_only": args.inh_only, "real_radius": args.real_radius,
        "require_types": args.require_types,
        "permute_labels": args.permute_labels,
        "solver": args.solver, "dtype": args.dtype,
        "chance_balanced_accuracy": chance,
        "feature_cache": (None if args.no_feature_cache else feature_cache.cache_dir(fc_key)),
    })
    _write_desc(out_dir, cohort, rep_list, mod_list, outer_k, inner_k, args, n_classes)
    _mirror(out_dir, RESULTS_DIR)

    csv_path = os.path.join(out_dir, "benchmark_metrics.csv")
    with open(csv_path, "w", newline="") as csv_fh:
        writer = csv.writer(csv_fh)
        writer.writerow(["feature_rep", "modality", "metric", "mean", "std", "n_cells",
                         "n_features", "n_classes", "per_fold", "chance_corrected",
                         "best_params", "grid_boundary"])

        for rep in rep_list:
            kind, extractor = EXTRACTORS[rep]
            for mod in mod_list:
                if kind == "control" and mod != mod_list[0]:
                    continue                       # soma_depth does not depend on modality
                _log(fh, f"--- {rep} x {mod} ---")
                t0 = time.time()

                if kind == "generative":
                    with patchseq_modality_types():
                        cv, skipped = extractor(cohort, mod, outer_k=outer_k,
                                                inner_k=inner_k, seed=SEED)
                    _log(fh, f"  gaussian cv: {len(skipped)} skipped in {time.time() - t0:.1f}s")
                    if cv is None:
                        _log(fh, "  too few cells or degenerate labels, skipping.")
                        continue
                    write_rows(writer, rep, mod, cv, chance=chance, csv_fh=csv_fh)
                    # No per-fold matrix and no OOF here: run_gaussian_cv is the untouched
                    # Patch-seq function and returns a plain CVResult.
                    persistence.write_confusion(
                        os.path.join(out_dir, f"confusion_{rep}_{mod}.csv"),
                        cv.confusion, cv.class_names)
                    plot_confusion(cv.confusion, cv.class_names,
                                   os.path.join(out_dir, f"confusion_{rep}_{mod}.png"),
                                   title=f"{rep} / {mod} (n={cv.n_samples})")
                    _mirror(out_dir, RESULTS_DIR)
                    continue

                fr = (None if args.no_feature_cache
                      else feature_cache.load(fc_key, rep, mod, cohort=cohort))
                if fr is not None:
                    _log(fh, f"  extract: cache hit, {len(fr.specimen_ids)} cells, "
                             f"{fr.X.shape[1]} features, {len(fr.skipped)} skipped "
                             f"({time.time() - t0:.1f}s)")
                else:
                    fr = extractor(cohort) if kind == "control" else extractor(cohort, mod)
                    _log(fh, f"  extract: {len(fr.specimen_ids)} cells, {fr.X.shape[1]} features, "
                             f"{len(fr.skipped)} skipped in {time.time() - t0:.1f}s")
                    if not args.no_feature_cache:
                        p = feature_cache.save(fc_key, rep, mod, fr, fc_meta)
                        _log(fh, f"  cached: {p} ({os.path.getsize(p) / 1e6:.0f} MB)")
                for sid, reason in fr.skipped[:10]:
                    _log(fh, f"    skip {sid}: {reason}")
                if len(fr.skipped) > 10:
                    _log(fh, f"    ...and {len(fr.skipped) - 10} more.")

                if len(fr.specimen_ids) < 30:
                    _log(fh, "  too few cells, skipping classifier.")
                    continue

                X, y = _align_features_and_labels(fr, cohort)
                classes_in_y = np.unique(y)
                if len(classes_in_y) < 2:
                    _log(fh, "  degenerate label set, skipping.")
                    continue
                class_names = [cohort.label_names[i] for i in classes_in_y]
                y_remap = np.searchsorted(classes_in_y, y)

                t0 = time.time()
                cv = run_cv(X, y_remap, class_names=class_names, outer_k=outer_k,
                            inner_k=inner_k, seed=SEED, solver=args.solver, dtype=args.dtype,
                            log=lambda m: _log(fh, m))
                _log(fh, f"  cv: outer={outer_k} inner={inner_k} in {time.time() - t0:.1f}s")
                _log(fh, f"  balanced_accuracy = {np.mean(cv.metrics['balanced_accuracy']):.4f} "
                         f"(chance {chance:.4f})")
                _log(fh, f"  best_params = {cv.best_params}")

                write_rows(writer, rep, mod, cv, chance=chance, csv_fh=csv_fh)

                stem = f"{rep}_{mod}"
                persistence.write_oof(os.path.join(out_dir, f"oof_{stem}.npz"),
                                      cv.oof_proba, y_remap, cv.oof_fold,
                                      fr.specimen_ids, class_names)
                persistence.write_confusion(os.path.join(out_dir, f"confusion_{stem}.csv"),
                                            cv.confusion, class_names,
                                            per_fold=cv.confusion_per_fold)
                plot_confusion(cv.confusion, class_names,
                               os.path.join(out_dir, f"confusion_{stem}.png"),
                               title=f"{rep} / {mod} (n={cv.n_samples})")
                if not args.smoke:
                    try:
                        _plot_embedding_capped(X, y_remap, class_names, out_dir, stem, fh)
                    except Exception as e:
                        _log(fh, f"  embedding failed: {type(e).__name__}: {e}")
                _mirror(out_dir, RESULTS_DIR)

    _log(fh, f"Done. Durable copy: {out_dir}")
    fh.close()
    _mirror(out_dir, RESULTS_DIR)


def _plot_embedding_capped(X, y, class_names, out_dir, stem, fh) -> None:
    """UMAP on 48k x 512 costs 5-15 min per rep, which would dominate the run.

    Subsample to EMBED_MAX_CELLS and pre-reduce with PCA(50). The cap is stated in the title
    so the plot is not read as the whole cohort.
    """
    from sklearn.decomposition import PCA

    n_total = len(X)
    title = f"{stem} (n={n_total})"
    if n_total > EMBED_MAX_CELLS:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(n_total, EMBED_MAX_CELLS, replace=False)
        X, y = X[idx], y[idx]
        title = f"{stem} (UMAP on {EMBED_MAX_CELLS} of {n_total} cells, PCA50 first)"
        _log(fh, f"  embedding: capped at {EMBED_MAX_CELLS} of {n_total} cells")
    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=SEED).fit_transform(X)
    plot_embedding(X, y, class_names, os.path.join(out_dir, f"embedding_{stem}.png"),
                   method="umap", title=title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="em_exc", choices=sorted(SOURCE_DEFAULTS),
                       help="Which cohort. Never combined; one run each.")
    parser.add_argument("--smoke", action="store_true", help="Small fast run to check plumbing.")
    parser.add_argument("--representations", nargs="+", default=None)
    parser.add_argument("--modalities", nargs="+", default=None)
    parser.add_argument("--n_cells", type=int, default=None,
                        help="Class-proportional subsample size (learning-curve point).")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Cap cells per class (balances the cohort; not a curve point).")
    parser.add_argument("--solver", default="gpu", choices=["gpu", "saga"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--label_column", default="coarse_label")
    parser.add_argument("--exc_only", type=lambda s: s.lower() != "false", default=True,
                        help="em_exc only: keep the 8 excitatory coarse classes.")
    parser.add_argument("--inh_only", type=lambda s: s.lower() != "false", default=True,
                        help="microns_inh only: keep the 11 inhibitory coarse classes.")
    parser.add_argument("--require_types", nargs="+", type=int, default=None,
                        help="microns_inh only: keep only cells carrying all these SWC types. "
                             "`--require_types 2` makes dendrite/axon/full a matched comparison "
                             "on one cohort instead of three cohorts of different sizes.")
    parser.add_argument("--real_radius", action="store_true",
                        help="microns_inh only: use the cached radii instead of 1.0. Off by "
                             "default -- radius-derived morphometrics are where a depth-like "
                             "leak hides, and 1.0 keeps the two cohorts comparable.")
    parser.add_argument("--permute_labels", action="store_true",
                        help="Negative control: balanced accuracy must collapse to 1/K.")
    parser.add_argument("--no_feature_cache", action="store_true",
                        help="Re-extract instead of reading scratch/features/. Use to confirm a "
                             "cached matrix still reproduces from the source cache.")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--labels", default=None)
    args = parser.parse_args()
    args.cache = args.cache or SOURCE_DEFAULTS[args.source]["cache"]
    args.labels = args.labels or SOURCE_DEFAULTS[args.source]["labels"]

    RESULTS_DIR = "/root/capsule/results"
    ARCHIVE_DIR = "/root/capsule/scratch/microns_benchmarks"

    MIN_CLASS_SIZE = 25
    SEED = 0

    GENERIC_NAMES = {"desc.txt", "run_log.txt", "run_config.json"}

    OUTER_K = 5
    INNER_K = 3

    # Cheapest-first: _mirror runs after every rep, so a crash late in the run still leaves
    # the earlier rows in results/.
    REPRESENTATIONS = [
        "soma_depth", "radial_arbors", "density_xz", "density_yz", "density_xy",
        "persistence_image", "morphometric_statistics", "sholl", "gaussian_arbor",
    ]
    # Default modalities are per-source and live in SOURCE_DEFAULTS at the top, because argparse
    # needs the source names for --source choices.

    EMBED_MAX_CELLS = 10000

    # Documentation only -- the driver never loops over these. One Code Ocean run per value.
    LEARNING_CURVE_POINTS = (1500, 8000, 24000, 0)

    SMOKE_PER_CLASS = 20
    SMOKE_REPS = ["soma_depth", "radial_arbors", "morphometric_statistics"]
    SMOKE_MODALITIES = ["dendrite"]
    SMOKE_OUTER_K = 3
    SMOKE_INNER_K = 2

    main(args)
