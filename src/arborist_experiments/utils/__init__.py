"""Infrastructure with no task and no cohort baked in.

`cells` is the SkeletonGraph-backed cell type plus the process-pool helper that every
extractor runs through; `runs` is the version-dir, logging and results-mirroring plumbing
that each driver used to carry its own copy of.

Nothing here imports from `global_classification`, `treemoco` or `merge_detection`, and that
is the property worth preserving: it is what let the cell type stop living inside one dataset
package, and what keeps `treemoco` importable without dragging a benchmark along.
"""
