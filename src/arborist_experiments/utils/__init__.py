"""Infrastructure with no task and no cohort baked in.

`cells` -- the SkeletonGraph-backed cell type plus the process-pool helper that every
extractor runs through -- is currently the whole of it.

Nothing here imports from `global_classification`, `treemoco` or `merge_detection`, which
keeps every package that depends on it importable without dragging a benchmark along.
"""
