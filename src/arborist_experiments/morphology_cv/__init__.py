"""Shared cross-validation machinery for the morphology benchmarks.

`classifier` is the one nested-CV elastic-net used by both the Patch-seq and MICrONS drivers,
`gpu_logreg` is its default solver, and `persistence` writes the per-cell out-of-fold record that
makes any follow-up question answerable without a refit. Nothing here knows about a cohort, a
feature representation or a data source -- that is what keeps it shareable.
"""
