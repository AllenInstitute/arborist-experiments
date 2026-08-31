"""Predicting a cell's transcriptomic type from one whole reconstruction.

Two cohorts -- `patchseq` (Patch-seq, 1618 cells, t-type labels) and `microns` (MICrONS EM,
excitatory and inhibitory) -- and several competing representations of the same cell within
each: handcrafted density/Sholl/persistence/morphometric blocks, radial arbor profiles, path
encodings, and TreeMoCo embeddings. The representation is the variable; the label space, the
cross-validation and the metrics CSV are held fixed so the rows are comparable.

`cv` is that fixed nested-CV harness and `rows` writes the CSV. Both live here rather than in
`utils` because nothing outside this task uses them -- `merge_detection` scores something else
entirely, and `treemoco` is a representation, not a task.

The TreeMoCo drivers are here, not under `treemoco`, for the same reason: they build a
*labelled* cohort, subsample it per class, and score a t-type classifier. That is this task.
The model they call is not.
"""
