"""Morphology-representation benchmark on MICrONS EM neurons.

Port of the sibling `patchseq` package to the MICrONS excitatory (EM_syn.hdf5) and
inhibitory (MICrONS_inh_swc) cohorts. Shared contracts (`Cell`, `Cohort`,
`cell_to_neurontree`, `plot_confusion`) are imported from the Patch-seq package directly; the
nested-CV classifier, its GPU solver and the out-of-fold writers live in the task's own `cv`,
which both cohorts share.

What is genuinely MICrONS-specific stays here: the binning constants derived from these cohorts
(`features`, `radial_arbor_features`, `inh_constants`), the modality table, the cache loaders, and
`feature_cache`, whose key includes those constants.
"""
