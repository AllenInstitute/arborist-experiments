"""Morphology-representation benchmark on MICrONS EM neurons.

Port of the sibling `patchseq` package to the MICrONS excitatory (EM_syn.hdf5) and
inhibitory (MICrONS_inh_swc) cohorts. The feature extractors themselves are the shared
`global_classification.extract`, parameterised by a `BinningConfig`; the nested-CV classifier,
its GPU solver and the out-of-fold writers live in the task's own `cv`, which both cohorts share.

What is genuinely MICrONS-specific stays here: `binning`, holding one `BinningConfig` per cohort
with the constants derived from it; the availability gate in `modality`; the cache loaders; and
`feature_cache`, whose key includes a digest of those constants.
"""
