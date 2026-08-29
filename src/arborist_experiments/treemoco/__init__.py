"""TreeMoCo: contrastive pretraining over whole reconstructions (Chen et al. 2022).

The core lives here rather than inside a dataset package because both benchmarks train
on it -- the MICrONS drivers were already importing these five modules across package
boundaries. The drivers themselves stay with their cohorts, since their `python -m`
paths are the documented way to launch a run.
"""
