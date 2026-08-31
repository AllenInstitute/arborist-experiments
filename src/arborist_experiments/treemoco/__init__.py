"""TreeMoCo: contrastive pretraining over whole reconstructions (Chen et al. 2022).

Task-agnostic by construction. These five modules take a `CellGraph`, collapse it to an
irreducible tree, augment it and return an embedding; they never see a label, a cohort or a
metric. `utils.cells.CellGraph` is the only import out of this package.

`TreeClassifier` in `model` is not an exception: it is the linear head MoCo's own fine-tuning
protocol prescribes, parameterised by `n_classes` and indifferent to what the classes mean.

The drivers that build a labelled cohort and score a classifier against it are part of a task
and live with it, in `global_classification`.
"""
