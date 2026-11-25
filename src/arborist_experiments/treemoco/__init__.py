"""TreeMoCo: contrastive pretraining over whole reconstructions (Chen et al. 2022).

Task-agnostic by construction. These modules take a cell (an arborist `SkeletonGraph`), collapse
it to an irreducible tree, augment it and return an embedding; they never see a label, a cohort or
a metric. The only imports out of this package are the cell type itself (`arborist.skeleton_graph`)
and the cell accessors in `utils.cells`.

`TreeClassifier` in `model` is not an exception: it is the linear head MoCo's own fine-tuning
protocol prescribes, parameterised by `n_classes` and indifferent to what the classes mean.

Code that builds a labelled cohort and scores a classifier against it belongs to a task, not here.
"""
