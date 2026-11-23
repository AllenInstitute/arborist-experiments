"""
Created on Sun November 23 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Merge detectors: models that score candidate sites of a skeleton graph.
LearnedMergeDetector fuses any combination of graph and image encoders
with a FeedForwardNet head; HeuristicMergeDetector scores sites with a set
of rules. Both share the ``forward(x, task)`` and ``config`` / ``save``
conventions of the capsule trainers and benchmark; ``registry.load_detector``
rebuilds either from a checkpoint.
"""

import torch
import torch.nn as nn

from arborist.utils.util import write_json
from neuron_proofreader.utils.ml_util import FeedForwardNet


def _qualname(obj):
    """Import path of a class or function, used to rebuild it from a config."""
    obj = obj if isinstance(obj, type) or callable(obj) and not isinstance(obj, nn.Module) else type(obj)
    return f"{obj.__module__}.{obj.__qualname__}"


class MergeDetector(nn.Module):
    """
    Base class: a module that maps a batch of sites to merge logits of shape
    (B, 1). Subclasses set ``modality`` and ``config``; ``config`` must hold
    "detector_type" and whatever ``registry.build_detector`` needs to
    rebuild the detector.
    """

    modality = None

    @staticmethod
    def tree_samples(x):
        """Extracts the List[TreeSample] from a batch dict or passes a list through."""
        return x["tree_sample"] if isinstance(x, dict) else x

    def forward(self, x, task=None):
        """
        Parameters
        ----------
        x : dict or List[TreeSample]
            Batch with "img" (B, C, D, H, W) and/or "tree_sample"
            (List[TreeSample]), or the list alone for graph-only models.
        task : str, optional
            Ignored; present for the trainers' ``forward(x, task)`` convention.

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 1).
        """
        raise NotImplementedError

    def save(self, path):
        """
        Saves the config and weights to a .pth checkpoint.

        Parameters
        ----------
        path : str
            Destination path.
        """
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    def save_config(self, path):
        """
        Saves the config to a JSON file.

        Parameters
        ----------
        path : str
            Destination path (e.g. .../model_config.json).
        """
        write_json(path, self.config)


class LearnedMergeDetector(MergeDetector):
    """
    Trained detector: concatenates the embeddings of a graph encoder and/or
    an image encoder and classifies them with a three-layer FeedForwardNet,
    the merge head of JointProofreader. Swapping the graph encoder is the
    only difference between Arborist and the baselines.

    Parameters
    ----------
    graph_encoder : torch.nn.Module, optional
        Maps List[TreeSample] to (B, embed_dim); e.g. ArboristEncoder or
        TMDEncoder. Default is None.
    vision_encoder : torch.nn.Module, optional
        Maps images (B, C, D, H, W) to (B, embed_dim); e.g.
        MergeVisionEncoder. Default is None.
    dropout : float, optional
        Dropout on the concatenated embedding. Default is 0.1.
    """

    def __init__(self, graph_encoder=None, vision_encoder=None, dropout=0.1):
        super().__init__()
        if graph_encoder is None and vision_encoder is None:
            raise ValueError("At least one of graph_encoder and vision_encoder is required")
        self.graph_encoder = graph_encoder
        self.vision_encoder = vision_encoder
        self.modality = "_".join(
            name for name, enc in (("image", vision_encoder), ("graph", graph_encoder)) if enc is not None
        )

        encoders = [enc for enc in (vision_encoder, graph_encoder) if enc is not None]
        self.drop = nn.Dropout(dropout)
        self.head = FeedForwardNet(sum(enc.embed_dim for enc in encoders), 1, 3)

        def describe(enc):
            return None if enc is None else {"type": _qualname(enc), "kwargs": enc.config}

        # Name runs by the graph encoder, the part that differs between models
        named = graph_encoder if graph_encoder is not None else vision_encoder
        self.config = {
            "detector_type": _qualname(self),
            "model_type": f"{type(named).__name__}_{self.modality}",
            "graph_encoder": describe(graph_encoder),
            "vision_encoder": describe(vision_encoder),
            "dropout": dropout,
        }

    def forward(self, x, task=None):
        z = []
        if self.vision_encoder is not None:
            z.append(self.vision_encoder(x["img"]))
        if self.graph_encoder is not None:
            z_graph = self.graph_encoder(self.tree_samples(x))
            z.append(z_graph.to(z[0].device) if z else z_graph)
        return self.head(self.drop(torch.cat(z, dim=1)))

    def half_vision(self):
        """Runs the image encoder in fp16, as JointProofreader.half_vision does."""
        if self.vision_encoder is not None:
            self.vision_encoder.half()
        return self


class HeuristicMergeDetector(MergeDetector):
    """
    Rule-based detector with no trainable parameters. ``rules`` maps a
    TreeSample to a dict of signed margins, positive when a rule fires; the
    logit is the largest weighted margin, so sigmoid > 0.5 is the rules'
    binary decision and the magnitude ranks sites for precision-recall.

    Parameters
    ----------
    rules : callable
        ``rules(tree_sample, **rule_kwargs) -> dict[str, float]``, e.g.
        ``baselines.neurd.rule_margins``. Must be importable by name so the
        detector can be rebuilt from its config.
    weights : dict[str, float], optional
        Per-rule multipliers that put margins in different units on a
        common scale. Rules not listed have weight 1. Default is None.
    **rule_kwargs
        Thresholds forwarded to ``rules``.
    """

    modality = "graph"

    def __init__(self, rules, weights=None, **rule_kwargs):
        super().__init__()
        self.rules = rules
        self.weights = dict(weights or {})
        self.rule_kwargs = rule_kwargs
        self.config = {
            "detector_type": _qualname(self),
            "model_type": f"{rules.__module__.split('.')[-2]}_{self.modality}",
            "rules": _qualname(rules),
            "weights": self.weights,
            "rule_kwargs": rule_kwargs,
        }
        self.register_buffer("_device_anchor", torch.zeros(1))

    def score(self, tree_sample):
        """
        Merge score of one site: the largest weighted rule margin.

        Parameters
        ----------
        tree_sample : TreeSample
            Rooted subgraph from ``arborist.data.datasets``, built by
            ``build_tree_sample``.

        Returns
        -------
        float
        """
        margins = self.rules(tree_sample, **self.rule_kwargs)
        return max(self.weights.get(name, 1.0) * margin for name, margin in margins.items())

    def forward(self, x, task=None):
        scores = [self.score(s) for s in self.tree_samples(x)]
        return torch.tensor(scores, dtype=torch.float32, device=self._device_anchor.device).unsqueeze(1)
