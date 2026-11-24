"""
Created on Sun November 23 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Merge detectors: models that score candidate sites of a skeleton graph.
LearnedMergeDetector fuses any combination of graph and image encoders
with a FeedForwardNet head; HeuristicMergeDetector scores sites with a set
of rules. Both share the ``forward(x)`` and ``config`` / ``save``
conventions of the capsule trainers and benchmark; ``registry.load_detector``
rebuilds either from a checkpoint.
"""

import torch
import torch.nn as nn

from arborist.utils.util import write_json
from neuron_proofreader.utils.ml_util import FeedForwardNet

from arborist_experiments.tasks.merge_proofreading.factory import GraphEncoderFactory


def _qualname(obj):
    """Import path of a function, class, or module instance's class."""
    if isinstance(obj, nn.Module):
        obj = type(obj)
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

    def forward(self, x):
        """
        Parameters
        ----------
        x : dict or List[TreeSample]
            Batch with "img" (B, C, D, H, W) and/or "tree_sample"
            (List[TreeSample]), or the list alone for graph-only models.

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
    an image encoder and classifies them with a FeedForwardNet. Swapping the
    graph encoder is the only difference between Arborist and the baselines.

    Parameters
    ----------
    graph_encoder : torch.nn.Module, optional
        Maps List[TreeSample] to (B, embed_dim); built by
        ``GraphEncoderFactory`` (e.g. "arborist", "tmd"). Default is None.
    vision_encoder : VisionEncoder, optional
        Maps images (B, C, D, H, W) to (B, embed_dim). The same CNN3D in
        every experiment. Default is None.
    dropout : float, optional
        Dropout on the concatenated embedding. Default is 0.1.
    """

    def __init__(self, graph_encoder=None, vision_encoder=None, dropout=0.1):
        # Call parent class constructor
        super().__init__()

        # Check that encoders are provided
        if graph_encoder is None and vision_encoder is None:
            raise ValueError("graph_encoder and vision_encoder is required")
        encoders = [enc for enc in (vision_encoder, graph_encoder) if enc is not None]

        # Instance attributes
        self.graph_encoder = graph_encoder
        self.vision_encoder = vision_encoder
        self.modality = "_".join(
            name for name, enc in (("image", vision_encoder), ("graph", graph_encoder)) if enc is not None
        )
        self.drop = nn.Dropout(dropout)
        self.head = FeedForwardNet(sum(enc.embed_dim for enc in encoders), 1, 3)

        # Name runs by the graph encoder, the part that differs between models
        graph_type = "cnn3d" if graph_encoder is None else GraphEncoderFactory.name_of(graph_encoder)
        self.config = {
            "detector_type": _qualname(self),
            "model_type": f"{graph_type}_{self.modality}",
            "graph_encoder": None if graph_encoder is None else {
                "type": graph_type, "kwargs": graph_encoder.config
            },
            "vision_encoder": None if vision_encoder is None else vision_encoder.config,
            "dropout": dropout,
        }

    def forward(self, x):
        z = []
        if self.vision_encoder is not None:
            z.append(self.vision_encoder(x["img"]))
        if self.graph_encoder is not None:
            z_graph = self.graph_encoder(self.tree_samples(x))
            z.append(z_graph.to(z[0].device) if z else z_graph)
        return self.head(self.drop(torch.cat(z, dim=1)))

    def half_vision(self):
        """Runs the image encoder in fp16, as MergeModel.half_vision does."""
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

    def forward(self, x):
        scores = [self.score(s) for s in self.tree_samples(x)]
        return torch.tensor(scores, dtype=torch.float32, device=self._device_anchor.device).unsqueeze(1)
