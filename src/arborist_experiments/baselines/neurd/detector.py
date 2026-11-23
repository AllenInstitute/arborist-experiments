"""
Created on Mon November 24 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

NEURDMergeDetector: parameter-free merge detector that scores a candidate
site with NEURD's heuristic rules (see ``heuristics``). It has the same
``forward`` / ``config`` / ``save`` / ``load`` interface as the trained
detectors so it runs through the same evaluation harness.
"""

import torch
import torch.nn as nn

from arborist.utils.util import write_json

from arborist_experiments.baselines.neurd.heuristics import rule_margins


class NEURDMergeDetector(nn.Module):
    """
    Scores each site by how far the most-violated NEURD rule is past its
    threshold. The logit is positive iff some rule fires, so a sigmoid
    threshold of 0.5 reproduces NEURD's binary decision while the magnitude
    ranks sites for precision-recall curves. Nothing is trained.

    Parameters
    ----------
    angle_scale : float, optional
        Degrees of angle margin that count as one unit of logit, putting the
        angle rules on the same footing as the degree rule. Default is 10.
    **rule_kwargs
        Thresholds forwarded to ``rule_margins`` (high_degree_min,
        pass_through_angle, fork_angle, bend_angle, comparison_distance,
        offset).
    """

    modality = "graph"

    def __init__(self, angle_scale=10.0, **rule_kwargs):
        super().__init__()
        self.angle_scale = angle_scale
        self.rule_kwargs = rule_kwargs
        self.config = {
            "model_type": f"{type(self).__name__}_{self.modality}",
            "detector_type": type(self).__name__,
            "angle_scale": angle_scale,
            **rule_kwargs,
        }
        self.register_buffer("_device_anchor", torch.zeros(1))

    def score(self, sample):
        """
        Merge score of one site: the largest rule margin, angles scaled.

        Parameters
        ----------
        sample : TreeSample
            Rooted subgraph from ``arborist.data.datasets``, built by
            ``build_tree_sample``.

        Returns
        -------
        float
        """
        margins = rule_margins(sample, **self.rule_kwargs)
        return max(
            margins["high_degree"],
            margins["pass_through"] / self.angle_scale,
            margins["fork"] / self.angle_scale,
            margins["bend"] / self.angle_scale,
        )

    def forward(self, x, task=None):
        """
        Parameters
        ----------
        x : List[TreeSample] or dict
            Batch of rooted subgraphs; a dict holds them under "tree_sample".
        task : str, optional
            Ignored; present for the trainers' ``forward(x, task)`` convention.

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 1).
        """
        samples = x["tree_sample"] if isinstance(x, dict) else x
        scores = [self.score(s) for s in samples]
        return torch.tensor(scores, dtype=torch.float32, device=self._device_anchor.device).unsqueeze(1)

    def save(self, path):
        """
        Saves the config (the rule thresholds) to a .pth checkpoint.

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

    @classmethod
    def load(cls, path, map_location=None):
        """
        Rebuilds the detector from a checkpoint's config.

        Parameters
        ----------
        path : str
            Path to .pth checkpoint file.
        map_location : str or torch.device, optional
            Unused; accepted for interface parity. Default is None.

        Returns
        -------
        NEURDMergeDetector
        """
        config = dict(torch.load(path, map_location="cpu", weights_only=False)["config"])
        config.pop("model_type")
        config.pop("detector_type")
        return cls(**config)
