"""
Created on Sun November 23 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

TMDDescriptor: represents a rooted subgraph by its flattened TMD
persistence image. It has no parameters and the same interface as the
learned graph encoders (a TreeSample batch in, one vector per TreeSample
out), so a task head can take it in place of Arborist and the head is the
only trained part, as in the paper.
"""

import numpy as np
import torch
import torch.nn as nn

from arborist_experiments.baselines.tmd.persistence import (
    persistence_barcode,
    persistence_image,
)


class TMDDescriptor(nn.Module):
    """
    Parameter-free graph descriptor: the flattened TMD persistence image of
    a rooted subgraph.

    Parameters
    ----------
    max_dist : float
        Upper bound of the persistence image axes; use the subgraph
        extraction radius so every barcode fits on the grid.
    resolution : int, optional
        Persistence image resolution per axis; the output has
        ``resolution ** 2`` entries. Default is 20.
    bandwidth : float, optional
        Gaussian bandwidth in microns. Defaults to one grid cell.
    distance : str, optional
        "radial" or "path", see ``tree_distances``. Default is "radial".
    """

    def __init__(self, max_dist, resolution=20, bandwidth=None, distance="radial"):
        super().__init__()
        self.config = {
            "max_dist": max_dist,
            "resolution": resolution,
            "bandwidth": bandwidth,
            "distance": distance,
        }
        self.embed_dim = resolution * resolution
        self.register_buffer("_device_anchor", torch.zeros(1))

    def featurize(self, tree_sample):
        """
        Flattened persistence image of one TreeSample.

        Parameters
        ----------
        tree_sample : TreeSample
            Rooted subgraph from ``arborist.data.datasets``, built by
            ``build_tree_sample``.

        Returns
        -------
        numpy.ndarray
            Shape (resolution ** 2,) float32.
        """
        barcode = persistence_barcode(tree_sample, self.config["distance"])
        img = persistence_image(
            barcode,
            self.config["max_dist"],
            resolution=self.config["resolution"],
            bandwidth=self.config["bandwidth"],
        )
        return img.ravel()

    @torch._dynamo.disable
    def forward(self, tree_samples):
        """
        Parameters
        ----------
        tree_samples : List[TreeSample]
            Rooted subgraphs from ``arborist.data.datasets``, built by
            ``build_tree_sample``.

        Returns
        -------
        torch.Tensor
            Shape (B, resolution ** 2).
        """
        feats = np.stack([self.featurize(s) for s in tree_samples])
        return torch.as_tensor(feats, device=self._device_anchor.device)
