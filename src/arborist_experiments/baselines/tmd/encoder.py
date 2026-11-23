"""
Created on Mon November 23 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

TMDEncoder: embeds a rooted subgraph through its TMD persistence image with
a small MLP. It has the same interface as the Arborist encoder as seen by
the task models in ``tasks`` (a TreeSample batch in, one embedding per
sample out), so any task head can take it in place of Arborist.
"""

import numpy as np
import torch
import torch.nn as nn

from arborist_experiments.baselines.tmd.persistence import (
    persistence_barcode,
    persistence_image,
)


class TMDEncoder(nn.Module):
    """
    Graph encoder that embeds a rooted subgraph through its TMD persistence
    image with a two-layer MLP.

    Maps a batch of arborist TreeSamples to one embedding per sample, the
    same interface as the Arborist encoder, so any task head can use either.
    The persistence image is computed without gradients; only the MLP trains.

    Parameters
    ----------
    max_dist : float
        Upper bound of the persistence image axes; use the subgraph
        extraction radius so every barcode fits on the grid.
    resolution : int, optional
        Persistence image resolution per axis. Default is 20.
    bandwidth : float, optional
        Gaussian bandwidth in microns. Defaults to one grid cell.
    distance : str, optional
        "radial" or "path", see ``tree_distances``. Default is "radial".
    hidden_dim : int, optional
        MLP hidden dimension. Default is 128.
    embed_dim : int, optional
        Output embedding dimension. Default is 128, the size of the Arborist
        readout used by the task models (CLS plus pooled root curves at
        latent_dim 64), so the two encoders are interchangeable.
    dropout : float, optional
        Dropout probability inside the MLP. Default is 0.1.
    """

    def __init__(
        self,
        max_dist,
        resolution=20,
        bandwidth=None,
        distance="radial",
        hidden_dim=128,
        embed_dim=128,
        dropout=0.1,
    ):
        super().__init__()
        self.config = {
            "max_dist": max_dist,
            "resolution": resolution,
            "bandwidth": bandwidth,
            "distance": distance,
            "hidden_dim": hidden_dim,
            "embed_dim": embed_dim,
            "dropout": dropout,
        }
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(resolution * resolution, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def featurize(self, sample):
        """
        Flattened persistence image of one TreeSample.

        Parameters
        ----------
        sample : TreeSample
            Rooted subgraph from ``arborist.data.datasets``, built by
            ``build_tree_sample``.

        Returns
        -------
        numpy.ndarray
            Shape (resolution ** 2,) float32.
        """
        barcode = persistence_barcode(sample, self.config["distance"])
        img = persistence_image(
            barcode,
            self.config["max_dist"],
            resolution=self.config["resolution"],
            bandwidth=self.config["bandwidth"],
        )
        return img.ravel()

    @torch._dynamo.disable
    def forward(self, samples):
        """
        Parameters
        ----------
        samples : List[TreeSample]
            Rooted subgraphs from ``arborist.data.datasets``, built by
            ``build_tree_sample``.

        Returns
        -------
        torch.Tensor
            Shape (B, embed_dim).
        """
        feats = np.stack([self.featurize(s) for s in samples])
        device = next(self.parameters()).device
        return self.mlp(torch.as_tensor(feats, device=device))
