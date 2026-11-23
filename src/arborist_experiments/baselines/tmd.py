"""
Created on Sun November 23 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

TMD baseline for merge detection.

Topological Morphology Descriptor (Kanari et al., 2018, Neuroinformatics).
A rooted tree is summarised as a persistence barcode: every branch is born
at the distance of its farthest leaf and dies at the branch point where it
meets a sibling that reaches farther, with the last survivor dying at the
root. The barcode is rendered as a persistence image on a fixed grid, which
a small MLP embeds. TMDEncoder stands in for the Arborist encoder inside
the detectors in ``detectors.py``, so the graph-only and multimodal
comparisons share everything but the graph branch.

Operates directly on arborist TreeSamples (see ``build_tree_sample``).
"""

import numpy as np
import torch
import torch.nn as nn



def tree_distances(sample, distance="radial"):
    """
    Distance from the root to the distal endpoint of each token.

    Parameters
    ----------
    sample : TreeSample
    distance : str, optional
        "radial" for Euclidean distance in the root frame, "path" for cable
        length along the tree. Default is "radial".

    Returns
    -------
    numpy.ndarray
        Shape (T,) float32; token 0 is the root with distance 0.
    """
    if distance == "radial":
        return np.linalg.norm(sample.endpoints[:, 1], axis=1).astype(np.float32)
    if distance == "path":
        lengths = [0.0] + [np.linalg.norm(c, axis=1).sum() for c in sample.curves]
        return (sample.cable + np.asarray(lengths)).astype(np.float32)
    raise ValueError(f"Unknown distance: {distance!r}. Expected 'radial' or 'path'.")


def persistence_barcode(sample, distance="radial"):
    """
    TMD persistence barcode of a rooted tree.

    Tokens are visited deepest-first. At each branch point the child branch
    with the larger birth value survives; every other child produces a bar
    (birth, death) with death at the branch point's distance. The survivor
    reaching the root produces the final bar dying at the root (distance 0).

    Parameters
    ----------
    sample : TreeSample
    distance : str, optional
        Distance function, see ``tree_distances``. Default is "radial".

    Returns
    -------
    numpy.ndarray
        Shape (n_leaves, 2) float32 array of (birth, death) pairs. Empty for a
        sample with no curves.
    """
    n_tokens = len(sample.parents)
    if n_tokens <= 1:
        return np.zeros((0, 2), dtype=np.float32)

    f = tree_distances(sample, distance)
    children = [[] for _ in range(n_tokens)]
    for k in range(1, n_tokens):
        children[sample.parents[k]].append(k)

    # hops counts curves to each token's proximal node, so a curve's distal
    # node sits one level deeper; the root stays at depth 0
    depth = sample.hops.copy()
    depth[1:] += 1

    bars = []
    alive = f.copy()
    for k in np.argsort(-depth, kind="stable"):
        if not children[k]:
            continue
        births = alive[children[k]]
        survivor = int(np.argmax(births))
        for i, birth in enumerate(births):
            if i != survivor:
                bars.append((birth, f[k]))
        alive[k] = births[survivor]
    bars.append((alive[0], f[0]))
    return np.asarray(bars, dtype=np.float32)


def persistence_image(barcode, max_dist, resolution=20, bandwidth=None):
    """
    Renders a barcode as a persistence image on a fixed grid.

    Each (birth, death) pair contributes an isotropic Gaussian; the grid
    spans [0, max_dist] on both axes so images from different subgraphs are
    comparable. The image is normalised to a peak of 1.

    Parameters
    ----------
    barcode : numpy.ndarray
        Shape (n_bars, 2) array of (birth, death) pairs.
    max_dist : float
        Upper bound of both axes (e.g. the subgraph extraction radius).
    resolution : int, optional
        Number of grid cells per axis. Default is 20.
    bandwidth : float, optional
        Gaussian standard deviation in microns. Defaults to one grid cell.

    Returns
    -------
    numpy.ndarray
        Shape (resolution, resolution) float32; rows index death, columns
        index birth.
    """
    img = np.zeros((resolution, resolution), dtype=np.float32)
    if len(barcode) == 0:
        return img
    bw = bandwidth or max_dist / resolution
    grid = np.linspace(0, max_dist, resolution)
    g_birth = np.exp(-0.5 * ((grid[None, :] - barcode[:, 0:1]) / bw) ** 2)
    g_death = np.exp(-0.5 * ((grid[None, :] - barcode[:, 1:2]) / bw) ** 2)
    img = (g_death[:, :, None] * g_birth[:, None, :]).sum(axis=0)
    peak = img.max()
    if peak > 0:
        img /= peak
    return img.astype(np.float32)


class TMDEncoder(nn.Module):
    """
    Graph encoder that embeds a rooted subgraph through its TMD persistence
    image with a two-layer MLP. Plug into GraphMergeDetector or
    MultimodalMergeDetector in place of the Arborist encoder.

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
        Output embedding dimension. Default is 128, matching the Arborist
        merge readout (CLS plus pooled root curves at latent_dim 64).
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

        Returns
        -------
        torch.Tensor
            Shape (B, embed_dim).
        """
        feats = np.stack([self.featurize(s) for s in samples])
        device = next(self.parameters()).device
        return self.mlp(torch.as_tensor(feats, device=device))
