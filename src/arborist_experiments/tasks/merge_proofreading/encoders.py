"""
Created on Mon November 24 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Encoders shared by the merge detectors: the Arborist graph encoder with the
merge readout, and the image encoder of neuron_proofreader's MergeModel.
Every encoder maps a batch to one embedding per site and exposes
``embed_dim`` and a ``config`` of its constructor arguments, the same
contract the baseline encoders in ``baselines`` follow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from arborist.models.arborist import Arborist
from neuron_proofreader.models.vision_models import CNN3D, CenterWeightedPool3D


class ArboristEncoder(nn.Module):
    """
    Arborist with the merge readout: the CLS embedding concatenated with the
    mean embedding of the curves adjacent to the root, so the local geometry
    at the candidate site is read out directly.

    Parameters
    ----------
    latent_dim : int, optional
        Arborist latent dimension. Default is 64.
    **arborist_kwargs
        Forwarded to ``arborist.models.arborist.Arborist``.
    """

    def __init__(self, latent_dim=64, **arborist_kwargs):
        super().__init__()
        self.config = {"latent_dim": latent_dim, **arborist_kwargs}
        self.arborist = Arborist(latent_dim=latent_dim, **arborist_kwargs)
        self.embed_dim = 2 * latent_dim

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
            Shape (B, 2 * latent_dim).
        """
        with torch.amp.autocast("cuda", enabled=False):
            z = []
            for s in tree_samples:
                z_tree, z_curves = self.arborist.encode(s)
                idx = list(s.root_curve_indices or range(len(z_curves)))
                z.append(torch.cat([z_tree, z_curves[idx].mean(dim=0)]))
        return torch.stack(z)


class VisionEncoder(nn.Module):
    """
    Image encoder of neuron_proofreader's MergeModel: a CNN3D backbone with
    multi-scale center-weighted pooling, a merge-specific 1x1 conv and pool
    on the final stage, and a projection to the image embedding.

    Parameters
    ----------
    input_shape : tuple
        Image patch shape (C, D, H, W).
    embed_dim : int, optional
        Image embedding dimension. Default is 96.
    dropout : float, optional
        Dropout after the projection. Default is 0.1.
    **cnn_kwargs
        Forwarded to CNN3D (e.g. base_channels, depth, block_type,
        max_channels, pool_stage_idxs).
    """

    def __init__(self, input_shape, embed_dim=96, dropout=0.1, **cnn_kwargs):
        super().__init__()
        self.config = {
            "input_shape": tuple(input_shape),
            "embed_dim": embed_dim,
            "dropout": dropout,
            **cnn_kwargs,
        }
        self.embed_dim = embed_dim

        self.backbone = CNN3D(input_shape, use_output_head=False, **cnn_kwargs)
        final_idx = self.backbone.pool_stage_idxs[-1]
        final_channels = self.backbone.encode.blocks[final_idx].out_channels
        sigma = self.backbone.config.get("center_pool_sigma", 0.4)

        learnable_sigma = self.backbone.config.get("learnable_center_sigma", True)
        self.task_conv = nn.Conv3d(final_channels, final_channels, kernel_size=1)
        self.task_pool = CenterWeightedPool3D(sigma=sigma, learnable=learnable_sigma)
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.feature_dim + 2 * final_channels, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, img):
        """
        Parameters
        ----------
        img : torch.Tensor
            Shape (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Shape (B, embed_dim), float32 regardless of the backbone dtype.
        """
        # Run the shared backbone once
        img = img.to(next(self.backbone.parameters()).dtype)
        stages = self.backbone.encode(img)

        # Multi-scale pooling of the shared stages (center-weighted + max)
        shared_feats = []
        for idx, pool in zip(self.backbone.pool_stage_idxs, self.backbone.center_pools):
            s = stages[idx]
            shared_feats.append(torch.cat([pool(s), F.adaptive_max_pool3d(s, 1).flatten(1)], dim=1))
        shared = self.backbone.drop(torch.cat(shared_feats, dim=1))

        # Merge-specific conv and pooling on the final stage
        task_out = self.task_conv(stages[self.backbone.pool_stage_idxs[-1]])
        task = torch.cat([self.task_pool(task_out), F.adaptive_max_pool3d(task_out, 1).flatten(1)], dim=1)

        # Project to the image embedding, in fp32 for the head
        return self.proj(torch.cat([shared, task], dim=1)).float()
