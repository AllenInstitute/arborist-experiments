"""
Created on Sun November 23 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Merge detectors that wrap a graph encoder from ``baselines`` so it can be
compared against Arborist under the same training and evaluation harness.
Register each new encoder in ``ENCODERS`` so checkpoints can be reloaded.

GraphMergeDetector mirrors the capsule's GraphOnly model (linear head on the
graph embedding). MultimodalMergeDetector mirrors the merge path of
neuron_proofreader's JointProofreader: the same CNN3D backbone, task conv,
center-weighted pooling, vision projection and FeedForwardNet head, with the
Arborist readout replaced by the given encoder. Both follow the trainers'
``forward(x, task)`` convention and the ``config`` / ``load`` checkpoint
protocol used by the capsule and the benchmark.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from arborist.utils.util import write_json
from neuron_proofreader.models.vision_models import CNN3D, CenterWeightedPool3D
from neuron_proofreader.utils.ml_util import FeedForwardNet

from arborist_experiments.baselines.tmd import TMDEncoder

ENCODERS = {
    "TMDEncoder": TMDEncoder,
}


def build_encoder(encoder_type, encoder_kwargs):
    """
    Instantiates a registered graph encoder by class name.

    Parameters
    ----------
    encoder_type : str
        Key in ``ENCODERS``.
    encoder_kwargs : dict
        Constructor keyword arguments (an encoder's ``config``).

    Returns
    -------
    torch.nn.Module
    """
    if encoder_type not in ENCODERS:
        raise ValueError(f"Unknown encoder: {encoder_type!r}. Expected one of {sorted(ENCODERS)}.")
    return ENCODERS[encoder_type](**encoder_kwargs)


def _tree_samples(x):
    return x["tree_sample"] if isinstance(x, dict) else x


def _load_state_dict(model, checkpoint):
    state_dict = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    return model


class GraphMergeDetector(nn.Module):
    """
    Graph-only merge detector: linear head on the encoder's embedding.

    Parameters
    ----------
    encoder : torch.nn.Module
        Graph encoder with ``embed_dim``, ``config`` and
        ``forward(List[TreeSample]) -> (B, embed_dim)``.
    """

    modality = "graph"

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, 1)
        encoder_type = type(encoder).__name__
        self.config = {
            "model_type": f"{encoder_type}_{self.modality}",
            "encoder_type": encoder_type,
            "encoder_kwargs": encoder.config,
        }

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
        return self.head(self.encoder(_tree_samples(x)))

    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    def save_config(self, path):
        write_json(path, self.config)

    @classmethod
    def load(cls, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = checkpoint["config"]
        model = cls(build_encoder(config["encoder_type"], config["encoder_kwargs"]))
        return _load_state_dict(model, checkpoint)


class MultimodalMergeDetector(nn.Module):
    """
    Image + graph merge detector with JointProofreader's merge path and a
    swappable graph encoder.

    Parameters
    ----------
    encoder : torch.nn.Module
        Graph encoder with ``embed_dim``, ``config`` and
        ``forward(List[TreeSample]) -> (B, embed_dim)``.
    vision_input_shape : tuple
        Image patch shape (C, D, H, W).
    img_embed_dim : int, optional
        Image embedding dimension. Default is 96.
    dropout : float, optional
        Dropout before the vision projection output and the head. Default is 0.1.
    **vision_backbone_kwargs
        Forwarded to CNN3D (e.g. base_channels, depth, block_type,
        max_channels, pool_stage_idxs).
    """

    modality = "image_graph"

    def __init__(
        self,
        encoder,
        vision_input_shape,
        img_embed_dim=96,
        dropout=0.1,
        **vision_backbone_kwargs,
    ):
        super().__init__()
        encoder_type = type(encoder).__name__
        self.config = {
            "model_type": f"{encoder_type}_{self.modality}",
            "encoder_type": encoder_type,
            "encoder_kwargs": encoder.config,
            "vision_input_shape": tuple(vision_input_shape),
            "img_embed_dim": img_embed_dim,
            "dropout": dropout,
            **vision_backbone_kwargs,
        }

        self.vision = CNN3D(vision_input_shape, use_output_head=False, **vision_backbone_kwargs)
        final_idx = self.vision.pool_stage_idxs[-1]
        final_channels = self.vision.encode.blocks[final_idx].out_channels
        sigma = self.vision.config.get("center_pool_sigma", 0.4)
        learnable_sigma = self.vision.config.get("learnable_center_sigma", True)
        self.merge_task_conv = nn.Conv3d(final_channels, final_channels, kernel_size=1)
        self.merge_task_pool = CenterWeightedPool3D(sigma=sigma, learnable=learnable_sigma)
        self.merge_vision_proj = nn.Sequential(
            nn.Linear(self.vision.feature_dim + 2 * final_channels, img_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.encoder = encoder
        self.drop = nn.Dropout(dropout)
        self.merge_head = FeedForwardNet(img_embed_dim + encoder.embed_dim, 1, 3)

    def _encode_image(self, x_img):
        x_img = x_img.to(next(self.vision.parameters()).dtype)
        stages = self.vision.encode(x_img)

        shared_feats = []
        for idx, pool in zip(self.vision.pool_stage_idxs, self.vision.center_pools):
            s = stages[idx]
            shared_feats.append(torch.cat([pool(s), F.adaptive_max_pool3d(s, 1).flatten(1)], dim=1))
        shared = self.vision.drop(torch.cat(shared_feats, dim=1))

        task_out = self.merge_task_conv(stages[self.vision.pool_stage_idxs[-1]])
        task = torch.cat(
            [self.merge_task_pool(task_out), F.adaptive_max_pool3d(task_out, 1).flatten(1)], dim=1
        )
        return self.merge_vision_proj(torch.cat([shared, task], dim=1)).float()

    def forward(self, x, task=None):
        """
        Parameters
        ----------
        x : dict
            Batch with "img" (B, C, D, H, W) and "tree_sample" (List[TreeSample]).
        task : str, optional
            Ignored; present for the trainers' ``forward(x, task)`` convention.

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 1).
        """
        z_img = self._encode_image(x["img"])
        z_graph = self.encoder(x["tree_sample"]).to(z_img.device)
        return self.merge_head(self.drop(torch.cat([z_img, z_graph], dim=1)))

    def half_vision(self):
        """Runs the image path in fp16, as JointProofreader.half_vision does."""
        for module in (self.vision, self.merge_task_conv, self.merge_task_pool, self.merge_vision_proj):
            module.half()
        return self

    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    def save_config(self, path):
        write_json(path, self.config)

    @classmethod
    def load(cls, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = dict(checkpoint["config"])
        config.pop("model_type")
        encoder = build_encoder(config.pop("encoder_type"), config.pop("encoder_kwargs"))
        model = cls(encoder, **config)
        return _load_state_dict(model, checkpoint)


def load_detector(path, map_location=None):
    """
    Loads a GraphMergeDetector or MultimodalMergeDetector checkpoint,
    dispatching on the modality recorded in its config.

    Parameters
    ----------
    path : str
        Path to a .pth checkpoint saved by ``save`` or a trainer.
    map_location : str or torch.device, optional

    Returns
    -------
    torch.nn.Module
    """
    config = torch.load(path, map_location="cpu", weights_only=False)["config"]
    if config["model_type"].endswith(MultimodalMergeDetector.modality):
        return MultimodalMergeDetector.load(path, map_location)
    return GraphMergeDetector.load(path, map_location)
