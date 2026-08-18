"""
Created on Sun August 10 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

MergeDetector: Arborist encoder + two-layer MLP classification head for
detecting merge errors from graph structure alone.
"""

import torch
import torch.nn as nn

from arborist.models.arborist import Arborist
from arborist.utils.util import write_json


class MergeDetector(nn.Module):
    """
    Binary merge-detection model based on graph structure only.

    Parameters
    ----------
    latent_dim : int, optional
        Arborist latent dimension (shared encoder output / MLP input).
        Default is 64.
    hidden_dim : int, optional
        MLP hidden dimension. Default is 128.
    dropout : float, optional
        Dropout probability applied inside the MLP head. Default is 0.1.
    pretrained_curve_encoder_path : str, optional
        Path to a CurveAutoencoder checkpoint (.pth). When provided, the
        CurveEncoder weights are loaded from the checkpoint. The Arborist
        kwargs must match the architecture used during pretraining.
        Default is None.
    freeze_curve_encoder : bool, optional
        If True (and a pretrained path is given), the CurveEncoder weights
        are frozen so only the GraphTransformer and head are trained. Call
        unfreeze_curve_encoder() to enable end-to-end fine-tuning later.
        Default is True.
    **arborist_kwargs
        Additional keyword arguments forwarded to the Arborist constructor
        (e.g. curve_segment_len, curve_d_token, curve_n_heads, curve_n_layers, curve_d_ff,
        graph_n_heads, graph_n_layers, graph_d_ff).
    """

    def __init__(
        self,
        latent_dim=64,
        hidden_dim=128,
        dropout=0.1,
        pretrained_curve_encoder_path=None,
        freeze_curve_encoder=True,
        **arborist_kwargs,
    ):
        super().__init__()
        self.config = {
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            **arborist_kwargs,
        }
        self.encoder = Arborist(
            latent_dim=latent_dim,
            dropout=dropout,
            **arborist_kwargs,
        )
        if pretrained_curve_encoder_path is not None:
            self._load_pretrained_curve_encoder(pretrained_curve_encoder_path)
            if freeze_curve_encoder:
                self.freeze_curve_encoder()

        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _load_pretrained_curve_encoder(self, path):
        """
        Loads CurveEncoder weights from a CurveAutoencoder checkpoint.

        The CurveAutoencoder checkpoint stores the encoder under "encoder.*"
        keys. This method strips that prefix and loads the weights into
        self.encoder.curve_encoder (the CurveEncoder inside Arborist).

        The Arborist kwargs passed to MergeDetector (curve_segment_len, curve_d_token,
        curve_n_heads, curve_n_layers, curve_d_ff, latent_dim) must match the
        architecture used when training the CurveAutoencoder.

        Parameters
        ----------
        path : str
            Path to a CurveAutoencoder checkpoint (.pth).
        """
        checkpoint = torch.load(path, weights_only=False)
        full_sd = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        curve_encoder_sd = {
            k[len("encoder."):]: v
            for k, v in full_sd.items()
            if k.startswith("encoder.")
        }
        self.encoder.curve_encoder.load_state_dict(curve_encoder_sd)

    def freeze_curve_encoder(self):
        """Freezes CurveEncoder weights so only the GraphTransformer and head train."""
        for param in self.encoder.curve_encoder.parameters():
            param.requires_grad = False

    def unfreeze_curve_encoder(self):
        """Unfreezes CurveEncoder weights for end-to-end fine-tuning."""
        for param in self.encoder.curve_encoder.parameters():
            param.requires_grad = True

    @torch._dynamo.disable
    def forward(self, samples):
        """
        Encodes a batch of rooted subgraphs and returns merge logits.

        Parameters
        ----------
        samples : List[TreeSample]
            Batch of rooted subgraphs as returned by MergeGraphDataset.

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 1). Apply sigmoid for probabilities.
        """
        z_trees = []
        for sample in samples:
            _, z_curves = self.encoder.encode(sample)
            if sample.root_curve_indices:
                idx = torch.tensor(sample.root_curve_indices, device=z_curves.device)
                z = z_curves[idx].mean(dim=0)
            else:
                z = z_curves.mean(dim=0)
            z_trees.append(z)
        z = torch.stack(z_trees)  # (B, latent_dim)
        return self.head(z)       # (B, 1)

    def save_config(self, path):
        """
        Saves model config to a JSON file.

        Parameters
        ----------
        path : str
            Destination path (e.g. .../model_config.json).
        """
        write_json(path, self.config)

    @classmethod
    def load(cls, path):
        """
        Loads a MergeDetector from a checkpoint saved by GraphTrainer.

        Parameters
        ----------
        path : str
            Path to .pth checkpoint file.

        Returns
        -------
        MergeDetector
        """
        checkpoint = torch.load(path, weights_only=False)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
