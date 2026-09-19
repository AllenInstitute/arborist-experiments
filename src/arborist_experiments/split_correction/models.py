"""
Created on Fri Sep 19 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Arborist skeleton embeddings for split correction. ProposalTreeEncoder turns
the rooted subgraph around each proposal into a latent vector, and
ArboristVisionHGAT fuses that vector with the image patch embedding of
NewVisionHGAT in place of its hand-crafted features.

"""

import torch
import torch.nn as nn

from arborist.models.arborist import Arborist
from neuron_proofreader.models.new_gnn_models import NewVisionHGAT
from neuron_proofreader.models.simple_models import FeedForwardNet


def proposal_length_feature(tree_samples, device, dtype=torch.float32):
    """
    Log-scaled proposal length, one scalar per sample. Length is a property
    of the candidate edge that neither the rooted-subgraph curves nor the
    resized image patch expose, so it is fed to the models explicitly.

    Parameters
    ----------
    tree_samples : List[TreeSample]
        Samples produced by ProposalTreeFeatureExtractor.

    Returns
    -------
    torch.Tensor
        Shape (n_samples, 1), values ``log1p(length_in_microns)``.
    """
    lengths = torch.tensor(
        [s.proposal_length for s in tree_samples], device=device, dtype=dtype
    )
    return torch.log1p(lengths).unsqueeze(1)


# --- Tree Encoder ---
class ProposalTreeEncoder(nn.Module):
    """
    Encodes proposal-rooted TreeSamples into fixed-size embeddings.

    Each sample is passed through Arborist (CurveEncoder + GraphTransformer).
    The two curves incident to the root, one per side of the proposal, are
    mean-pooled to form the proposal embedding. The GraphTransformer has
    already contextualized them against the rest of the subgraph.

    Parameters
    ----------
    latent_dim : int, optional
        Arborist latent dimension and output embedding size. Default is 64.
    pretrained_curve_encoder_path : str, optional
        Path to a CurveAutoencoder checkpoint whose encoder weights are
        loaded into the CurveEncoder. Default is None.
    freeze_curve_encoder : bool, optional
        If True and a pretrained path is given, the CurveEncoder is frozen.
        Default is True.
    **arborist_kwargs
        Forwarded to the Arborist constructor (curve_segment_len,
        curve_d_token, curve_n_heads, curve_n_layers, curve_d_ff,
        graph_n_heads, graph_n_layers, graph_d_ff, dropout).
    """

    def __init__(
        self,
        latent_dim=64,
        pretrained_curve_encoder_path=None,
        freeze_curve_encoder=True,
        **arborist_kwargs,
    ):
        super().__init__()
        self.config = {"latent_dim": latent_dim, **arborist_kwargs}
        self.arborist = Arborist(latent_dim=latent_dim, **arborist_kwargs)
        if pretrained_curve_encoder_path is not None:
            load_pretrained_curve_encoder(
                self.arborist, pretrained_curve_encoder_path
            )
            if freeze_curve_encoder:
                self.freeze_curve_encoder()

    def freeze_curve_encoder(self):
        for param in self.arborist.curve_encoder.parameters():
            param.requires_grad = False

    def unfreeze_curve_encoder(self):
        for param in self.arborist.curve_encoder.parameters():
            param.requires_grad = True

    @torch._dynamo.disable
    def forward(self, tree_samples):
        """
        Parameters
        ----------
        tree_samples : List[TreeSample]
            One sample per proposal, in proposal-index order.

        Returns
        -------
        torch.Tensor
            Shape (n_proposals, latent_dim).
        """
        zs = []
        for sample in tree_samples:
            _, z_curves = self.arborist.encode(sample)
            idx = sample.root_curve_indices or range(len(z_curves))
            zs.append(z_curves[list(idx)].mean(dim=0))
        return torch.stack(zs)


def load_pretrained_curve_encoder(arborist, path):
    """
    Loads CurveEncoder weights from a CurveAutoencoder checkpoint into an
    Arborist model. The checkpoint stores the encoder under "encoder.*".

    Parameters
    ----------
    arborist : Arborist
        Model whose ``curve_encoder`` receives the weights.
    path : str
        Path to the CurveAutoencoder checkpoint (.pth).
    """
    checkpoint = torch.load(path, weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    encoder_state = {
        k[len("encoder."):]: v
        for k, v in state.items()
        if k.startswith("encoder.")
    }
    arborist.curve_encoder.load_state_dict(encoder_state)


# --- Standalone Baseline ---
class SplitDetector(nn.Module):
    """
    Graph-only proposal classifier used as a benchmark baseline.

    Scores a proposal from its Arborist embedding and its length, with no
    image and no message passing between proposals. Comparing it against
    ArboristVisionHGAT isolates how much the skeleton modality contributes
    on its own.

    Parameters
    ----------
    latent_dim : int, optional
        Arborist latent dimension. Default is 64.
    hidden_dim : int, optional
        MLP hidden dimension. Default is 128.
    dropout : float, optional
        Dropout probability inside the MLP head. Default is 0.1.
    **encoder_kwargs
        Forwarded to ProposalTreeEncoder (pretrained_curve_encoder_path,
        freeze_curve_encoder and Arborist kwargs).
    """

    def __init__(self, latent_dim=64, hidden_dim=128, dropout=0.1, **encoder_kwargs):
        super().__init__()
        self.config = {
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            **{
                k: v
                for k, v in encoder_kwargs.items()
                if k != "pretrained_curve_encoder_path"
            },
        }
        self.encoder = ProposalTreeEncoder(latent_dim=latent_dim, **encoder_kwargs)
        self.head = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : TensorDict or List[TreeSample]
            Batch of proposal-rooted subgraphs. A TensorDict holds them under
            "tree_sample" (inference); training passes a plain list.

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 1).
        """
        samples = x["tree_sample"] if isinstance(x, dict) else x
        z = self.encoder(samples)
        length = proposal_length_feature(samples, z.device, z.dtype)
        return self.head(torch.cat((z, length), dim=1))

    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**checkpoint["config"])
        state_dict = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state_dict)
        return model


# --- HGAT with Arborist Modality ---
class ArboristVisionHGAT(NewVisionHGAT):
    """
    NewVisionHGAT with no hand-crafted features. Proposal nodes are
    initialized from an image patch, an Arborist embedding of the
    proposal-rooted skeleton subgraph, and the proposal length. The subgraph
    is the only place skeleton geometry enters the model; length is passed
    explicitly because neither the curves nor the resized patch expose it. Branch nodes are relays: they start from one
    learned vector and acquire their state through message passing from the
    proposals they touch, so proposals sharing a fragment can still exchange
    information. The hand-crafted proposal and branch features in the batch
    are ignored.

    The forward pass expects ``input_dict["tree_samples"]`` (TreeSamples in
    proposal-index order) in addition to the keys NewVisionHGAT consumes.

    Parameters
    ----------
    patch_shape : Tuple[int]
        Shape of input image patch.
    tree_embed_dim : int, optional
        Arborist latent dimension. Default is 64.
    tree_encoder_kwargs : dict, optional
        Extra keyword arguments for ProposalTreeEncoder (e.g.
        pretrained_curve_encoder_path, freeze_curve_encoder, curve_n_layers).
        Default is None.
    **hgat_kwargs
        Forwarded to NewVisionHGAT (concat_heads, heads, hidden_dim,
        img_embed_dim, n_layers). ``geometry_embed_dim`` is accepted but
        unused since there is no geometric feature encoder.
    """

    def __init__(
        self,
        patch_shape,
        tree_embed_dim=64,
        tree_encoder_kwargs=None,
        **hgat_kwargs,
    ):
        super().__init__(patch_shape, **hgat_kwargs)
        tree_encoder_kwargs = dict(tree_encoder_kwargs or {})
        self.config.update(
            {
                "model_type": "ArboristVisionHGAT",
                "tree_embed_dim": tree_embed_dim,
                "tree_encoder_kwargs": {
                    k: v
                    for k, v in tree_encoder_kwargs.items()
                    if k != "pretrained_curve_encoder_path"
                },
            }
        )

        # No hand-crafted proposal features: the tree embedding replaces them
        self.proposal_encoder = None

        # Skeleton modality and a fusion layer over image + tree + length
        self.tree_encoder = ProposalTreeEncoder(
            latent_dim=tree_embed_dim, **tree_encoder_kwargs
        )
        self.fusion = FeedForwardNet(
            self.config["img_embed_dim"] + tree_embed_dim + 1,
            self.config["hidden_dim"],
            2,
        )

        # Branch nodes are relays: one learned embedding shared by all
        # branches, refined only through message passing.
        self.branch_encoder = None
        self.branch_relay = nn.Parameter(torch.zeros(self.config["hidden_dim"]))
        nn.init.normal_(self.branch_relay, std=0.02)

    def forward(self, input_dict):
        x_dict = input_dict["x_dict"]
        x_img = input_dict["img"]
        edge_index_dict = input_dict["edge_index_dict"]
        tree_samples = input_dict["tree_samples"]

        # Initial embeddings
        x_img = self.img_encoder(x_img)
        x_tree = self.tree_encoder(tree_samples).to(x_img.dtype)
        n_branches = x_dict["branch"].shape[0]
        x_dict["branch"] = (
            self.branch_relay.to(x_img.dtype).unsqueeze(0).expand(n_branches, -1)
        )

        # Fuse image and tree embeddings with the proposal length
        x_len = proposal_length_feature(tree_samples, x_img.device, x_img.dtype)
        x_dict["proposal"] = self.fusion(torch.cat((x_img, x_tree, x_len), dim=1))

        # Message passing with residual connections
        for gat, proj in zip(self.gats, self.head_projs):
            x_dict_new = gat(x_dict, edge_index_dict)
            if self.concat_heads:
                x_dict_new = proj(x_dict_new)
            x_dict = {
                key: torch.relu(x_dict_new[key]) + x_dict[key]
                if key in x_dict_new
                else x_dict[key]
                for key in x_dict
            }
        return self.output(x_dict["proposal"])

