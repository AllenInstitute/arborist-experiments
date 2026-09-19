"""
Created on Fri Sep 19 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Arborist as a third modality for split correction. ProposalTreeEncoder turns
the rooted subgraph around each proposal into a latent vector, and
ArboristVisionHGAT fuses that vector with the geometric and image embeddings
that NewVisionHGAT already computes for each proposal node.

"""

import torch
import torch.nn as nn

from arborist.models.arborist import Arborist
from neuron_proofreader.models.new_gnn_models import NewVisionHGAT
from neuron_proofreader.models.simple_models import FeedForwardNet
from neuron_proofreader.split_proofreading import split_feature_extraction

# Column layout of the skeleton part of the proposal feature vector, as built
# in SkeletonFeatureExtractor.extract_proposal_features:
#   0 is_leaf2leaf | 1 count_nearby_leafs | 2 length / search_radius
#   3-4 radius_i, radius_j | 5-16 directionals at depths 16, 32, 64, 128
#   (3 dot products each) | 17+ image intensity profiles
DIRECTIONAL_COLUMNS = list(range(5, 17))


# --- Tree Encoder ---
class ProposalTreeEncoder(nn.Module):
    """
    Encodes proposal-rooted TreeSamples into fixed-size embeddings.

    Each sample is passed through Arborist (CurveEncoder + GraphTransformer).
    The two curves incident to the root, one per side of the proposal, are
    mean-pooled to form the proposal embedding. With ``use_tree_context`` the
    mean over all curves is concatenated and projected back to ``latent_dim``
    so the embedding also carries the surrounding morphology.

    Parameters
    ----------
    latent_dim : int, optional
        Arborist latent dimension and output embedding size. Default is 64.
    use_tree_context : bool, optional
        If True, concatenate the global tree embedding to the root embedding
        and project to ``latent_dim``. Default is True.
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
        use_tree_context=True,
        pretrained_curve_encoder_path=None,
        freeze_curve_encoder=True,
        **arborist_kwargs,
    ):
        super().__init__()
        self.config = {
            "latent_dim": latent_dim,
            "use_tree_context": use_tree_context,
            **arborist_kwargs,
        }
        self.latent_dim = latent_dim
        self.use_tree_context = use_tree_context
        self.arborist = Arborist(latent_dim=latent_dim, **arborist_kwargs)
        if use_tree_context:
            self.proj = nn.Linear(2 * latent_dim, latent_dim)
        else:
            self.proj = nn.Identity()

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
            z_tree = z_curves.mean(dim=0)
            if sample.root_curve_indices:
                idx = torch.as_tensor(
                    sample.root_curve_indices, device=z_curves.device
                )
                z_root = z_curves[idx].mean(dim=0)
            else:
                z_root = z_tree
            z = torch.cat([z_root, z_tree]) if self.use_tree_context else z_root
            zs.append(z)
        return self.proj(torch.stack(zs))


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

    Scores a proposal from its Arborist embedding alone, with no image or
    hand-crafted geometric features and no message passing between
    proposals. Comparing it against ArboristVisionHGAT isolates how much the
    skeleton modality contributes on its own.

    Parameters
    ----------
    latent_dim : int, optional
        Arborist latent dimension. Default is 64.
    hidden_dim : int, optional
        MLP hidden dimension. Default is 128.
    dropout : float, optional
        Dropout probability inside the MLP head. Default is 0.1.
    **encoder_kwargs
        Forwarded to ProposalTreeEncoder (use_tree_context,
        pretrained_curve_encoder_path, freeze_curve_encoder and Arborist
        kwargs).
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
            nn.Linear(latent_dim, hidden_dim),
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
        return self.head(self.encoder(samples))

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
    NewVisionHGAT whose proposal nodes are initialized from three
    modalities: geometric features, an image patch, and an Arborist
    embedding of the proposal-rooted skeleton subgraph.

    The forward pass expects ``input_dict["tree_samples"]``, a list of
    TreeSamples in proposal-index order, in addition to the keys
    NewVisionHGAT consumes.

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
    use_directionals : bool, optional
        If False, the hand-crafted directional features (columns 5-16 of the
        proposal feature vector) are removed before the proposal encoder, so
        the Arborist embedding replaces them rather than supplementing them.
        Default is True.
    **hgat_kwargs
        Forwarded to NewVisionHGAT (concat_heads, geometry_embed_dim, heads,
        hidden_dim, img_embed_dim, n_layers).
    """

    def __init__(
        self,
        patch_shape,
        tree_embed_dim=64,
        tree_encoder_kwargs=None,
        use_directionals=True,
        **hgat_kwargs,
    ):
        super().__init__(patch_shape, **hgat_kwargs)
        tree_encoder_kwargs = dict(tree_encoder_kwargs or {})
        self.config.update(
            {
                "model_type": "ArboristVisionHGAT",
                "tree_embed_dim": tree_embed_dim,
                "use_directionals": use_directionals,
                "tree_encoder_kwargs": {
                    k: v
                    for k, v in tree_encoder_kwargs.items()
                    if k != "pretrained_curve_encoder_path"
                },
            }
        )

        # Optionally drop the hand-crafted directionals from the geometric
        # features so the tree embedding is their replacement.
        n_features = split_feature_extraction.get_feature_dict()["proposal"]
        self.use_directionals = use_directionals
        if use_directionals:
            keep = list(range(n_features))
        else:
            keep = [c for c in range(n_features) if c not in DIRECTIONAL_COLUMNS]
            self.proposal_encoder = FeedForwardNet(
                len(keep), self.config["geometry_embed_dim"], 3
            )
        self.register_buffer("feature_columns", torch.tensor(keep), persistent=False)

        # Third modality and a fusion layer wide enough to take it
        self.tree_encoder = ProposalTreeEncoder(
            latent_dim=tree_embed_dim, **tree_encoder_kwargs
        )
        self.fusion = FeedForwardNet(
            self.config["img_embed_dim"]
            + self.config["geometry_embed_dim"]
            + tree_embed_dim,
            self.config["hidden_dim"],
            2,
        )

    def forward(self, input_dict):
        x_dict = input_dict["x_dict"]
        x_img = input_dict["img"]
        edge_index_dict = input_dict["edge_index_dict"]
        tree_samples = input_dict["tree_samples"]

        # Initial embeddings
        x_img = self.img_encoder(x_img)
        x_tree = self.tree_encoder(tree_samples).to(x_img.dtype)
        x_dict["branch"] = self.branch_encoder(x_dict["branch"])
        x_geo = x_dict["proposal"]
        if not self.use_directionals:
            x_geo = x_geo.index_select(1, self.feature_columns)
        x_dict["proposal"] = self.proposal_encoder(x_geo)

        # Fuse geometry, image and tree embeddings
        x_dict["proposal"] = torch.cat(
            (x_dict["proposal"], x_img, x_tree), dim=1
        )
        x_dict["proposal"] = self.fusion(x_dict["proposal"])

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
