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

# Column layout of the proposal feature vector built by FeaturePipeline:
#   0 is_leaf2leaf | 1 count_nearby_leafs | 2 length / search_radius
#   3-4 radius_i, radius_j | 5-16 directionals at depths 16, 32, 64, 128
#   (3 dot products each) | 17-66 image intensity profiles along the two
#   branches and the proposal line
N_SKELETON_FEATURES = 17
SKELETON_COLUMNS = list(range(N_SKELETON_FEATURES))
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
    NewVisionHGAT with no hand-crafted features. Proposal nodes are
    initialized from an image patch and an Arborist embedding of the
    proposal-rooted skeleton subgraph. Branch nodes are initialized from
    Arborist CurveEncoder embeddings of their skeleton paths, sharing weights
    with the tree encoder. Every hand-crafted input of NewVisionHGAT can be
    re-enabled individually for ablations.

    The forward pass expects ``input_dict["tree_samples"]`` (TreeSamples in
    proposal-index order) and ``input_dict["branch_curves"]`` (first-
    difference curves in branch-index order), in addition to the keys
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
    use_skeleton_features : bool, optional
        If True, the 17 hand-crafted skeleton features of each proposal
        (leaf flags, nearby-leaf count, length, radii, directionals) are kept
        as an input alongside the Arborist embedding. Default is False: the
        embedding replaces them entirely.
    use_image_profiles : bool, optional
        If True, the hand-crafted image intensity profiles sampled along the
        branches and proposal line are kept as a proposal input. Default is
        False; the CNN patch embedding is the image modality.
    use_branch_features : bool, optional
        If True, branch nodes keep NewVisionHGAT's hand-crafted features
        (mean radius, length). Default is False: branch nodes are encoded
        from their skeleton curves with the shared CurveEncoder.
    **hgat_kwargs
        Forwarded to NewVisionHGAT (concat_heads, geometry_embed_dim, heads,
        hidden_dim, img_embed_dim, n_layers).
    """

    def __init__(
        self,
        patch_shape,
        tree_embed_dim=64,
        tree_encoder_kwargs=None,
        use_skeleton_features=False,
        use_image_profiles=False,
        use_branch_features=False,
        **hgat_kwargs,
    ):
        super().__init__(patch_shape, **hgat_kwargs)
        tree_encoder_kwargs = dict(tree_encoder_kwargs or {})
        self.config.update(
            {
                "model_type": "ArboristVisionHGAT",
                "tree_embed_dim": tree_embed_dim,
                "use_skeleton_features": use_skeleton_features,
                "use_image_profiles": use_image_profiles,
                "use_branch_features": use_branch_features,
                "tree_encoder_kwargs": {
                    k: v
                    for k, v in tree_encoder_kwargs.items()
                    if k != "pretrained_curve_encoder_path"
                },
            }
        )

        # Select which columns of the proposal feature vector survive. By
        # default the hand-crafted skeleton features are dropped and the
        # Arborist embedding takes their place; image profiles are kept.
        n_features = split_feature_extraction.get_feature_dict()["proposal"]
        keep = []
        if use_skeleton_features:
            keep += SKELETON_COLUMNS
        if use_image_profiles:
            keep += list(range(N_SKELETON_FEATURES, n_features))
        self.use_skeleton_features = use_skeleton_features
        self.use_image_profiles = use_image_profiles
        self.register_buffer("feature_columns", torch.tensor(keep, dtype=torch.long), persistent=False)
        geometry_embed_dim = self.config["geometry_embed_dim"] if keep else 0
        if keep and len(keep) != n_features:
            self.proposal_encoder = FeedForwardNet(len(keep), geometry_embed_dim, 3)
        elif not keep:
            self.proposal_encoder = None

        # Third modality and a fusion layer wide enough to take it
        self.tree_encoder = ProposalTreeEncoder(
            latent_dim=tree_embed_dim, **tree_encoder_kwargs
        )
        self.fusion = FeedForwardNet(
            self.config["img_embed_dim"] + geometry_embed_dim + tree_embed_dim,
            self.config["hidden_dim"],
            2,
        )

        # Branch nodes: learned curve embeddings instead of hand-crafted
        # features. The CurveEncoder is shared with the tree encoder.
        self.use_branch_features = use_branch_features
        if not use_branch_features:
            self.branch_encoder = None
            self.branch_curve_proj = nn.Linear(tree_embed_dim, self.config["hidden_dim"])

    def encode_branch_curves(self, curves):
        """
        Embeds branch skeleton curves with the shared CurveEncoder.

        Parameters
        ----------
        curves : List[numpy.ndarray]
            First-difference curves in branch-index order.

        Returns
        -------
        torch.Tensor
            Shape (n_branches, hidden_dim).
        """
        arborist = self.tree_encoder.arborist
        if len(curves) == 0:
            device = next(self.parameters()).device
            return torch.zeros(0, self.config["hidden_dim"], device=device)
        diffs, mask = arborist._collate_curves(curves)
        z, _ = arborist.curve_encoder(diffs, mask)
        return self.branch_curve_proj(z)

    def forward(self, input_dict):
        x_dict = input_dict["x_dict"]
        x_img = input_dict["img"]
        edge_index_dict = input_dict["edge_index_dict"]
        tree_samples = input_dict["tree_samples"]

        # Initial embeddings
        x_img = self.img_encoder(x_img)
        x_tree = self.tree_encoder(tree_samples).to(x_img.dtype)
        if self.branch_encoder is not None:
            x_dict["branch"] = self.branch_encoder(x_dict["branch"])
        else:
            x_dict["branch"] = self.encode_branch_curves(
                input_dict["branch_curves"]
            ).to(x_img.dtype)

        # Fuse the surviving proposal features, image and tree embeddings
        parts = [x_img, x_tree]
        if self.proposal_encoder is not None:
            x_feat = x_dict["proposal"].index_select(1, self.feature_columns)
            parts.insert(0, self.proposal_encoder(x_feat))
        x_dict["proposal"] = self.fusion(torch.cat(parts, dim=1))

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
