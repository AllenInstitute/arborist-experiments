"""TreeMoCo encoder -- Chen et al. 2022, section 3.2.

phi (a 3-layer MLP) lifts the 30 raw node features, eta (two stacked Tree-LSTM cells) folds the tree
bottom-up, and the representation is z at the root: the *cell* state of the second cell, per Eqn 4.

One documented deviation. Tai et al.'s child-sum Tree-LSTM gives every child its own forget gate,
f_kj = sigmoid(W x_k + U h_j + b), which needs the parent's input x_k while visiting the child. That
is expressible for LSTM1, whose input x~_k is precomputed for every node, but not for LSTM2, whose
input is c_{k,1} -- only available once the parent itself has been visited, i.e. after its children.
The paper's Eqns 1-4 sidestep this by passing the *sums* into each cell, so both cells here use one
forget gate over the summed child state. That is what the paper writes; it is not Tai's variant.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batching import TreeBatch, tree_lstm_forward
from .tree_features import N_FEATURES


class TreeLSTMCell(nn.Module):
    """Sum-based child-sum cell: (x, sum_h, sum_c) -> (h, c)."""

    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.d_hidden = d_hidden
        self.wx = nn.Linear(d_in, 4 * d_hidden)
        self.uh = nn.Linear(d_hidden, 4 * d_hidden, bias=False)

    def forward(self, x, sum_h, sum_c):
        i, f, o, u = (self.wx(x) + self.uh(sum_h)).chunk(4, dim=-1)
        c = torch.sigmoid(i) * torch.tanh(u) + torch.sigmoid(f) * sum_c
        return torch.sigmoid(o) * torch.tanh(c), c


class TreeEncoder(nn.Module):
    """phi + eta. Maps a `TreeBatch` to one d_hidden vector per tree."""

    def __init__(self, d_in: int = N_FEATURES, d_hidden: int = 128, d_phi: int = 128):
        super().__init__()
        # BatchNorm on the input layer is doing real work here, not just regularizing: the raw
        # features mix coordinates and arc lengths in microns (hundreds) with a contraction ratio in
        # [0, 1] and a 0/1 flag.
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_phi), nn.BatchNorm1d(d_phi), nn.ReLU(inplace=True),
            nn.Linear(d_phi, d_phi), nn.BatchNorm1d(d_phi), nn.ReLU(inplace=True),
            nn.Linear(d_phi, d_phi),
        )
        self.cell1 = TreeLSTMCell(d_phi, d_hidden)
        self.cell2 = TreeLSTMCell(d_hidden, d_hidden)
        self.d_hidden = d_hidden

    def forward(self, batch: TreeBatch) -> torch.Tensor:
        return tree_lstm_forward(self.cell1, self.cell2, self.phi(batch.x), batch)


class ProjectionHead(nn.Module):
    """2-layer MLP to a 128-d L2-normalized embedding.

    Table 5 attributes ~1.6pp to the BatchNorm and shows the model collapsing without it, so it is on
    by default and `batch_norm=False` exists only to reproduce that ablation.
    """

    def __init__(self, d_in: int = 128, d_hidden: int = 128, d_out: int = 128,
                 batch_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d_in, d_hidden)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(d_hidden))
        layers += [nn.ReLU(inplace=True), nn.Linear(d_hidden, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=1)


class TreeMoCoNet(nn.Module):
    """Encoder + projection head; one branch of the MoCo pair."""

    def __init__(self, d_in: int = N_FEATURES, d_hidden: int = 128, d_phi: int = 128,
                 d_proj: int = 128, projector_bn: bool = True):
        super().__init__()
        self.encoder = TreeEncoder(d_in=d_in, d_hidden=d_hidden, d_phi=d_phi)
        self.head = ProjectionHead(d_in=d_hidden, d_hidden=d_hidden, d_out=d_proj,
                                   batch_norm=projector_bn)

    def forward(self, batch: TreeBatch) -> torch.Tensor:
        return self.head(self.encoder(batch))


class TreeClassifier(nn.Module):
    """Pretrained encoder + linear head, for the fine-tuning protocol."""

    def __init__(self, encoder: TreeEncoder, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.d_hidden, n_classes)

    def forward(self, batch: TreeBatch) -> torch.Tensor:
        return self.fc(self.encoder(batch))
