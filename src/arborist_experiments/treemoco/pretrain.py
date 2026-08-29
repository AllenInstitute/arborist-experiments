"""MoCo v2 contrastive pretraining -- Chen et al. 2022, section 3.3.

Importable so the standalone driver and the per-fold probe share one code path.

Defaults depart from MoCo's published ImageNet settings because the cohort is three orders of
magnitude smaller: the queue cannot exceed the dataset (1618 trees), and a 0.999 momentum barely
moves the key encoder inside 200 epochs of 25 steps. There is no shuffling-BN -- it needs multiple
devices, and this runs on one T4 -- so the query and key batches share BN statistics, the known MoCo
leak. Node-level BN over ~10000 nodes per batch is a much weaker signature of batch membership than
image-level BN, but it is a real caveat.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .augment import AugmentConfig, augment, identity_view
from .batching import collate
from .model import TreeMoCoNet
from .tree_features import CollapsedTree, n_features


@dataclass
class PretrainConfig:
    d_hidden: int = 128
    d_phi: int = 128
    d_proj: int = 128
    projector_bn: bool = True

    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.2
    queue_size: int = 1024
    momentum: float = 0.99
    grad_clip: float = 5.0

    num_workers: int = 4
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpochStats:
    epoch: int
    loss: float
    top1: float
    embed_std: float
    lr: float


class _PairDataset(Dataset):
    """Two independently augmented views of one tree."""

    def __init__(self, trees: list[CollapsedTree], aug: AugmentConfig):
        self.trees = trees
        self.aug = aug
        self._rng: np.random.Generator | None = None

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, i):
        if self._rng is None:
            # torch reseeds every worker per epoch, so deriving from it keeps the run reproducible
            # from the global torch seed while giving each worker a distinct stream.
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        t = self.trees[i]
        return augment(t, self.aug, self._rng), augment(t, self.aug, self._rng)


def _collate_pairs(items):
    return collate([a for a, _ in items]), collate([b for _, b in items])


class _SingleDataset(Dataset):
    def __init__(self, trees: list[CollapsedTree], aug: AugmentConfig):
        self.trees = trees
        self.aug = aug

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, i):
        return identity_view(self.trees[i], self.aug)


def _collate_single(items):
    return collate(list(items))


@torch.no_grad()
def _momentum_update(query: nn.Module, key: nn.Module, m: float) -> None:
    for pq, pk in zip(query.parameters(), key.parameters()):
        pk.mul_(m).add_(pq.detach(), alpha=1.0 - m)
    for bq, bk in zip(query.buffers(), key.buffers()):
        bk.copy_(bq)


def pretrain(trees: list[CollapsedTree], aug: AugmentConfig, cfg: PretrainConfig,
             log=print) -> tuple[dict, list[EpochStats]]:
    """Contrastively pretrain on `trees`. Returns the query network's state dict and per-epoch stats."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    net_q = TreeMoCoNet(d_in=n_features(aug.include_type), d_hidden=cfg.d_hidden, d_phi=cfg.d_phi,
                        d_proj=cfg.d_proj, projector_bn=cfg.projector_bn).to(device)
    net_k = copy.deepcopy(net_q)
    for p in net_k.parameters():
        p.requires_grad_(False)

    if cfg.epochs == 0:
        # Random-encoder control: a Tree-LSTM over these node features is a strong random
        # projection of the morphology, so this is the baseline any pretraining gain is measured
        # against, not a degenerate case.
        log("epochs=0: returning the randomly initialized encoder (control run)")
        return {k: v.detach().cpu() for k, v in net_q.state_dict().items()}, []

    queue = F.normalize(torch.randn(cfg.queue_size, cfg.d_proj, device=device), dim=1)
    queue_ptr = 0

    opt = torch.optim.AdamW(net_q.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(
        _PairDataset(trees, aug), batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, collate_fn=_collate_pairs,
        persistent_workers=cfg.num_workers > 0,
    )
    if len(loader) == 0:
        raise ValueError(f"batch_size {cfg.batch_size} exceeds {len(trees)} trees")

    history: list[EpochStats] = []
    for epoch in range(cfg.epochs):
        lr = 0.5 * cfg.lr * (1.0 + math.cos(math.pi * epoch / cfg.epochs))
        for group in opt.param_groups:
            group["lr"] = lr

        net_q.train()
        net_k.train()
        tot_loss = tot_top1 = tot_std = 0.0
        for batch_q, batch_k in loader:
            q = net_q(batch_q.to(device))
            with torch.no_grad():
                _momentum_update(net_q, net_k, cfg.momentum)
                k = net_k(batch_k.to(device))

            l_pos = (q * k).sum(dim=1, keepdim=True)
            l_neg = q @ queue.t()
            logits = torch.cat([l_pos, l_neg], dim=1) / cfg.temperature
            loss = F.cross_entropy(logits, torch.zeros(len(q), dtype=torch.long, device=device))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net_q.parameters(), cfg.grad_clip)
            opt.step()

            with torch.no_grad():
                n = len(k)
                idx = (queue_ptr + torch.arange(n, device=device)) % cfg.queue_size
                queue[idx] = k
                queue_ptr = int((queue_ptr + n) % cfg.queue_size)
                tot_loss += loss.item()
                tot_top1 += (logits.argmax(dim=1) == 0).float().mean().item()
                tot_std += q.std(dim=0).mean().item()

        nb = len(loader)
        stats = EpochStats(epoch=epoch, loss=tot_loss / nb, top1=tot_top1 / nb,
                           embed_std=tot_std / nb, lr=lr)
        history.append(stats)
        log(f"epoch {epoch + 1}/{cfg.epochs} loss {stats.loss:.4f} "
            f"top1 {stats.top1:.3f} embed_std {stats.embed_std:.4f} lr {lr:.2e}")

    return {k: v.detach().cpu() for k, v in net_q.state_dict().items()}, history


@torch.no_grad()
def embed(encoder: nn.Module, trees: list[CollapsedTree], aug: AugmentConfig,
          cfg: PretrainConfig, batch_size: int | None = None) -> np.ndarray:
    """Root z for every tree, un-augmented, in the order given. Shape (n_trees, d_hidden)."""
    device = torch.device(cfg.device)
    encoder = encoder.to(device).eval()
    loader = DataLoader(
        _SingleDataset(trees, aug), batch_size=batch_size or cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=_collate_single,
    )
    return np.concatenate([encoder(b.to(device)).cpu().numpy() for b in loader], axis=0)
