"""Level-synchronous batching for the child-sum Tree-LSTM.

Nodes are renumbered by *height* (0 = tip, height = 1 + max child height), so every node at height h
has all its children strictly below h and the whole batch can be walked in `max_height + 1` cell
calls regardless of how many trees it holds. On this cohort collapsed height is median 14 / p90 24 /
max 46, so a batch of 64 trees costs ~25 sequential steps instead of ~10000 per-node ones.

Renumbering makes each height a contiguous slice, so the per-step gather of accumulated child sums is
a view rather than an index_select, and child contributions are pushed into the parents' accumulator
rows with `index_add_` as soon as a height finishes -- a child's state is never read again, so no
global per-node state buffer is needed apart from `z_all`, which is written in disjoint slices and not
read until the end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def node_heights(parents: np.ndarray) -> np.ndarray:
    """Height of every node: 0 at a tip, 1 + max(child heights) otherwise.

    Kahn-style, so it works whatever order the collapsed parent array happens to be in -- `collapse`
    keeps ascending order but the topology augmentations rewire across levels.
    """
    n = len(parents)
    remaining = np.bincount(parents[parents >= 0], minlength=n)
    height = np.zeros(n, dtype=np.int64)
    queue = list(np.flatnonzero(remaining == 0))
    seen = 0
    while queue:
        node = queue.pop()
        seen += 1
        p = int(parents[node])
        if p < 0:
            continue
        height[p] = max(height[p], height[node] + 1)
        remaining[p] -= 1
        if remaining[p] == 0:
            queue.append(p)
    if seen != n:
        raise ValueError("parent array contains a cycle")
    return height


@dataclass
class TreeBatch:
    """A batch of trees flattened into one node array, ordered by height.

    Attributes
    ----------
    x : torch.Tensor
        (N, F) float32 node features, height-ordered.
    parent : torch.Tensor
        (N,) int64 parent index into the same ordering, -1 at a root.
    bounds : np.ndarray
        (max_height + 2,) start offset of each height band; band h is [bounds[h], bounds[h + 1]).
    roots : torch.Tensor
        (B,) int64 node index of each tree's root, in the order the trees were passed in.
    n_trees : int
    """

    x: torch.Tensor
    parent: torch.Tensor
    bounds: np.ndarray
    roots: torch.Tensor
    n_trees: int

    def to(self, device) -> "TreeBatch":
        return TreeBatch(
            x=self.x.to(device, non_blocking=True),
            parent=self.parent.to(device, non_blocking=True),
            bounds=self.bounds,
            roots=self.roots.to(device, non_blocking=True),
            n_trees=self.n_trees,
        )

    @property
    def n_nodes(self) -> int:
        return len(self.parent)

    @property
    def max_height(self) -> int:
        return len(self.bounds) - 2


def collate(views: list[tuple[np.ndarray, np.ndarray]]) -> TreeBatch:
    """Pack `(parents, features)` views into a height-ordered `TreeBatch`."""
    sizes = [len(p) for p, _ in views]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    n = int(offsets[-1])

    parents = np.empty(n, dtype=np.int64)
    heights = np.empty(n, dtype=np.int64)
    roots = np.empty(len(views), dtype=np.int64)
    for i, (p, _) in enumerate(views):
        lo = int(offsets[i])
        parents[lo:lo + len(p)] = np.where(p >= 0, p + lo, -1)
        heights[lo:lo + len(p)] = node_heights(p)
        roots[i] = lo + int(np.flatnonzero(p < 0)[0])

    order = np.argsort(heights, kind="stable")
    new_of_old = np.empty(n, dtype=np.int64)
    new_of_old[order] = np.arange(n)

    old_parent = parents[order]
    new_parent = np.where(old_parent >= 0, new_of_old[old_parent], -1)
    counts = np.bincount(heights, minlength=heights.max() + 1)
    bounds = np.concatenate([[0], np.cumsum(counts)])

    x = np.concatenate([f for _, f in views], axis=0)[order]
    return TreeBatch(
        x=torch.from_numpy(np.ascontiguousarray(x)),
        parent=torch.from_numpy(new_parent),
        bounds=bounds,
        roots=torch.from_numpy(new_of_old[roots]),
        n_trees=len(views),
    )


def tree_lstm_forward(cell1, cell2, x_tilde: torch.Tensor, batch: TreeBatch) -> torch.Tensor:
    """Run both cells bottom-up over `batch`; return each tree's root second cell state z.

    Equations 1-4 of the paper: LSTM1 consumes the transformed node feature and its children's
    (h1, c1); LSTM2 consumes LSTM1's cell state and its children's (h2, z). The representation is z
    at the root -- the *cell* state of the second cell, not its hidden state.
    """
    n, d = batch.n_nodes, cell1.d_hidden
    sums = [x_tilde.new_zeros((n, d)) for _ in range(4)]
    sum_h1, sum_c1, sum_h2, sum_z = sums
    z_all = x_tilde.new_zeros((n, d))

    parent = batch.parent
    for h in range(batch.max_height + 1):
        lo, hi = int(batch.bounds[h]), int(batch.bounds[h + 1])
        if lo == hi:
            continue
        # Cloned, not sliced: a slice shares its base's version counter, and the index_add_ below
        # bumps it. The cells' Linear layers save their inputs for backward, so autograd would then
        # reject the saved views.
        band = [s[lo:hi].clone() for s in sums]
        h1, c1 = cell1(x_tilde[lo:hi], band[0], band[1])
        h2, z = cell2(c1, band[2], band[3])
        z_all[lo:hi] = z

        p = parent[lo:hi]
        keep = p >= 0
        if bool(keep.any()):
            dst = p[keep]
            sum_h1.index_add_(0, dst, h1[keep])
            sum_c1.index_add_(0, dst, c1[keep])
            sum_h2.index_add_(0, dst, h2[keep])
            sum_z.index_add_(0, dst, z[keep])

    return z_all[batch.roots]
