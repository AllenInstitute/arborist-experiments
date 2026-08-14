"""
Created on Mon August 10 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

GraphTrainer: thin Trainer subclass for models that consume List[TreeSample]
inputs instead of image tensors.
"""

import os
import torch

from neuron_proofreader.machine_learning.train import Trainer


class GraphTrainer(Trainer):
    """
    Trainer subclass for graph-input models (List[TreeSample] → logits).
    """

    def forward_pass(self, samples, y):
        """
        Parameters
        ----------
        samples : List[TreeSample]
            Batch of rooted subgraphs.
        y : torch.Tensor
            Ground-truth labels, shape (B, 1).
        """
        y = y.to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y_pred = self.model(samples)
            loss = self.criterion(y_pred, y)
        return y, y_pred, loss

    def run(self, train_dataloader, val_dataloader):
        """
        Full training loop.

        Parameters
        ----------
        train_dataloader : DataLoader
            Dataloader that iterates over training examples.
        val_dataloader : DataLoader
            Dataloader that iterates over validation examples.
        """
        exp_name = os.path.basename(os.path.normpath(self.log_dir))
        print("\nExperiment:", exp_name)
        for epoch in range(self.max_epochs):
            # Resample negatives at the start of each epoch
            _rebuild_index(train_dataloader.dataset)

            # Run epoch
            train_stats = self.train_step(train_dataloader, epoch)
            val_stats = self.validate_step(val_dataloader, epoch)
            self.scheduler.step()

            # Report results
            new_best = self.check_model_performance(val_stats, epoch)
            print(f"Epoch: {epoch} " + ("- New Best!" if new_best else " "))
            self.report_stats(train_stats, is_train=True)
            self.report_stats(val_stats, is_train=False)
            print()

def _rebuild_index(dataset):
    """Calls rebuild_index() on a MergeGraphDataset or a collection of them."""
    if hasattr(dataset, "rebuild_index"):
        dataset.rebuild_index()
    elif hasattr(dataset, "datasets"):
        for ds in dataset.datasets:
            if hasattr(ds, "rebuild_index"):
                ds.rebuild_index()
