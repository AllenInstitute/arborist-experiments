"""
Created on Mon August 10 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Training script for a graph-only merge detection model. Loads fragment graphs
and labeled merge/nonmerge sites from cloud storage, builds a MergeGraphDataset
per brain, and trains a MergeDetector (Arborist encoder + MLP head).

"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse

from torch.utils.data import DataLoader

from arborist.data.datasets import DatasetCollection
from arborist_experiments.merge_detection import (
    GraphTrainer,
    MergeDetector,
    MergeGraphDataset,
    collate_merge_samples,
    load_merge_sites,
)
from neuron_proofreader.configs import GraphConfig
from neuron_proofreader.fragments_graph import FragmentsGraph
from neuron_proofreader.merge_proofreading.merge_datamodules import (
    get_segmentation_id,
)


def main():
    # Save configs
    os.makedirs(output_dir, exist_ok=True)
    graph_config.save(output_dir)
    model.save_config(os.path.join(output_dir, "model_config.json"))

    # Create datasets
    print("\nLoading Train Dataset...")
    train_dataset = build_dataset_collection(train_ids, rebalance_classes=True)
    print("\nLoading Val Dataset...")
    val_dataset = build_dataset_collection(val_ids, rebalance_classes=False)
    print("\nDataset Summary")
    print("   Train:", train_dataset)
    print("   Val:  ", val_dataset)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_merge_samples,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_merge_samples,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    # Train
    trainer = init_trainer()
    trainer.run(train_loader, val_loader)


def build_dataset_collection(brain_ids, rebalance_classes):
    """
    Builds a DatasetCollection of MergeGraphDatasets, one per brain.

    Parameters
    ----------
    brain_ids : list[str]
        Brain IDs to load.
    rebalance_classes : bool
        Passed to MergeGraphDataset — True for train, False for val.

    Returns
    -------
    DatasetCollection
    """
    datasets = []
    for i, brain_id in enumerate(brain_ids, start=1):
        print(f"\n   Brain [{i}/{len(brain_ids)}]: {brain_id}")

        # Resolve paths
        segmentation_id = get_segmentation_id(sites_root_path, brain_id)
        sites_prefix = os.path.join(sites_root_path, brain_id, segmentation_id)
        swcs_path = os.path.join(
            swcs_root_path, brain_id, segmentation_id, "fragments"
        )

        # Load graph
        graph = FragmentsGraph(
            anisotropy=graph_config.anisotropy,
            min_cable_length=graph_config.min_cable_length,
            min_swc_pts=graph_config.min_swc_pts,
            node_spacing=graph_config.node_spacing,
            use_anisotropy=graph_config.use_anisotropy,
            verbose=graph_config.verbose,
        )
        graph.load(swcs_path)

        # Load sites and build dataset
        merge_nodes, nonmerge_nodes = load_merge_sites(graph, sites_prefix)
        print(f"      merge sites: {len(merge_nodes)}, nonmerge sites: {len(nonmerge_nodes)}")

        dataset = MergeGraphDataset(
            graph,
            merge_nodes,
            nonmerge_nodes,
            max_depth=subgraph_depth,
            rebalance_classes=rebalance_classes,
        )
        print(f"      {dataset}")
        datasets.append(dataset)

    return DatasetCollection(datasets)


def get_argparser():
    parser = argparse.ArgumentParser(
        description="Train a graph-only merge detection model."
    )

    # Run bookkeeping
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--run_cnt", type=int, default=1)

    # Encoder (must match pretrained CurveAutoencoder if loading weights)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--curve_n_layers", type=int, default=4)
    parser.add_argument("--d_token", type=int, default=128)
    parser.add_argument("--graph_n_layers", type=int, default=3)

    # Classifier head
    parser.add_argument("--hidden_dim", type=int, default=128)

    # Pretrained CurveEncoder
    parser.add_argument("--pretrained_curve_encoder_path", type=str, default=None)
    parser.add_argument("--freeze_curve_encoder", action="store_true", default=True)

    # Training hyperparameters
    parser.add_argument("--min_recall", type=float, default=0.9)

    return parser


def init_trainer():
    exp_name = f"merge_detection_graph_{args.version}_run{args.run_cnt}"
    trainer = GraphTrainer(
        model,
        model_name,
        output_dir,
        device="cuda",
        exp_name=exp_name,
        lr=lr,
        max_epochs=max_epochs,
        min_recall=args.min_recall,
        verbose=True,
    )
    if model_path:
        trainer.load_pretrained_weights(model_path)
    return trainer


if __name__ == "__main__":
    args, _ = get_argparser().parse_known_args()

    # Paths
    model_path = None
    output_dir = "/root/capsule/results"
    sites_root_path = (
        "gs://allen-nd-goog/automated_proofreading_dataset/curated_sites_05202026/"
    )
    swcs_root_path = sites_root_path

    # Brain IDs
    train_ids = ["653159", "715345", "730902", "789202", "794491", "802449"]
    val_ids = ["751473", "794495"]

    # Training hyperparameters
    batch_size = 32
    lr = 1e-4
    max_epochs = 150
    num_workers = 4
    subgraph_depth = 1000

    # Graph loading config
    graph_config = GraphConfig(
        anisotropy=(0.748, 0.748, 1.0),
        min_cable_length=0,
        node_spacing=5,
        use_anisotropy=False,
        verbose=True,
    )

    # Model
    model_name = "MergeDetector"
    model = MergeDetector(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        d_token=args.d_token,
        curve_n_layers=args.curve_n_layers,
        graph_n_layers=args.graph_n_layers,
        pretrained_curve_encoder_path=args.pretrained_curve_encoder_path,
        freeze_curve_encoder=args.freeze_curve_encoder,
    )

    main()
