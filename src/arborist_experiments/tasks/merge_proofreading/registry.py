"""
Created on Mon November 24 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Rebuilds merge detectors from their configs. Encoders and rule functions
are recorded in configs by import path, so any importable class or function
can be used without registering it here.
"""

import importlib

import torch

from arborist_experiments.tasks.merge_proofreading.detectors import (
    HeuristicMergeDetector,
    LearnedMergeDetector,
)


def resolve(qualname):
    """
    Imports a class or function from its "module.name" path.

    Parameters
    ----------
    qualname : str
        Import path as recorded by the detectors' configs.

    Returns
    -------
    object
    """
    module, _, name = qualname.rpartition(".")
    return getattr(importlib.import_module(module), name)


def build_encoder(spec):
    """
    Instantiates an encoder from a config entry.

    Parameters
    ----------
    spec : dict or None
        {"type": import path, "kwargs": constructor kwargs}, or None.

    Returns
    -------
    torch.nn.Module or None
    """
    return None if spec is None else resolve(spec["type"])(**spec["kwargs"])


def build_detector(config):
    """
    Instantiates an untrained detector from a config saved by ``save``.

    Parameters
    ----------
    config : dict
        A detector's ``config``.

    Returns
    -------
    MergeDetector
    """
    detector_cls = resolve(config["detector_type"])
    if detector_cls is LearnedMergeDetector:
        return LearnedMergeDetector(
            graph_encoder=build_encoder(config["graph_encoder"]),
            vision_encoder=build_encoder(config["vision_encoder"]),
            dropout=config["dropout"],
        )
    if detector_cls is HeuristicMergeDetector:
        return HeuristicMergeDetector(
            resolve(config["rules"]), weights=config["weights"], **config["rule_kwargs"]
        )
    raise ValueError(f"Unknown detector type: {config['detector_type']!r}")


def load_detector(path, map_location=None):
    """
    Loads a detector and its weights from a checkpoint saved by ``save`` or
    a trainer.

    Parameters
    ----------
    path : str
        Path to .pth checkpoint file.
    map_location : str or torch.device, optional
        Device to map the weights to. Default is None.

    Returns
    -------
    MergeDetector
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model = build_detector(checkpoint["config"])
    state_dict = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    return model
