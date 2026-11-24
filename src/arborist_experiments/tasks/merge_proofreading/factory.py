"""
Created on Mon November 24 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Factory that builds graph encoders by name, so a training routine can pick
the graph encoder of a LearnedMergeDetector from a config string and
checkpoints can record it the same way. The vision encoder is the same
CNN3D in every experiment and is built directly.
"""

from arborist.models.arborist import ArboristEncoder

from arborist_experiments.baselines.tmd import TMD


class GraphEncoderFactory:
    """
    Graph encoders: List[TreeSample] -> (B, embed_dim). Arborist and the
    morphology baselines, keyed by name.
    """

    registry = {
        "arborist": ArboristEncoder,
        "tmd": TMD,
    }

    @classmethod
    def build(cls, encoder_type, *args, **kwargs):
        """
        Instantiates the encoder registered under ``encoder_type``.

        Parameters
        ----------
        encoder_type : str
            Registered name, see ``types``.
        *args, **kwargs
            Forwarded to the encoder constructor.

        Returns
        -------
        torch.nn.Module
        """
        if encoder_type not in cls.registry:
            raise ValueError(f"Unknown encoder type {encoder_type!r}; expected one of {cls.types()}")
        return cls.registry[encoder_type](*args, **kwargs)

    @classmethod
    def register(cls, encoder_type, encoder_cls):
        """
        Adds an encoder class to the factory.

        Parameters
        ----------
        encoder_type : str
            Name to build it by.
        encoder_cls : type
            Class with ``embed_dim``, ``config`` and the encoder ``forward``.
        """
        cls.registry[encoder_type] = encoder_cls

    @classmethod
    def name_of(cls, encoder):
        """
        Registered name of an encoder instance.

        Parameters
        ----------
        encoder : torch.nn.Module

        Returns
        -------
        str
        """
        for name, encoder_cls in cls.registry.items():
            if type(encoder) is encoder_cls:
                return name
        raise ValueError(f"{type(encoder).__name__} is not registered in {cls.__name__}")

    @classmethod
    def types(cls):
        """Registered encoder names."""
        return sorted(cls.registry)
