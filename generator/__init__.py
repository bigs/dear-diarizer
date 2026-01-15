"""Speaker Attractor Generator module.

Second stage of the speaker diarization pipeline:
WavLeJEPA embeddings -> Generator -> Speaker attractors
"""

from .config import GeneratorConfig
from .mamba import (
    RMSNorm,
    Mamba2Layer,
    Mamba2Block,
    LinearAttentionStack,
)
from .attractor import AttractorGenerator
from .energy import (
    compute_distances,
    soft_assignment,
    energy_assignment,
    energy_separation,
    energy_coverage,
    total_energy,
    get_assignment_weights,
)
from .losses import confidence_loss, combined_loss
from .synthetic import (
    make_synthetic_data,
    make_variable_length_data,
    make_overlapping_speakers,
)

__all__ = [
    # Config
    "GeneratorConfig",
    # Mamba components
    "RMSNorm",
    "Mamba2Layer",
    "Mamba2Block",
    "LinearAttentionStack",
    # Main generator
    "AttractorGenerator",
    # Energy functions
    "compute_distances",
    "soft_assignment",
    "energy_assignment",
    "energy_separation",
    "energy_coverage",
    "total_energy",
    "get_assignment_weights",
    # Losses
    "confidence_loss",
    "combined_loss",
    # Synthetic data
    "make_synthetic_data",
    "make_variable_length_data",
    "make_overlapping_speakers",
]
