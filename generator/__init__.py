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
]
