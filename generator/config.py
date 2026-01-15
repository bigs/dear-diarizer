"""Configuration for the Speaker Attractor Generator."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class GeneratorConfig:
    """Configuration for the AttractorGenerator.

    See docs/generator_design.md for full specification.
    """

    # Dimensions
    input_dim: int = 768  # WavLeJEPA output dim
    hidden_dim: int = 768  # GRU hidden size
    attractor_dim: int = 768  # Output attractor dimension

    # Mamba2 / Linear attention stack
    num_layers: int = 4  # Number of Mamba2 layers
    state_size: int = 128  # SSM state size (N)
    head_dim: int = 64  # Dimension per head
    expand: int = 2  # Expansion factor for intermediate size
    conv_kernel: int = 4  # Depthwise conv kernel size
    chunk_size: int = 256  # Chunk size for SSD computation

    # Cross-attention
    num_cross_attn_heads: int = 8  # Multi-head attention heads

    # Generation
    max_attractors: int = 10  # Maximum speakers to generate
    confidence_threshold: float = 0.5

    # Energy weights
    lambda_separation: float = 1.0  # Weight for separation term (hinge loss)
    lambda_coverage: float = 0.1  # Weight for coverage term
    separation_margin: float = 1.0  # Margin for hinge loss

    # Temperature annealing (deterministic annealing)
    tau_start: float = 1.0  # Initial temperature (soft assignment)
    tau_end: float = 0.1  # Final temperature (hard assignment)

    # Confidence training
    usage_threshold: float = 0.5  # Seconds of audio an attractor must explain

    # Mamba2 specific
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: Tuple[float, float] = (0.0, float("inf"))
    A_initializer_range: Tuple[float, float] = (1.0, 16.0)

    @property
    def intermediate_size(self) -> int:
        """Intermediate size for Mamba2 mixer."""
        return int(self.expand * self.hidden_dim)

    @property
    def num_heads(self) -> int:
        """Number of heads in Mamba2 (derived from intermediate_size / head_dim)."""
        return self.intermediate_size // self.head_dim
