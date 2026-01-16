"""Configuration for the Speaker Attractor Generator."""

from dataclasses import dataclass
from typing import Optional, Tuple

from jax_gated_deltanet import GatedDeltaNetConfig


@dataclass
class Mamba2Config:
    """Configuration for Mamba2 SSM layers.

    Used as nested config in GeneratorConfig.mamba2
    """

    # Architecture
    state_size: int = 128  # SSM state size (N)
    head_dim: int = 64  # Dimension per head
    expand: int = 2  # Expansion factor for intermediate size
    conv_kernel: int = 4  # Depthwise conv kernel size
    chunk_size: int = 256  # Chunk size for SSD computation

    # Initialization
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: Tuple[float, float] = (0.0, float("inf"))
    A_initializer_range: Tuple[float, float] = (1.0, 16.0)

    def intermediate_size(self, hidden_dim: int) -> int:
        """Compute intermediate size given parent hidden_dim."""
        return int(self.expand * hidden_dim)

    def num_heads(self, hidden_dim: int) -> int:
        """Compute number of heads given parent hidden_dim."""
        return self.intermediate_size(hidden_dim) // self.head_dim


@dataclass
class GeneratorConfig:
    """Configuration for the AttractorGenerator.

    See docs/generator_design.md for full specification.

    SSM Backend:
        Exactly one of `mamba2` or `deltanet` must be provided.
        Example YAML:
            hidden_dim: 768
            num_layers: 4
            mamba2:
              state_size: 128
              chunk_size: 256
    """

    # Dimensions
    input_dim: int = 768  # WavLeJEPA output dim
    hidden_dim: int = 768  # Internal hidden size
    attractor_dim: int = 768  # Output attractor dimension

    # Linear attention stack
    num_layers: int = 4  # Number of SSM layers

    # SSM backend (exactly one must be provided)
    mamba2: Optional[Mamba2Config] = None
    deltanet: Optional[GatedDeltaNetConfig] = None

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

    def __post_init__(self):
        """Validate that exactly one SSM backend is configured."""
        if self.mamba2 is not None and self.deltanet is not None:
            raise ValueError(
                "Cannot specify both 'mamba2' and 'deltanet' config. "
                "Choose one SSM backend."
            )
        if self.mamba2 is None and self.deltanet is None:
            raise ValueError(
                "Must specify either 'mamba2' or 'deltanet' config. "
                "Example: GeneratorConfig(mamba2=Mamba2Config())"
            )
        # Validate deltanet hidden_size matches generator hidden_dim
        if self.deltanet is not None and self.deltanet.hidden_size != self.hidden_dim:
            raise ValueError(
                f"deltanet.hidden_size ({self.deltanet.hidden_size}) must match "
                f"hidden_dim ({self.hidden_dim})"
            )

    @property
    def ssm_type(self) -> str:
        """Return which SSM backend is configured."""
        return "mamba2" if self.mamba2 is not None else "deltanet"

    # Convenience properties for Mamba2 (delegate to nested config)
    @property
    def intermediate_size(self) -> int:
        """Intermediate size for Mamba2 mixer."""
        if self.mamba2 is None:
            raise AttributeError("intermediate_size only available with mamba2 config")
        return self.mamba2.intermediate_size(self.hidden_dim)

    @property
    def num_heads(self) -> int:
        """Number of heads in Mamba2."""
        if self.mamba2 is None:
            raise AttributeError("num_heads only available with mamba2 config")
        return self.mamba2.num_heads(self.hidden_dim)
