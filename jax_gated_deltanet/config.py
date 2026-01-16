"""Configuration for Gated DeltaNet.

Based on Qwen3-Next architecture defaults.
Reference: https://github.com/fla-org/flash-linear-attention
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GatedDeltaNetConfig:
    """Configuration for Gated DeltaNet layers.

    Architectural parameters (commonly adjusted):
        hidden_size: Input/output dimension of the layer.
        num_heads: Number of attention heads for queries/keys.
        num_v_heads: Number of heads for values. If None, equals num_heads.
            Set > num_heads for Grouped Value Attention (GVA).
        head_k_dim: Dimension of each key/query head.
        expand_v: Expansion factor for value head dimension.
            head_v_dim = int(head_k_dim * expand_v)
        num_layers: Number of layers in a stack.

    Ablation parameters (rarely changed):
        use_gate: Whether to use output gating. Default True.
        use_short_conv: Whether to use short convolutions. Default True.
        conv_size: Kernel size for short convolutions. Default 4.
        norm_eps: Epsilon for layer normalization. Default 1e-5.

    Derived (computed automatically):
        head_v_dim: Dimension of each value head.
        key_dim: Total key dimension (num_heads * head_k_dim).
        value_dim: Total value dimension (num_v_heads * head_v_dim).
    """

    # Architectural parameters
    hidden_size: int = 2048
    num_heads: int = 6
    num_v_heads_: Optional[int] = None  # defaults to num_heads; use num_v_heads property
    head_k_dim: int = 256
    expand_v: float = 2.0
    num_layers: int = 6

    # Ablation parameters
    use_gate: bool = True
    use_short_conv: bool = True
    conv_size: int = 4
    norm_eps: float = 1e-5

    # Initialization parameters (fixed, from paper)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    a_init_range: tuple[float, float] = (0.0, 16.0)

    def __post_init__(self):
        # Validate GVA configuration
        if self.num_v_heads > self.num_heads:
            if self.num_v_heads % self.num_heads != 0:
                raise ValueError(
                    f"num_v_heads ({self.num_v_heads}) must be divisible by "
                    f"num_heads ({self.num_heads}) for Grouped Value Attention."
                )

        # Validate expand_v produces integer head_v_dim
        head_v_dim = self.head_k_dim * self.expand_v
        if not head_v_dim.is_integer():
            raise ValueError(
                f"expand_v={self.expand_v} with head_k_dim={self.head_k_dim} "
                f"produces non-integer head_v_dim={head_v_dim}."
            )

    @property
    def num_v_heads(self) -> int:
        """Number of value heads. Defaults to num_heads if not specified."""
        return self.num_v_heads_ if self.num_v_heads_ is not None else self.num_heads

    @property
    def head_v_dim(self) -> int:
        """Dimension of each value head."""
        return int(self.head_k_dim * self.expand_v)

    @property
    def key_dim(self) -> int:
        """Total key/query dimension."""
        return self.num_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        """Total value dimension."""
        return self.num_v_heads * self.head_v_dim

    @property
    def gva_groups(self) -> int:
        """Number of GVA groups (how many times to repeat q/k)."""
        return self.num_v_heads // self.num_heads
