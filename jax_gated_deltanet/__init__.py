"""JAX implementation of Gated DeltaNet.

A pure JAX/Equinox implementation of Gated Delta Networks (ICLR 2025),
structured for future Pallas kernel optimization.

Reference:
    - Paper: https://arxiv.org/abs/2412.06464
    - Official PyTorch: https://github.com/NVlabs/GatedDeltaNet
    - FLA: https://github.com/fla-org/flash-linear-attention
"""

from .config import GatedDeltaNetConfig
from .deltanet import gated_delta_rule, gated_delta_rule_recurrent, l2_normalize
from .layers import GatedDeltaNetBlock, GatedDeltaNetLayer, GatedDeltaNetStack
from .norm import FusedRMSNormGated, RMSNorm
from .conv import ShortConvolution
from .ops import gated_delta_rule_chunk

__all__ = [
    # Config
    "GatedDeltaNetConfig",
    # Core recurrence (Pallas swap-in point)
    "gated_delta_rule",
    "gated_delta_rule_recurrent",
    "gated_delta_rule_chunk",
    "l2_normalize",
    # Layers
    "GatedDeltaNetLayer",
    "GatedDeltaNetBlock",
    "GatedDeltaNetStack",
    # Components
    "RMSNorm",
    "FusedRMSNormGated",
    "ShortConvolution",
]
