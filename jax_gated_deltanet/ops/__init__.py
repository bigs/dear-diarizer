"""Optimized operations for Gated DeltaNet.

Includes chunkwise parallel computation and future Pallas kernels.
"""

from .chunk import gated_delta_rule_chunk, gated_delta_rule_chunk_simple

__all__ = [
    "gated_delta_rule_chunk",
    "gated_delta_rule_chunk_simple",
]
