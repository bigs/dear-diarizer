"""
Mixed precision utilities for WavLeJEPA training.

Supports bfloat16 mixed precision for H100/GB10 (Blackwell) GPUs,
giving ~2x throughput while maintaining training stability.

Mixed precision strategy:
- Master weights are stored in float32 (for optimizer state)
- Forward pass uses compute_dtype (bfloat16) for speed
- Loss is computed in float32 for numerical stability
- Gradients flow back in compute_dtype, then cast to float32 for updates
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array

from .config import PrecisionConfig


# Dtype mapping from string config to JAX dtype
DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
}


def get_compute_dtype(config: PrecisionConfig):
    """Get the JAX dtype for compute operations."""
    return DTYPE_MAP[config.compute_dtype]


def cast_to_compute(x: Array, config: PrecisionConfig) -> Array:
    """Cast array to compute dtype for forward pass."""
    dtype = get_compute_dtype(config)
    return x.astype(dtype)


def cast_to_float32(x: Array) -> Array:
    """Cast array back to float32 for loss computation."""
    return x.astype(jnp.float32)


def cast_model_to_compute(model, compute_dtype):
    """
    Cast all float32 arrays in model to compute_dtype for forward pass.

    This enables mixed precision by running the forward pass in bfloat16
    while keeping master weights in float32.

    Args:
        model: Equinox model with float32 weights
        compute_dtype: Target dtype (e.g., jnp.bfloat16)

    Returns:
        Model with arrays cast to compute_dtype
    """

    def cast_if_float(x):
        if eqx.is_array(x) and x.dtype == jnp.float32:
            return x.astype(compute_dtype)
        return x

    return jax.tree.map(cast_if_float, model, is_leaf=eqx.is_array)
