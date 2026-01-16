"""Normalization modules for Gated DeltaNet.

Provides RMSNorm and FusedRMSNormGated (gated normalization for output).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class RMSNorm(eqx.Module):
    """Root Mean Square Layer Normalization.

    Normalizes inputs to have unit RMS, then scales by learned weights.
    More efficient than LayerNorm (no mean subtraction).
    """

    weight: Float[Array, " dim"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5, *, key: PRNGKeyArray):
        del key  # unused, kept for consistent API
        self.weight = jnp.ones(dim)
        self.eps = eps

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        # Compute in float32 for numerical stability
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(x_f32**2, axis=-1, keepdims=True)
        x_normed = x_f32 * jax.lax.rsqrt(variance + self.eps)
        return (x_normed * self.weight).astype(x.dtype)


class FusedRMSNormGated(eqx.Module):
    """RMSNorm with output gating.

    Applies RMSNorm then multiplies by a gating signal.
    Used for output gating in Gated DeltaNet:
        output = RMSNorm(x) * sigmoid(gate)

    This is the Qwen3-Next style output gating that helps with
    training stability by eliminating attention sink issues.
    """

    weight: Float[Array, " dim"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5, *, key: PRNGKeyArray):
        del key  # unused, kept for consistent API
        self.weight = jnp.ones(dim)
        self.eps = eps

    def __call__(
        self,
        x: Float[Array, " dim"],
        gate: Float[Array, " dim"],
    ) -> Float[Array, " dim"]:
        """Apply gated normalization.

        Args:
            x: Input to normalize [dim]
            gate: Gating signal (pre-sigmoid) [dim]

        Returns:
            Normalized and gated output [dim]
        """
        # Compute in float32 for numerical stability
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(x_f32**2, axis=-1, keepdims=True)
        x_normed = x_f32 * jax.lax.rsqrt(variance + self.eps)

        # Apply gating with sigmoid
        gate_f32 = jax.nn.sigmoid(gate.astype(jnp.float32))

        return (x_normed * self.weight * gate_f32).astype(x.dtype)
