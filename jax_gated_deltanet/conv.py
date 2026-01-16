"""Short convolution module for Gated DeltaNet.

Causal 1D convolution applied to Q, K, V before the delta rule.
Critical for performance per the FLA authors.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class ShortConvolution(eqx.Module):
    """Causal short convolution with SiLU activation.

    Applied to Q, K, V projections before the delta rule recurrence.
    Supports caching for efficient autoregressive inference.

    The convolution is depthwise (each channel convolved independently)
    and causal (only looks at past and current positions).
    """

    hidden_size: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)

    # Depthwise conv weights: [hidden_size, kernel_size]
    weight: Float[Array, "hidden_size kernel_size"]
    bias: Optional[Float[Array, " hidden_size"]]

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        bias: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        # Initialize weights with small values
        key1, key2 = jax.random.split(key)
        # Fan-in initialization
        std = 1.0 / (kernel_size**0.5)
        self.weight = jax.random.normal(key1, (hidden_size, kernel_size)) * std

        if bias:
            self.bias = jnp.zeros(hidden_size)
        else:
            self.bias = None

    def __call__(
        self,
        x: Float[Array, "seq_len hidden_size"],
        cache: Optional[Float[Array, "hidden_size kernel_size_minus_1"]] = None,
        output_cache: bool = False,
    ) -> Tuple[
        Float[Array, "seq_len hidden_size"],
        Optional[Float[Array, "hidden_size kernel_size_minus_1"]],
    ]:
        """Apply causal convolution with SiLU activation.

        Args:
            x: Input tensor [seq_len, hidden_size]
            cache: Optional cache from previous call [hidden_size, kernel_size-1]
                   Used for autoregressive inference.
            output_cache: Whether to return updated cache for next call.

        Returns:
            (output, new_cache): Output tensor and optional updated cache.
        """
        seq_len, hidden_size = x.shape

        # Transpose for conv: [hidden_size, seq_len]
        x_t = x.T

        if cache is not None:
            # Prepend cache for causal continuity
            x_t = jnp.concatenate([cache, x_t], axis=1)

        # Causal padding: pad left only
        pad_len = self.kernel_size - 1
        if cache is None:
            x_padded = jnp.pad(x_t, ((0, 0), (pad_len, 0)), mode="constant")
        else:
            # Cache already provides the padding
            x_padded = x_t

        # Depthwise convolution via vmap over channels
        # For each channel: convolve x_padded[c, :] with weight[c, :]
        def conv_channel(x_ch: Float[Array, " padded_len"], w: Float[Array, " kernel"]):
            # Valid convolution (no padding, since we pre-padded)
            return jnp.convolve(x_ch, w[::-1], mode="valid")

        y_t = jax.vmap(conv_channel)(x_padded, self.weight)

        # Add bias if present
        if self.bias is not None:
            y_t = y_t + self.bias[:, None]

        # SiLU activation
        y_t = jax.nn.silu(y_t)

        # Transpose back: [seq_len, hidden_size]
        y = y_t.T

        # Compute new cache if requested
        new_cache = None
        if output_cache:
            # Cache is the last (kernel_size - 1) positions of x_t (before padding)
            if cache is not None:
                # x_t includes old cache, take from the original x part
                new_cache = x_t[:, -(self.kernel_size - 1) :]
            else:
                # Pad if sequence is shorter than kernel_size - 1
                if seq_len >= self.kernel_size - 1:
                    new_cache = x.T[:, -(self.kernel_size - 1) :]
                else:
                    # Need to pad with zeros
                    pad_needed = self.kernel_size - 1 - seq_len
                    new_cache = jnp.concatenate(
                        [jnp.zeros((hidden_size, pad_needed)), x.T], axis=1
                    )

        return y, new_cache
