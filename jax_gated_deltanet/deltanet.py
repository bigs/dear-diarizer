"""Core Gated DeltaNet recurrence.

This module contains the delta rule computation - the kernel-swappable piece.
Currently implements a naive sequential version using jax.lax.scan.
Future: dispatch to Pallas kernels in ops/chunk.py for hardware efficiency.

The delta rule update:
    h = h * exp(g)                              # decay (gating)
    error = beta * (v - h @ k)                  # delta rule error
    h = h + outer(k, error)                     # update state
    o = h @ q                                   # query output
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def l2_normalize(
    x: Float[Array, "... dim"], eps: float = 1e-6
) -> Float[Array, "... dim"]:
    """L2 normalize along the last dimension."""
    norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + eps)
    return x / norm


def gated_delta_rule(
    q: Float[Array, "batch seq num_heads head_k"],
    k: Float[Array, "batch seq num_heads head_k"],
    v: Float[Array, "batch seq num_heads head_v"],
    g: Float[Array, "batch seq num_heads"],
    beta: Float[Array, "batch seq num_heads"],
    scale: Optional[float] = None,
    initial_state: Optional[Float[Array, "batch num_heads head_k head_v"]] = None,
    output_final_state: bool = False,
) -> Tuple[
    Float[Array, "batch seq num_heads head_v"],
    Optional[Float[Array, "batch num_heads head_k head_v"]],
]:
    """Gated delta rule recurrence.

    This is the naive O(seq_len) sequential implementation using jax.lax.scan.
    The interface is designed for future Pallas kernel drop-in.

    Args:
        q: Queries [batch, seq, num_heads, head_k] - should be L2 normalized
        k: Keys [batch, seq, num_heads, head_k] - should be L2 normalized
        v: Values [batch, seq, num_heads, head_v]
        g: Decay gates in LOG SPACE [batch, seq, num_heads]
           (will be exponentiated, should be negative for decay)
        beta: Learning rates / update gates [batch, seq, num_heads]
              (controls how much to update state, typically sigmoid output)
        scale: Scale factor for queries. Default: 1/sqrt(head_k)
        initial_state: Initial state [batch, num_heads, head_k, head_v]
        output_final_state: Whether to return final state.

    Returns:
        output: [batch, seq, num_heads, head_v]
        final_state: [batch, num_heads, head_k, head_v] if output_final_state else None
    """
    batch, seq_len, num_heads, head_k = q.shape
    head_v = v.shape[-1]

    if scale is None:
        scale = head_k**-0.5

    # Scale queries
    q = q * scale

    # Initialize state
    if initial_state is None:
        h0 = jnp.zeros((batch, num_heads, head_k, head_v), dtype=q.dtype)
    else:
        h0 = initial_state

    def scan_fn(
        h: Float[Array, "batch num_heads head_k head_v"],
        inputs: Tuple[
            Float[Array, "batch num_heads head_k"],  # q_t
            Float[Array, "batch num_heads head_k"],  # k_t
            Float[Array, "batch num_heads head_v"],  # v_t
            Float[Array, "batch num_heads"],  # g_t
            Float[Array, "batch num_heads"],  # beta_t
        ],
    ) -> Tuple[
        Float[Array, "batch num_heads head_k head_v"],
        Float[Array, "batch num_heads head_v"],
    ]:
        q_t, k_t, v_t, g_t, beta_t = inputs

        # Decay state: h = h * exp(g)
        # g is in log space, so exp(g) is the decay factor
        # Expand g for broadcasting: [batch, num_heads, 1, 1]
        decay = jnp.exp(g_t[:, :, None, None])
        h = h * decay

        # Delta rule update:
        # error = v - h @ k (what we predicted vs what we got)
        # h @ k: [batch, num_heads, head_k, head_v] @ [batch, num_heads, head_k]
        #      = [batch, num_heads, head_v] (contract over head_k)
        h_k = jnp.einsum("bnkv,bnk->bnv", h, k_t)
        error = v_t - h_k  # [batch, num_heads, head_v]

        # Scale error by beta (learning rate)
        # beta: [batch, num_heads] -> [batch, num_heads, 1]
        error = beta_t[:, :, None] * error

        # Update state: h = h + k ⊗ error (outer product)
        # k: [batch, num_heads, head_k]
        # error: [batch, num_heads, head_v]
        # k ⊗ error: [batch, num_heads, head_k, head_v]
        h = h + jnp.einsum("bnk,bnv->bnkv", k_t, error)

        # Output: o = h @ q
        # h: [batch, num_heads, head_k, head_v]
        # q: [batch, num_heads, head_k]
        # o: [batch, num_heads, head_v]
        o_t = jnp.einsum("bnkv,bnk->bnv", h, q_t)

        return h, o_t

    # Transpose for scan: [seq, batch, num_heads, ...]
    q_scan = jnp.transpose(q, (1, 0, 2, 3))
    k_scan = jnp.transpose(k, (1, 0, 2, 3))
    v_scan = jnp.transpose(v, (1, 0, 2, 3))
    g_scan = jnp.transpose(g, (1, 0, 2))
    beta_scan = jnp.transpose(beta, (1, 0, 2))

    # Run scan
    final_state, outputs = jax.lax.scan(
        scan_fn, h0, (q_scan, k_scan, v_scan, g_scan, beta_scan)
    )

    # Transpose outputs back: [batch, seq, num_heads, head_v]
    outputs = jnp.transpose(outputs, (1, 0, 2, 3))

    if output_final_state:
        return outputs, final_state
    else:
        return outputs, None


def gated_delta_rule_recurrent(
    q: Float[Array, "batch 1 num_heads head_k"],
    k: Float[Array, "batch 1 num_heads head_k"],
    v: Float[Array, "batch 1 num_heads head_v"],
    g: Float[Array, "batch 1 num_heads"],
    beta: Float[Array, "batch 1 num_heads"],
    state: Float[Array, "batch num_heads head_k head_v"],
    scale: Optional[float] = None,
) -> Tuple[
    Float[Array, "batch 1 num_heads head_v"],
    Float[Array, "batch num_heads head_k head_v"],
]:
    """Single-step recurrent update for inference.

    More efficient than gated_delta_rule for autoregressive generation
    where we process one token at a time.

    Args:
        q, k, v, g, beta: Single timestep inputs [batch, 1, ...]
        state: Current state [batch, num_heads, head_k, head_v]
        scale: Scale factor for queries.

    Returns:
        output: [batch, 1, num_heads, head_v]
        new_state: [batch, num_heads, head_k, head_v]
    """
    # Squeeze the seq dimension
    q_t = q[:, 0]  # [batch, num_heads, head_k]
    k_t = k[:, 0]
    v_t = v[:, 0]
    g_t = g[:, 0]
    beta_t = beta[:, 0]

    head_k = q_t.shape[-1]
    if scale is None:
        scale = head_k**-0.5

    q_t = q_t * scale

    # Decay
    decay = jnp.exp(g_t[:, :, None, None])
    h = state * decay

    # Delta update
    h_k = jnp.einsum("bnkv,bnk->bnv", h, k_t)
    error = beta_t[:, :, None] * (v_t - h_k)
    h = h + jnp.einsum("bnk,bnv->bnkv", k_t, error)

    # Output
    o_t = jnp.einsum("bnkv,bnk->bnv", h, q_t)

    # Add seq dimension back
    output = o_t[:, None, :, :]

    return output, h
