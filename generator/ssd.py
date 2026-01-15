"""Stable State Space Duality (SSD) implementation.

A numerically stable JAX implementation of the Mamba2 SSD algorithm,
addressing known issues with the naive implementation:

1. Numerical overflow in cumulative products (segsum)
2. NaN propagation for non-chunk-aligned sequences
3. Underflow in exponentials for long sequences

References:
- https://tridao.me/blog/2024/mamba2-part3-algorithm/
- https://github.com/state-spaces/mamba/issues/352
- https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float


# Numerical stability constants
_MIN_EXP_ARG = -30.0  # Below this, exp() returns ~0 anyway
_MAX_EXP_ARG = 20.0   # Above this, we risk overflow


def safe_exp(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Exponential with underflow/overflow protection."""
    return jnp.exp(jnp.clip(x, _MIN_EXP_ARG, _MAX_EXP_ARG))


def segsum_stable(x: Float[Array, "... T"]) -> Float[Array, "... T T"]:
    """Numerically stable segment sum computation.

    Computes a matrix M where M[..., i, j] = sum(x[..., i:j+1]) for i <= j.
    
    The naive approach computes cumsum then subtracts, which loses precision.
    This implementation uses independent cumsums to avoid subtraction.

    Args:
        x: Input tensor with last dimension T

    Returns:
        Segment sum matrix [..., T, T], lower triangular with -inf above diagonal
    """
    T = x.shape[-1]
    
    # For each position i, compute cumsum starting from i
    # Create [T, T] index matrix where row i has [0,0,...,0,1,1,...,1]
    # with 1s starting at position i
    indices = jnp.arange(T)
    mask_start = indices[None, :] >= indices[:, None]  # [T, T]
    
    # Broadcast x to [..., T, T] and mask
    x_expanded = jnp.broadcast_to(x[..., None, :], (*x.shape, T))  # [..., T, T]
    x_masked = jnp.where(mask_start, x_expanded, 0.0)
    
    # Cumsum along the last axis (within each row)
    x_cumsum = jnp.cumsum(x_masked, axis=-1)  # [..., T, T]
    
    # We want M[i,j] = sum(x[i:j+1]), which is x_cumsum[i, j] when i <= j
    # The cumsum gives us sum from position i to position j
    
    # Create lower triangular mask (including diagonal)
    mask_lower = jnp.tril(jnp.ones((T, T), dtype=bool))
    
    # Apply mask: valid values below/on diagonal, -inf above
    result = jnp.where(mask_lower, x_cumsum, -jnp.inf)
    
    return result


def ssd_stable(
    x: Float[Array, "batch seq_len num_heads head_dim"],
    dt: Float[Array, "batch seq_len num_heads"],
    A: Float[Array, " num_heads"],
    B: Float[Array, "batch seq_len num_heads state_size"],
    C: Float[Array, "batch seq_len num_heads state_size"],
    chunk_size: int,
    D: Float[Array, " num_heads"],
    dt_bias: Float[Array, " num_heads"],
    dt_min: float = 0.0,
    dt_max: float = float("inf"),
    initial_states: Optional[Float[Array, "batch 1 num_heads head_dim state_size"]] = None,
    return_final_state: bool = False,
) -> Tuple[
    Float[Array, "batch seq_len num_heads head_dim"],
    Optional[Float[Array, "batch num_heads head_dim state_size"]],
]:
    """Numerically stable SSD computation.

    This is a stable reimplementation of the Mamba2 SSD algorithm that:
    1. Properly handles non-chunk-aligned sequences
    2. Uses stable segment sum computation
    3. Clamps intermediate values to prevent NaN

    Args:
        x: Input tensor [batch, seq_len, num_heads, head_dim]
        dt: Time deltas [batch, seq_len, num_heads]
        A: State transition (should be negative) [num_heads]
        B: Input projection [batch, seq_len, num_heads, state_size]
        C: Output projection [batch, seq_len, num_heads, state_size]
        chunk_size: Size of chunks for block computation
        D: Skip connection weight [num_heads]
        dt_bias: Bias for dt (before softplus) [num_heads]
        dt_min: Minimum dt value after softplus
        dt_max: Maximum dt value after softplus
        initial_states: Optional initial state [batch, 1, num_heads, head_dim, state_size]
        return_final_state: Whether to return final state

    Returns:
        y: Output tensor [batch, seq_len, num_heads, head_dim]
        final_state: Optional final state [batch, num_heads, head_dim, state_size]
    """
    batch_size, seq_len, num_heads, head_dim = x.shape
    state_size = B.shape[-1]

    # Compute padding to make sequence length a multiple of chunk_size
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    # Apply dt bias and softplus
    dt = jax.nn.softplus(dt + dt_bias)
    dt = jnp.clip(dt, dt_min, dt_max)

    # Pad all sequence-dimension tensors
    def pad_seq(tensor):
        if pad_size == 0:
            return tensor
        pad_width = [(0, 0)] * tensor.ndim
        pad_width[1] = (0, pad_size)
        return jnp.pad(tensor, pad_width, mode="constant", constant_values=0.0)

    x_pad = pad_seq(x)
    dt_pad = pad_seq(dt)
    B_pad = pad_seq(B)
    C_pad = pad_seq(C)

    # D residual connection (before chunking)
    D_residual = D[None, None, :, None] * x_pad  # [batch, seq_len_padded, num_heads, head_dim]

    # Discretize: x_disc = x * dt, A_disc = A * dt
    x_disc = x_pad * dt_pad[..., None]  # [batch, seq_len_padded, num_heads, head_dim]
    A_disc = A[None, None, :] * dt_pad  # [batch, seq_len_padded, num_heads]

    # Reshape into chunks: [batch, num_chunks, chunk_size, ...]
    def to_chunks(tensor):
        return rearrange(tensor, "b (c l) ... -> b c l ...", l=chunk_size)

    x_blk = to_chunks(x_disc)   # [batch, num_chunks, chunk_size, num_heads, head_dim]
    A_blk = to_chunks(A_disc)   # [batch, num_chunks, chunk_size, num_heads]
    B_blk = to_chunks(B_pad)    # [batch, num_chunks, chunk_size, num_heads, state_size]
    C_blk = to_chunks(C_pad)    # [batch, num_chunks, chunk_size, num_heads, state_size]

    # Rearrange A for cumsum: [batch, num_heads, num_chunks, chunk_size]
    A_blk_t = rearrange(A_blk, "b c l h -> b h c l")

    # Cumulative sum of A within each chunk (for intra-chunk computation)
    A_cumsum = jnp.cumsum(A_blk_t, axis=-1)  # [batch, num_heads, num_chunks, chunk_size]

    # Clamp to prevent numerical issues
    A_cumsum = jnp.clip(A_cumsum, _MIN_EXP_ARG, 0.0)

    # =========================================================================
    # Step 1: Intra-chunk outputs (diagonal blocks)
    # =========================================================================
    # L_mat[i,j] = exp(sum(A[i:j+1])) for the lower triangular matrix
    L_mat = safe_exp(segsum_stable(A_cumsum))  # [batch, num_heads, num_chunks, chunk_size, chunk_size]

    # Compute Y_diag = C @ (L_mat * (B @ x))
    # einsum: "bclhn,bcshn,bhcls,bcshp->bclhp"
    Y_diag = jnp.einsum(
        "bclhn,bcshn,bhcls,bcshp->bclhp",
        C_blk,  # [batch, num_chunks, chunk_size, num_heads, state_size]
        B_blk,  # [batch, num_chunks, chunk_size, num_heads, state_size]
        L_mat,  # [batch, num_heads, num_chunks, chunk_size, chunk_size]
        x_blk,  # [batch, num_chunks, chunk_size, num_heads, head_dim]
    )  # [batch, num_chunks, chunk_size, num_heads, head_dim]

    # =========================================================================
    # Step 2: Compute chunk states (final state of each chunk assuming zero initial)
    # =========================================================================
    # decay_states[l] = exp(A_cumsum[-1] - A_cumsum[l]) = exp(sum(A[l+1:end]))
    A_chunk_end = A_cumsum[..., -1:]  # [batch, num_heads, num_chunks, 1]
    decay_states = safe_exp(A_chunk_end - A_cumsum)  # [batch, num_heads, num_chunks, chunk_size]

    # states = sum_l(B[l] * decay_states[l] * x[l])
    states = jnp.einsum(
        "bclhn,bhcl,bclhp->bchpn",
        B_blk,       # [batch, num_chunks, chunk_size, num_heads, state_size]
        decay_states,  # [batch, num_heads, num_chunks, chunk_size]
        x_blk,       # [batch, num_chunks, chunk_size, num_heads, head_dim]
    )  # [batch, num_chunks, num_heads, head_dim, state_size]

    # =========================================================================
    # Step 3: Inter-chunk recurrence (pass states between chunks)
    # =========================================================================
    # Prepend initial states
    if initial_states is None:
        initial_states = jnp.zeros((batch_size, 1, num_heads, head_dim, state_size), dtype=x.dtype)

    states_with_init = jnp.concatenate([initial_states, states], axis=1)  # [batch, num_chunks+1, ...]

    # A at chunk boundaries (total decay within each chunk)
    A_chunk_total = A_cumsum[..., -1]  # [batch, num_heads, num_chunks]

    # Pad with zero at the beginning for initial state
    A_chunk_total_padded = jnp.pad(
        A_chunk_total, ((0, 0), (0, 0), (1, 0))
    )  # [batch, num_heads, num_chunks+1]

    # Compute inter-chunk decay matrix
    decay_chunk = safe_exp(segsum_stable(A_chunk_total_padded))  # [batch, num_heads, num_chunks+1, num_chunks+1]

    # Apply inter-chunk recurrence
    new_states = jnp.einsum(
        "bhzc,bchpn->bzhpn",
        decay_chunk,  # [batch, num_heads, num_chunks+1, num_chunks+1]
        states_with_init,  # [batch, num_chunks+1, num_heads, head_dim, state_size]
    )  # [batch, num_chunks+1, num_heads, head_dim, state_size]

    # Split into chunk initial states and final state
    chunk_init_states = new_states[:, :-1, ...]  # [batch, num_chunks, num_heads, head_dim, state_size]
    final_state = new_states[:, -1, ...]  # [batch, num_heads, head_dim, state_size]

    # =========================================================================
    # Step 4: Compute output contribution from initial states
    # =========================================================================
    # state_decay_out[l] = exp(A_cumsum[l]) = exp(sum(A[0:l+1]))
    state_decay_out = safe_exp(A_cumsum)  # [batch, num_heads, num_chunks, chunk_size]

    Y_off = jnp.einsum(
        "bclhn,bchpn,bhcl->bclhp",
        C_blk,  # [batch, num_chunks, chunk_size, num_heads, state_size]
        chunk_init_states,  # [batch, num_chunks, num_heads, head_dim, state_size]
        state_decay_out,  # [batch, num_heads, num_chunks, chunk_size]
    )  # [batch, num_chunks, chunk_size, num_heads, head_dim]

    # =========================================================================
    # Combine outputs
    # =========================================================================
    y = Y_diag + Y_off  # [batch, num_chunks, chunk_size, num_heads, head_dim]

    # Reshape back to sequence
    y = rearrange(y, "b c l h p -> b (c l) h p")  # [batch, seq_len_padded, num_heads, head_dim]

    # Add D residual
    y = y + D_residual

    # Remove padding
    if pad_size > 0:
        y = y[:, :seq_len, :, :]

    if return_final_state:
        return y, final_state
    else:
        return y, None
