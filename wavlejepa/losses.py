"""
Loss functions for WavLeJEPA training.

- Invariance loss: L2 between predicted and target latent representations
- SIGReg loss: Encourages isotropic Gaussian distribution in projected space
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .sigreg import sigreg


def masked_invariance_loss_single(
    predictions: Float[Array, "seq dim"],
    targets: Float[Array, "seq dim"],
    num_valid: Int[Array, ""],
) -> Float[Array, ""]:
    """
    L2 loss for a single sample, masked to valid positions.

    Args:
        predictions: Predicted representations [seq, dim] (padded)
        targets: Target representations [seq, dim] (padded)
        num_valid: Number of valid (non-padding) positions

    Returns:
        Scalar MSE loss over valid positions only
    """
    seq_len = predictions.shape[0]

    # Create validity mask [seq_len]
    valid_mask = jnp.arange(seq_len) < num_valid

    # Compute per-position MSE
    mse_per_pos = jnp.mean((predictions - targets) ** 2, axis=-1)  # [seq_len]

    # Mask and average over valid positions only
    masked_mse = jnp.where(valid_mask, mse_per_pos, 0.0)
    # Use num_valid for averaging (avoid division by zero)
    return jnp.sum(masked_mse) / jnp.maximum(num_valid, 1)


def masked_invariance_loss(
    predictions: Float[Array, "batch seq dim"],
    targets: Float[Array, "batch seq dim"],
    num_valid: Int[Array, "batch"],
) -> Float[Array, ""]:
    """
    L2 loss between predicted and target representations, masked to valid positions.

    This is the core JEPA objective: predict latent targets from context.

    Args:
        predictions: Predicted representations [batch, seq, dim] (padded)
        targets: Target representations [batch, seq, dim] (padded)
        num_valid: Number of valid (non-padding) positions [batch]

    Returns:
        Scalar MSE loss averaged over batch and valid positions
    """
    # vmap over batch dimension
    per_sample_loss = jax.vmap(masked_invariance_loss_single)(
        predictions, targets, num_valid
    )
    return jnp.mean(per_sample_loss)


def masked_sigreg_loss(
    embeddings: Float[Array, "batch seq dim"],
    num_valid: Int[Array, "batch"],
    key: PRNGKeyArray,
    num_slices: int = 256,
) -> Float[Array, ""]:
    """
    SIGReg loss on embeddings.

    Flattens batch and seq dimensions, uses all positions for distribution estimation.
    The padded positions are duplicates which slightly biases the distribution,
    but this is acceptable for regularization purposes.

    Args:
        embeddings: Embeddings [batch, seq, dim] (padded)
        num_valid: Number of valid positions per sample [batch]
        key: PRNG key for random projections
        num_slices: Number of random slices for SIGReg

    Returns:
        Scalar SIGReg loss
    """
    batch_size, seq_len, dim = embeddings.shape

    # Flatten to [batch * seq, dim] for SIGReg
    # This treats all positions (including padding) as samples
    projected_flat = embeddings.reshape(-1, dim)

    return jnp.mean(sigreg(projected_flat, key, num_slices))


def compute_loss(
    outputs: dict[str, Array],
    key: PRNGKeyArray,
    sigreg_weight: float = 0.02,
    num_slices: int = 256,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """
    Compute total loss from forward_train outputs.

    Total loss = invariance_loss + λ * sigreg_loss

    Handles padded arrays from JIT-compatible forward pass by using
    num_targets and num_context counts for proper averaging.

    Args:
        outputs: Dictionary from WavLeJEPA.forward_train containing:
            - predictions: Predicted target representations [seq, dim]
            - targets: Actual target representations [seq, dim]
            - context_embeddings: Context representations [seq, dim]
            - num_context: Number of valid context positions
            - num_targets: Number of valid target positions
        key: PRNG key for SIGReg random projections
        sigreg_weight: Weight for SIGReg loss on encoder embeddings (λ, default 0.02)
        num_slices: Number of random slices for SIGReg

    Returns:
        total_loss: Scalar loss for backprop
        metrics: Dict of individual loss components for logging
    """
    num_targets = outputs["num_targets"]
    num_context = outputs["num_context"]

    # Invariance loss: L2 between predictions and targets (masked)
    inv_loss = masked_invariance_loss(
        outputs["predictions"],
        outputs["targets"],
        num_targets,
    )

    # SIGReg on encoder embeddings (context + target)
    key1, key2 = jax.random.split(key, 2)
    sig_loss_ctx = masked_sigreg_loss(
        outputs["context_embeddings"], num_context, key1, num_slices
    )
    sig_loss_tgt = masked_sigreg_loss(
        outputs["targets"], num_targets, key2, num_slices
    )
    sig_loss = (sig_loss_ctx + sig_loss_tgt) / 2

    # Total loss
    total = inv_loss + sigreg_weight * sig_loss

    metrics = {
        "loss/total": total,
        "loss/invariance": inv_loss,
        "loss/sigreg": sig_loss,
        "loss/sigreg_context": sig_loss_ctx,
        "loss/sigreg_targets": sig_loss_tgt,
    }

    return total, metrics
