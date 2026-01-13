"""
Loss functions for WavLeJEPA training.

- Invariance loss: L2 between predicted and target latent representations
- SIGReg loss: Encourages isotropic Gaussian distribution in projected space
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .sigreg import sigreg


def invariance_loss(
    predictions: Float[Array, "*batch dim"],
    targets: Float[Array, "*batch dim"],
) -> Float[Array, ""]:
    """
    L2 loss between predicted and target representations.

    This is the core JEPA objective: predict latent targets from context.

    Args:
        predictions: Predicted representations [..., dim]
        targets: Target representations [..., dim]

    Returns:
        Scalar MSE loss
    """
    return jnp.mean((predictions - targets) ** 2)


def sigreg_loss(
    projected: Float[Array, "batch dim"],
    key: PRNGKeyArray,
    num_slices: int = 256,
) -> Float[Array, ""]:
    """
    SIGReg loss on projected representations.

    Encourages the representation distribution to be an isotropic Gaussian,
    which provably minimizes downstream risk.

    Args:
        projected: Projected representations [batch, dim]
        key: PRNG key for random projections
        num_slices: Number of random slices for SIGReg

    Returns:
        Scalar SIGReg loss
    """
    # sigreg returns per-slice losses (num_slices,) - take mean
    return jnp.mean(sigreg(projected, key, num_slices))


def compute_loss(
    outputs: dict[str, Array],
    key: PRNGKeyArray,
    sigreg_weight: float = 0.02,
    num_slices: int = 256,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """
    Compute total loss from forward_train outputs.

    Total loss = invariance_loss + λ * sigreg_loss

    Args:
        outputs: Dictionary from WavLeJEPA.forward_train containing:
            - predictions: Predicted target representations
            - targets: Actual target representations
            - projected_context: Projected context for SIGReg
            - projected_predictions: Projected predictions for SIGReg
        key: PRNG key for SIGReg random projections
        sigreg_weight: Weight for SIGReg loss (λ, default 0.02)
        num_slices: Number of random slices for SIGReg

    Returns:
        total_loss: Scalar loss for backprop
        metrics: Dict of individual loss components for logging
    """
    # Invariance loss: L2 between predictions and targets
    inv_loss = invariance_loss(outputs["predictions"], outputs["targets"])

    # SIGReg on both context and prediction projections
    key1, key2 = jax.random.split(key)
    sig_loss_ctx = sigreg_loss(outputs["projected_context"], key1, num_slices)
    sig_loss_pred = sigreg_loss(outputs["projected_predictions"], key2, num_slices)
    sig_loss = (sig_loss_ctx + sig_loss_pred) / 2

    # Total loss
    total = inv_loss + sigreg_weight * sig_loss

    metrics = {
        "loss/total": total,
        "loss/invariance": inv_loss,
        "loss/sigreg": sig_loss,
        "loss/sigreg_context": sig_loss_ctx,
        "loss/sigreg_predictions": sig_loss_pred,
    }

    return total, metrics
