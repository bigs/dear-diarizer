"""Training losses for the attractor generator.

Includes:
- Confidence loss: BCE against usage-based targets
- Combined loss: energy + confidence
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from .energy import get_assignment_weights, total_energy


def confidence_loss(
    confidences: Float[Array, " num_attractors"],
    frames: Float[Array, "num_frames dim"],
    attractors: Float[Array, "num_attractors dim"],
    tau: float,
    usage_threshold: float,
    attractor_mask: Float[Array, " num_attractors"] | None = None,
) -> Float[Array, ""]:
    """Binary cross-entropy loss for confidence head.

    Trains confidence to predict whether each attractor is "useful"
    based on its soft-assignment mass.

    target_k = 1 if usage_k > threshold else 0
    L = (1/K) * Σ_k BCE(c_k, target_k)

    Args:
        confidences: [K] predicted confidence scores (sigmoid outputs)
        frames: [N, D] frame embeddings
        attractors: [K, D] attractor embeddings
        tau: Temperature for soft assignment
        usage_threshold: Minimum usage to be considered "useful"
        attractor_mask: [K] binary mask for valid attractors

    Returns:
        Scalar BCE loss
    """
    # Get soft assignment weights and compute usage
    weights = get_assignment_weights(frames, attractors, tau, attractor_mask)  # [N, K]
    usage = jnp.sum(weights, axis=0)  # [K]

    # Binary targets: 1 if usage > threshold, else 0
    targets = (usage > usage_threshold).astype(jnp.float32)  # [K]

    # Binary cross-entropy (numerically stable)
    # BCE = -[t * log(c) + (1-t) * log(1-c)]
    eps = 1e-7
    confidences = jnp.clip(confidences, eps, 1.0 - eps)
    bce = -(targets * jnp.log(confidences) + (1 - targets) * jnp.log(1 - confidences))

    # Mask invalid attractors
    if attractor_mask is not None:
        bce = bce * attractor_mask
        num_valid = jnp.sum(attractor_mask)
        return jnp.sum(bce) / jnp.maximum(num_valid, 1.0)
    else:
        return jnp.mean(bce)


def combined_loss(
    frames: Float[Array, "num_frames dim"],
    attractors: Float[Array, "num_attractors dim"],
    confidences: Float[Array, " num_attractors"],
    tau: float,
    lambda_separation: float = 1.0,
    lambda_coverage: float = 0.1,
    lambda_confidence: float = 1.0,
    separation_margin: float = 1.0,
    min_usage: float = 1.0,
    usage_threshold: float = 10.0,
    attractor_mask: Float[Array, " num_attractors"] | None = None,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """Combined training loss: energy + confidence.

    L = E(A, X) + λ_conf * L_confidence

    Args:
        frames: [N, D] frame embeddings
        attractors: [K, D] attractor embeddings
        confidences: [K] predicted confidence scores
        tau: Temperature for soft assignment
        lambda_separation: Weight for separation term
        lambda_coverage: Weight for coverage term
        lambda_confidence: Weight for confidence loss
        separation_margin: Margin for hinge loss
        min_usage: Minimum usage for coverage penalty
        usage_threshold: Threshold for confidence targets
        attractor_mask: [K] binary mask for valid attractors

    Returns:
        (total_loss, loss_dict): Total loss and breakdown for logging
    """
    # Energy loss
    e_total = total_energy(
        frames=frames,
        attractors=attractors,
        tau=tau,
        lambda_separation=lambda_separation,
        lambda_coverage=lambda_coverage,
        separation_margin=separation_margin,
        min_usage=min_usage,
        attractor_mask=attractor_mask,
    )

    # Confidence loss
    l_conf = confidence_loss(
        confidences=confidences,
        frames=frames,
        attractors=attractors,
        tau=tau,
        usage_threshold=usage_threshold,
        attractor_mask=attractor_mask,
    )

    total = e_total + lambda_confidence * l_conf

    loss_dict = {
        "total": total,
        "energy": e_total,
        "confidence": l_conf,
    }

    return total, loss_dict
