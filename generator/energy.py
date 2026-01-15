"""Energy functions for attractor-based speaker diarization.

The energy function measures how well attractors explain the frame embeddings.
Low energy means:
- Each frame is close to at least one attractor (assignment)
- Attractors are distinct from each other (separation)
- All attractors explain some frames (coverage)

All functions are JAX-differentiable for use in training and test-time optimization.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def compute_distances(
    frames: Float[Array, "num_frames dim"],
    attractors: Float[Array, "num_attractors dim"],
) -> Float[Array, "num_frames num_attractors"]:
    """Compute squared L2 distances between frames and attractors.

    Args:
        frames: [N, D] frame embeddings
        attractors: [K, D] attractor embeddings

    Returns:
        distances: [N, K] squared L2 distance matrix
    """
    # ||x - a||^2 = ||x||^2 + ||a||^2 - 2 * x @ a.T
    frames_sq = jnp.sum(frames**2, axis=-1, keepdims=True)  # [N, 1]
    attractors_sq = jnp.sum(attractors**2, axis=-1, keepdims=True)  # [K, 1]
    cross = frames @ attractors.T  # [N, K]
    distances = frames_sq + attractors_sq.T - 2 * cross  # [N, K]
    return jnp.maximum(distances, 0.0)  # Numerical stability


def soft_assignment(
    distances: Float[Array, "num_frames num_attractors"],
    tau: float | Float[Array, ""],
    attractor_mask: Float[Array, " num_attractors"] | None = None,
) -> Float[Array, "num_frames num_attractors"]:
    """Compute soft assignment weights using softmin.

    Args:
        distances: [N, K] squared L2 distances
        tau: Temperature (lower = harder assignments)
        attractor_mask: [K] binary mask for valid attractors (1=valid, 0=invalid)

    Returns:
        weights: [N, K] soft assignment weights (sum to 1 over K)
    """
    # Softmin: softmax(-d / tau)
    logits = -distances / tau

    # Mask invalid attractors with -inf
    if attractor_mask is not None:
        mask = jnp.where(attractor_mask > 0.5, 0.0, -jnp.inf)
        logits = logits + mask[None, :]

    return jax.nn.softmax(logits, axis=-1)


def energy_assignment(
    frames: Float[Array, "num_frames dim"],
    attractors: Float[Array, "num_attractors dim"],
    tau: float | Float[Array, ""],
    attractor_mask: Float[Array, " num_attractors"] | None = None,
) -> Float[Array, ""]:
    """Frame-to-attractor assignment energy.

    Each frame should be well-explained by its closest attractor.
    Uses soft assignment (softmin) for smooth gradients.

    E_assignment = (1/N) * sum_i sum_k w_ik * d_ik

    Args:
        frames: [N, D] frame embeddings
        attractors: [K, D] attractor embeddings
        tau: Temperature for soft assignment
        attractor_mask: [K] binary mask for valid attractors

    Returns:
        Scalar assignment energy
    """
    distances = compute_distances(frames, attractors)  # [N, K]
    weights = soft_assignment(distances, tau, attractor_mask)  # [N, K]

    # Weighted sum of distances
    weighted_distances = weights * distances  # [N, K]
    return jnp.mean(jnp.sum(weighted_distances, axis=-1))


def energy_separation(
    attractors: Float[Array, "num_attractors dim"],
    margin: float,
    attractor_mask: Float[Array, " num_attractors"] | None = None,
) -> Float[Array, ""]:
    """Attractor separation energy (hinge loss).

    Attractors should be at least `margin` apart from each other.
    Once they're far enough, no more gradient pushes them apart.

    E_separation = sum_{k != j} max(0, margin - ||a_k - a_j||)

    Args:
        attractors: [K, D] attractor embeddings
        margin: Minimum distance between attractors
        attractor_mask: [K] binary mask for valid attractors

    Returns:
        Scalar separation energy
    """
    K = attractors.shape[0]

    # Pairwise L2 distances (not squared)
    # ||a_k - a_j|| = sqrt(||a_k||^2 + ||a_j||^2 - 2 * a_k @ a_j)
    attractors_sq = jnp.sum(attractors**2, axis=-1, keepdims=True)  # [K, 1]
    cross = attractors @ attractors.T  # [K, K]
    sq_distances = attractors_sq + attractors_sq.T - 2 * cross  # [K, K]
    distances = jnp.sqrt(jnp.maximum(sq_distances, 1e-12))  # [K, K]

    # Hinge loss: max(0, margin - distance)
    hinge = jnp.maximum(0.0, margin - distances)  # [K, K]

    # Zero out diagonal (self-distance)
    hinge = hinge * (1.0 - jnp.eye(K))

    # Mask invalid attractor pairs
    if attractor_mask is not None:
        pair_mask = attractor_mask[:, None] * attractor_mask[None, :]  # [K, K]
        hinge = hinge * pair_mask

    # Sum over all pairs (each pair counted twice, but consistent)
    return jnp.sum(hinge)


def energy_coverage(
    frames: Float[Array, "num_frames dim"],
    attractors: Float[Array, "num_attractors dim"],
    tau: float | Float[Array, ""],
    min_usage: float,
    attractor_mask: Float[Array, " num_attractors"] | None = None,
) -> Float[Array, ""]:
    """Coverage energy: penalize attractors that don't explain enough frames.

    Encourages parsimony - don't keep attractors that aren't useful.

    E_coverage = sum_k max(0, min_usage - usage_k)

    where usage_k = sum_i w_ik (soft assignment mass)

    Args:
        frames: [N, D] frame embeddings
        attractors: [K, D] attractor embeddings
        tau: Temperature for soft assignment
        min_usage: Minimum soft-assignment mass for each attractor
        attractor_mask: [K] binary mask for valid attractors

    Returns:
        Scalar coverage energy
    """
    distances = compute_distances(frames, attractors)  # [N, K]
    weights = soft_assignment(distances, tau, attractor_mask)  # [N, K]

    # Usage: sum of soft assignments per attractor
    usage = jnp.sum(weights, axis=0)  # [K]

    # Hinge: penalize if usage < min_usage
    deficit = jnp.maximum(0.0, min_usage - usage)  # [K]

    # Only count valid attractors
    if attractor_mask is not None:
        deficit = deficit * attractor_mask

    return jnp.sum(deficit)


def total_energy(
    frames: Float[Array, "num_frames dim"],
    attractors: Float[Array, "num_attractors dim"],
    tau: float | Float[Array, ""],
    lambda_separation: float = 1.0,
    lambda_coverage: float = 0.1,
    separation_margin: float = 1.0,
    min_usage: float = 1.0,
    attractor_mask: Float[Array, " num_attractors"] | None = None,
) -> Float[Array, ""]:
    """Combined energy function.

    E(A, X) = E_assignment + λ_sep * E_separation + λ_cov * E_coverage

    Args:
        frames: [N, D] frame embeddings
        attractors: [K, D] attractor embeddings
        tau: Temperature for soft assignment
        lambda_separation: Weight for separation term
        lambda_coverage: Weight for coverage term
        separation_margin: Margin for hinge loss in separation
        min_usage: Minimum usage for coverage penalty
        attractor_mask: [K] binary mask for valid attractors

    Returns:
        Scalar total energy
    """
    e_assign = energy_assignment(frames, attractors, tau, attractor_mask)
    e_sep = energy_separation(attractors, separation_margin, attractor_mask)
    e_cov = energy_coverage(frames, attractors, tau, min_usage, attractor_mask)

    return e_assign + lambda_separation * e_sep + lambda_coverage * e_cov


def get_assignment_weights(
    frames: Float[Array, "num_frames dim"],
    attractors: Float[Array, "num_attractors dim"],
    tau: float | Float[Array, ""],
    attractor_mask: Float[Array, " num_attractors"] | None = None,
) -> Float[Array, "num_frames num_attractors"]:
    """Get soft assignment weights (useful for confidence training).

    Args:
        frames: [N, D] frame embeddings
        attractors: [K, D] attractor embeddings
        tau: Temperature for soft assignment
        attractor_mask: [K] binary mask for valid attractors

    Returns:
        weights: [N, K] soft assignment weights
    """
    distances = compute_distances(frames, attractors)
    return soft_assignment(distances, tau, attractor_mask)
