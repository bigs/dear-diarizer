"""Regularization terms for attractor training.

Additional loss terms beyond the core energy function:
- Cardinality: constrain number of high-confidence attractors
- Temporal spread: attractors should persist across time
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def cardinality_loss(
    confidences: Float[Array, " num_attractors"],
    target_count: int | Float[Array, ""] | None = None,
    max_count: int | None = None,
    over_penalty: float = 2.0,
    attractor_mask: Float[Array, " num_attractors"] | None = None,
) -> Float[Array, ""]:
    """Penalize wrong number of high-confidence attractors.

    Two modes:
    1. Supervised (target_count provided): MSE toward exact count
    2. Unsupervised (max_count provided): Soft cap, penalize exceeding max

    Args:
        confidences: [K] predicted confidence scores (after sigmoid)
        target_count: Exact number of speakers (if known from e.g. Gemini)
        max_count: Maximum allowed speakers (soft cap if target unknown)
        over_penalty: Asymmetry factor — penalize over-counting this much more
                      (1.0 = symmetric, 2.0 = 2x penalty for over-counting)
        attractor_mask: [K] binary mask for valid attractors

    Returns:
        Scalar cardinality loss
    """
    if target_count is None and max_count is None:
        raise ValueError("Must provide either target_count or max_count")

    # Count high-confidence attractors (soft count via sum of confidences)
    if attractor_mask is not None:
        num_confident = jnp.sum(confidences * attractor_mask)
    else:
        num_confident = jnp.sum(confidences)

    if target_count is not None:
        # Supervised: penalize deviation from target
        diff = num_confident - target_count
        # Asymmetric penalty
        penalty_multiplier = jnp.where(diff > 0, over_penalty, 1.0)
        return penalty_multiplier * diff**2
    else:
        # Unsupervised: soft cap at max_count
        assert max_count is not None  # guaranteed by earlier check
        excess = jnp.maximum(0.0, num_confident - max_count)
        return over_penalty * excess**2


def temporal_spread_loss(
    frames: Float[Array, "num_frames dim"],
    attractors: Float[Array, "num_attractors dim"],
    tau: float | Float[Array, ""],
    num_segments: int = 10,
    min_segments: int = 3,
    presence_threshold: float = 1.0,
    attractor_mask: Float[Array, " num_attractors"] | None = None,
) -> Float[Array, ""]:
    """Penalize attractors that don't persist across time.

    Divides the timeline into segments and checks that each attractor
    has meaningful presence in multiple segments. This encourages
    attractors to represent speakers who appear throughout the audio,
    not transient noise.

    Args:
        frames: [N, D] frame embeddings (full sequence)
        attractors: [K, D] attractor embeddings
        tau: Temperature for soft assignment
        num_segments: Number of temporal segments to divide audio into
        min_segments: Minimum segments an attractor should appear in
        presence_threshold: Minimum mass in a segment to count as "present"
        attractor_mask: [K] binary mask for valid attractors

    Returns:
        Scalar temporal spread loss
    """
    N, D = frames.shape
    K = attractors.shape[0]

    # Compute soft assignment weights
    # ||x - a||^2
    frames_sq = jnp.sum(frames**2, axis=-1, keepdims=True)  # [N, 1]
    attractors_sq = jnp.sum(attractors**2, axis=-1, keepdims=True)  # [K, 1]
    distances = frames_sq + attractors_sq.T - 2 * (frames @ attractors.T)  # [N, K]
    distances = jnp.maximum(distances, 0.0)

    # Softmin assignment
    logits = -distances / tau
    if attractor_mask is not None:
        mask_penalty = jnp.where(attractor_mask > 0.5, 0.0, -jnp.inf)
        logits = logits + mask_penalty[None, :]
    weights = jax.nn.softmax(logits, axis=-1)  # [N, K]

    # Divide into segments
    segment_size = N // num_segments
    usable_frames = num_segments * segment_size

    # Trim and reshape: [num_segments, segment_size, K]
    weights_trimmed = weights[:usable_frames]
    segmented = weights_trimmed.reshape(num_segments, segment_size, K)

    # Mass per segment per attractor: [num_segments, K]
    segment_mass = jnp.sum(segmented, axis=1)

    # Presence: does attractor have enough mass in this segment?
    presence = (segment_mass > presence_threshold).astype(jnp.float32)  # [num_segments, K]

    # Count segments where each attractor is present: [K]
    num_present = jnp.sum(presence, axis=0)

    # Penalize if attractor appears in fewer than min_segments
    deficit = jnp.maximum(0.0, min_segments - num_present)  # [K]

    # Only count valid attractors
    if attractor_mask is not None:
        deficit = deficit * attractor_mask

    return jnp.sum(deficit)


def compute_regularization(
    frames: Float[Array, "num_frames dim"],
    attractors: Float[Array, "num_attractors dim"],
    confidences: Float[Array, " num_attractors"],
    tau: float | Float[Array, ""],
    target_count: int | Float[Array, ""] | None = None,
    max_count: int | None = 4,
    lambda_cardinality: float = 1.0,
    lambda_spread: float = 0.1,
    over_penalty: float = 2.0,
    num_segments: int = 10,
    min_segments: int = 3,
    presence_threshold: float = 1.0,
    attractor_mask: Float[Array, " num_attractors"] | None = None,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """Compute all regularization terms.

    Args:
        frames: [N, D] frame embeddings
        attractors: [K, D] attractor embeddings
        confidences: [K] confidence scores
        tau: Temperature for soft assignment
        target_count: Exact speaker count (supervised) or None
        max_count: Maximum speakers (unsupervised cap)
        lambda_cardinality: Weight for cardinality loss
        lambda_spread: Weight for temporal spread loss
        over_penalty: Asymmetry for cardinality (>1 penalizes over-counting more)
        num_segments: Temporal segments for spread calculation
        min_segments: Minimum segments for presence
        presence_threshold: Mass threshold for segment presence
        attractor_mask: [K] binary mask for valid attractors

    Returns:
        (total_reg, reg_dict): Total regularization and breakdown
    """
    l_card = cardinality_loss(
        confidences=confidences,
        target_count=target_count,
        max_count=max_count if target_count is None else None,
        over_penalty=over_penalty,
        attractor_mask=attractor_mask,
    )

    l_spread = temporal_spread_loss(
        frames=frames,
        attractors=attractors,
        tau=tau,
        num_segments=num_segments,
        min_segments=min_segments,
        presence_threshold=presence_threshold,
        attractor_mask=attractor_mask,
    )

    total = lambda_cardinality * l_card + lambda_spread * l_spread

    reg_dict = {
        "cardinality": l_card,
        "temporal_spread": l_spread,
        "total_regularization": total,
    }

    return total, reg_dict
