"""
WavLeJEPA: Waveform-based Lean JEPA.

Full model assembly combining WavJEPA's time-domain processing
with LeJEPA's SIGReg regularization.
"""

from dataclasses import dataclass, fields
from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Bool, Int, PRNGKeyArray

from .waveform_encoder import WaveformEncoder
from .context_encoder import ContextEncoder
from .predictor import Predictor


# =============================================================================
# Masking Utilities
# =============================================================================


@dataclass
class MaskingConfig:
    """Configuration for context/target masking.

    Uses ratio-based configuration to match WavJEPA's effective masking ratios.
    Default values target ~35% context and ~23% targets.
    """

    # Context configuration
    context_ratio: float = 0.35  # Target ratio of sequence as context
    context_block_length: int = 10  # Length of each context block (frames)
    min_context_ratio: float = 0.10  # Minimum acceptable context ratio

    # Target configuration
    # Note: Set higher than desired ratio to compensate for context exclusion
    # (0.28 -> ~23% actual after excluding context positions)
    target_ratio: float = 0.28  # Target ratio of sequence as targets
    target_block_length: int = 10  # Length of each target block (frames)
    num_target_groups: int = 4  # Number of disjoint target groups (WavJEPA default)

    # Retry configuration
    max_retries: int = 10  # Maximum resampling attempts for min_context_ratio


def _expand_block_starts(
    starts: Bool[Array, " seq_len"],
    seq_len: int,
    block_length: int,
) -> Bool[Array, " seq_len"]:
    """Expand block start positions into full blocks."""
    mask = jnp.zeros(seq_len, dtype=bool)
    for offset in range(block_length):
        indices = jnp.arange(seq_len) - offset
        valid = (indices >= 0) & (indices < seq_len)
        shifted_starts = jnp.where(
            valid, starts[jnp.clip(indices, 0, seq_len - 1)], False
        )
        mask = mask | shifted_starts
    return mask


def _num_blocks_for_ratio(seq_len: int, target_ratio: float, block_length: int) -> int:
    """
    Compute number of blocks needed to achieve target coverage ratio.

    With random block placement, blocks overlap. Expected coverage is:
        coverage = 1 - (1 - M/N)^k
    where M=block_length, N=seq_len, k=num_blocks.

    Solving for k: k = log(1 - ratio) / log(1 - M/N)
    """
    import math

    if seq_len <= 0 or block_length <= 0:
        return 1

    p = block_length / seq_len
    if p >= 1.0:
        return 1

    # Clamp target_ratio to avoid log(0)
    ratio = min(target_ratio, 0.99)
    if ratio <= 0:
        return 1

    num_blocks = math.log(1 - ratio) / math.log(1 - p)
    return max(1, int(math.ceil(num_blocks)))


def sample_context_mask(
    seq_len: int,
    config: MaskingConfig,
    key: PRNGKeyArray,
) -> Bool[Array, " seq_len"]:
    """
    Sample context mask using ratio-based block sampling.

    Samples enough blocks to achieve the target context_ratio.

    Args:
        seq_len: Length of sequence
        config: Masking configuration
        key: PRNG key

    Returns:
        context_mask: Boolean mask [seq_len] where True = context position
    """
    num_blocks = _num_blocks_for_ratio(
        seq_len, config.context_ratio, config.context_block_length
    )

    # Sample num_blocks random start positions (without replacement)
    # Use top-k of random values for differentiable selection
    rand_vals = jax.random.uniform(key, (seq_len,))
    _, top_indices = jax.lax.top_k(rand_vals, num_blocks)

    # Create starts mask from selected indices
    starts = jnp.zeros(seq_len, dtype=bool).at[top_indices].set(True)

    # Expand to full blocks
    return _expand_block_starts(starts, seq_len, config.context_block_length)


def sample_target_mask(
    seq_len: int,
    context_mask: Bool[Array, " seq_len"],
    config: MaskingConfig,
    key: PRNGKeyArray,
) -> Bool[Array, " seq_len"]:
    """
    Sample target mask from non-context positions using ratio-based block sampling.

    Targets are sampled only from positions not already marked as context,
    ensuring disjoint masks.

    Args:
        seq_len: Length of sequence
        context_mask: Existing context mask
        config: Masking configuration
        key: PRNG key

    Returns:
        target_mask: Boolean mask [seq_len] where True = target position
    """
    # Compute blocks needed based on full sequence length
    # (context exclusion happens after block expansion)
    num_blocks = _num_blocks_for_ratio(
        seq_len, config.target_ratio, config.target_block_length
    )

    # Sample from non-context positions only
    # Set context positions to -inf so they're never selected
    rand_vals = jax.random.uniform(key, (seq_len,))
    rand_vals = jnp.where(context_mask, -jnp.inf, rand_vals)

    # Get top num_blocks positions from non-context area
    _, top_indices = jax.lax.top_k(rand_vals, num_blocks)

    # Create starts mask from selected indices
    starts = jnp.zeros(seq_len, dtype=bool).at[top_indices].set(True)

    # Expand to full blocks, then remove any overlap with context
    target_mask = _expand_block_starts(starts, seq_len, config.target_block_length)
    target_mask = target_mask & ~context_mask

    return target_mask


def sample_masks(
    seq_len: int,
    config: MaskingConfig,
    key: PRNGKeyArray,
) -> tuple[Bool[Array, " seq_len"], Bool[Array, " seq_len"]]:
    """
    Sample context and target masks with retry mechanism.

    Resamples if context ratio falls below min_context_ratio.
    Uses jax.lax.while_loop for JIT compatibility.

    Args:
        seq_len: Length of sequence
        config: Masking configuration
        key: PRNG key

    Returns:
        context_mask: Boolean mask [seq_len] where True = context position
        target_mask: Boolean mask [seq_len] where True = target position
    """
    max_retries = config.max_retries
    min_context = config.min_context_ratio

    def cond_fn(state):
        context_mask, _, _, attempt = state
        context_ratio = jnp.sum(context_mask) / seq_len
        return (context_ratio < min_context) & (attempt < max_retries)

    def body_fn(state):
        _, _, key, attempt = state
        key, k1, k2 = jax.random.split(key, 3)
        context_mask = sample_context_mask(seq_len, config, k1)
        target_mask = sample_target_mask(seq_len, context_mask, config, k2)
        return (context_mask, target_mask, key, attempt + 1)

    # Initial sampling
    key, k1, k2 = jax.random.split(key, 3)
    init_context = sample_context_mask(seq_len, config, k1)
    init_target = sample_target_mask(seq_len, init_context, config, k2)
    init_state = (init_context, init_target, key, jnp.array(0))

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return final_state[0], final_state[1]


def sample_target_groups(
    seq_len: int,
    context_mask: Bool[Array, " seq_len"],
    config: MaskingConfig,
    key: PRNGKeyArray,
) -> Bool[Array, "num_groups seq_len"]:
    """
    Sample multiple disjoint target groups.

    Each target group is disjoint from the context and from all other target groups.
    This prevents shortcut learning where targets copy from nearby targets.

    Args:
        seq_len: Length of sequence
        context_mask: Existing context mask [seq_len]
        config: Masking configuration (includes num_target_groups)
        key: PRNG key

    Returns:
        target_masks: Boolean masks [num_groups, seq_len] where True = target position
    """
    num_groups = config.num_target_groups

    def sample_one_group(carry, key_i):
        used_positions = carry
        target_mask = sample_target_mask(seq_len, used_positions, config, key_i)
        new_used = used_positions | target_mask
        return new_used, target_mask

    keys = jax.random.split(key, num_groups)
    _, target_masks = jax.lax.scan(sample_one_group, context_mask, keys)

    return target_masks  # [num_groups, seq_len]


def sample_masks_with_groups(
    seq_len: int,
    config: MaskingConfig,
    key: PRNGKeyArray,
) -> tuple[Bool[Array, " seq_len"], Bool[Array, "num_groups seq_len"]]:
    """
    Sample context mask and multiple disjoint target groups with retry mechanism.

    Resamples if context ratio falls below min_context_ratio.
    Uses jax.lax.while_loop for JIT compatibility.

    Args:
        seq_len: Length of sequence
        config: Masking configuration
        key: PRNG key

    Returns:
        context_mask: Boolean mask [seq_len] where True = context position
        target_masks: Boolean masks [num_groups, seq_len] where True = target position
    """
    max_retries = config.max_retries
    min_context = config.min_context_ratio
    num_groups = config.num_target_groups

    def cond_fn(state):
        context_mask, _, _, attempt = state
        context_ratio = jnp.sum(context_mask) / seq_len
        return (context_ratio < min_context) & (attempt < max_retries)

    def body_fn(state):
        _, _, key, attempt = state
        key, k1, k2 = jax.random.split(key, 3)
        context_mask = sample_context_mask(seq_len, config, k1)
        target_masks = sample_target_groups(seq_len, context_mask, config, k2)
        return (context_mask, target_masks, key, attempt + 1)

    # Initial sampling
    key, k1, k2 = jax.random.split(key, 3)
    init_context = sample_context_mask(seq_len, config, k1)
    init_targets = sample_target_groups(seq_len, init_context, config, k2)
    init_state = (init_context, init_targets, key, jnp.array(0))

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return final_state[0], final_state[1]


def mask_to_indices(
    mask: Bool[Array, " seq_len"],
    max_len: int,
) -> tuple[Int[Array, " max_len"], Int[Array, ""]]:
    """
    Convert boolean mask to padded indices array.

    Args:
        mask: Boolean mask [seq_len]
        max_len: Maximum number of indices to return (for static shapes)

    Returns:
        indices: Padded indices array [max_len], with -1 for padding
        count: Number of valid indices
    """
    # Count valid indices
    count = jnp.sum(mask)

    # Gather valid indices
    valid_indices = jnp.where(mask, size=max_len, fill_value=-1)[0]

    return valid_indices, count


# =============================================================================
# Model Configuration
# =============================================================================


@dataclass
class WavLeJEPAConfig:
    """Full model configuration."""

    # Waveform Encoder
    waveform_embed_dim: int = 768
    waveform_num_groups: int = 32

    # Context Encoder
    context_embed_dim: int = 768
    context_num_heads: int = 12
    context_num_layers: int = 12
    context_ffn_dim: int = 3072
    context_dropout: float = 0.0
    context_top_k_layers: int = 8
    context_top_k_norm: str = "instance"

    # Predictor
    predictor_dim: int = 384
    predictor_num_heads: int = 12
    predictor_num_layers: int = 12
    predictor_ffn_dim: int = 1536
    predictor_dropout: float = 0.0

    # Masking
    masking: MaskingConfig = None  # type: ignore

    # Sequence length
    max_seq_len: int = 1000

    def __post_init__(self):
        if self.masking is None:
            self.masking = MaskingConfig()

    @classmethod
    def from_dict(cls, data: dict) -> "WavLeJEPAConfig":
        """Load config from dict, ignoring unknown keys."""
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)


# =============================================================================
# WavLeJEPA Model
# =============================================================================


class WavLeJEPA(eqx.Module):
    """WavLeJEPA: Waveform-based Lean JEPA.

    Combines WavJEPA's time-domain processing with LeJEPA's
    SIGReg regularization for heuristics-free self-supervised learning.

    Components:
    - WaveformEncoder: Raw audio → embeddings at 100Hz
    - ContextEncoder: Embeddings → contextualized representations
    - Predictor: Context → predicted target representations
    """

    config: WavLeJEPAConfig = eqx.field(static=True)

    waveform_encoder: WaveformEncoder
    context_encoder: ContextEncoder
    predictor: Predictor

    def __init__(self, config: WavLeJEPAConfig, *, key: PRNGKeyArray):
        self.config = config

        keys = jax.random.split(key, 3)

        self.waveform_encoder = WaveformEncoder(
            embed_dim=config.waveform_embed_dim,
            num_groups=config.waveform_num_groups,
            key=keys[0],
        )

        self.context_encoder = ContextEncoder(
            embed_dim=config.context_embed_dim,
            num_heads=config.context_num_heads,
            num_layers=config.context_num_layers,
            ffn_dim=config.context_ffn_dim,
            dropout=config.context_dropout,
            max_seq_len=config.max_seq_len,
            top_k_layers=config.context_top_k_layers,
            top_k_norm=config.context_top_k_norm,
            key=keys[1],
        )

        self.predictor = Predictor(
            context_dim=config.context_embed_dim,
            predictor_dim=config.predictor_dim,
            output_dim=config.context_embed_dim,  # Match context encoder output
            num_heads=config.predictor_num_heads,
            num_layers=config.predictor_num_layers,
            ffn_dim=config.predictor_ffn_dim,
            dropout=config.predictor_dropout,
            max_seq_len=config.max_seq_len,
            key=keys[2],
        )

    def forward_train(
        self,
        waveform: Float[Array, " time"],
        *,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        """
        Training forward pass with multiple target groups.

        JIT-compatible: uses fixed-size arrays with validity counts instead
        of dynamic boolean indexing.

        Each target group is predicted independently - targets in one group
        cannot attend to targets in other groups, preventing shortcut learning.

        Args:
            waveform: Raw audio waveform [T] at 16kHz
            key: PRNG key

        Returns:
            Dictionary containing:
            - predictions: Predicted representations [num_groups, seq_len, 768]
            - targets: Actual target representations [num_groups, seq_len, 768]
            - context_embeddings: Context representations [seq_len, 768]
            - context_mask: Boolean context mask [seq_len]
            - target_masks: Boolean target masks [num_groups, seq_len]
            - num_context: Number of valid context positions
            - num_targets_per_group: Number of valid targets per group [num_groups]
        """
        key1, key2, key3 = jax.random.split(key, 3)

        # 1. Extract waveform features
        features = self.waveform_encoder(waveform)  # [N, 768]
        seq_len = features.shape[0]
        num_groups = self.config.masking.num_target_groups

        # 2. Sample context and multiple disjoint target groups
        context_mask, target_masks = sample_masks_with_groups(
            seq_len, self.config.masking, key2
        )  # context: [seq_len], targets: [num_groups, seq_len]

        # 3. Get context position indices
        context_positions, num_context = mask_to_indices(context_mask, seq_len)
        context_positions = jnp.where(context_positions >= 0, context_positions, 0)

        # 4. Get target position indices for each group
        def get_target_indices(target_mask):
            positions, count = mask_to_indices(target_mask, seq_len)
            positions = jnp.where(positions >= 0, positions, 0)
            return positions, count

        target_positions_per_group, num_targets_per_group = jax.vmap(get_target_indices)(
            target_masks
        )  # [num_groups, seq_len], [num_groups]

        # 5. Encode with context masking (all positions, but attention restricted)
        context_output = self.context_encoder.forward_with_top_k(
            features,
            context_mask=context_mask,
            key=key1,
            inference=False,
        )
        context_at_positions = context_output[context_positions]  # [seq_len, 768]

        # 6. Get target representations (encode full sequence without masking)
        full_output = self.context_encoder.forward_with_top_k(
            features,
            context_mask=None,
            key=key3,
            inference=False,
        )

        # Gather targets for each group: [num_groups, seq_len, 768]
        targets = full_output[target_positions_per_group]

        # 7. Predict targets from context for each group using vmap
        # Tile context for each group
        context_tiled = jnp.broadcast_to(
            context_at_positions[None, :, :],
            (num_groups, context_at_positions.shape[0], context_at_positions.shape[1]),
        )
        context_positions_tiled = jnp.broadcast_to(
            context_positions[None, :], (num_groups, context_positions.shape[0])
        )
        num_context_tiled = jnp.broadcast_to(num_context, (num_groups,))

        # vmap predictor over groups
        # Extract predictor to avoid capturing full model (self) in closure
        predictor = self.predictor
        predictions = jax.vmap(
            lambda ctx, ctx_pos, tgt_pos, n_ctx, n_tgt: predictor(
                context_output=ctx,
                context_positions=ctx_pos,
                target_positions=tgt_pos,
                num_context=n_ctx,
                num_targets=n_tgt,
                inference=False,
            )
        )(
            context_tiled,
            context_positions_tiled,
            target_positions_per_group,
            num_context_tiled,
            num_targets_per_group,
        )  # [num_groups, seq_len, 768]

        return {
            "predictions": predictions,
            "targets": targets,
            "context_embeddings": context_at_positions,
            "context_mask": context_mask,
            "target_masks": target_masks,
            "num_context": num_context,
            "num_targets_per_group": num_targets_per_group,
        }

    def extract_features(
        self,
        waveform: Float[Array, " time"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "frames embed_dim"]:
        """
        Extract features for downstream tasks.

        Uses Top-K layer averaging on full sequence (no masking).

        Args:
            waveform: Raw audio waveform [T] at 16kHz
            key: Optional PRNG key (not used in inference mode)

        Returns:
            Features [frames, embed_dim]
        """
        features = self.waveform_encoder(waveform)
        output = self.context_encoder.forward_with_top_k(
            features,
            context_mask=None,
            key=key,
            inference=True,
        )
        return output

    def count_params(self) -> dict[str, int]:
        """Count parameters in each component."""

        def count(module):
            params = eqx.filter(module, eqx.is_array)
            return sum(x.size for x in jax.tree_util.tree_leaves(params))

        return {
            "waveform_encoder": count(self.waveform_encoder),
            "context_encoder": count(self.context_encoder),
            "predictor": count(self.predictor),
            "total": count(self),
        }
