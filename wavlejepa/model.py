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
    """Configuration for context/target masking."""

    # Context block sampling
    context_prob: float = 0.065  # Probability of starting a context block
    context_length: int = 10  # Length of each context block (frames)
    min_context_ratio: float = 0.10  # Minimum ratio of sequence as context

    # Target block sampling
    target_prob: float = 0.025  # Probability of starting a target block
    target_length: int = 10  # Length of each target block (frames)


def sample_context_mask(
    seq_len: int,
    config: MaskingConfig,
    key: PRNGKeyArray,
) -> Bool[Array, " seq_len"]:
    """
    Sample context mask following WavJEPA's strategy.

    Args:
        seq_len: Length of sequence
        config: Masking configuration
        key: PRNG key

    Returns:
        context_mask: Boolean mask [seq_len] where True = context position
    """
    # Sample start positions with probability context_prob
    context_starts = jax.random.bernoulli(key, config.context_prob, (seq_len,))

    # Expand each start to a block of length context_length
    context_mask = jnp.zeros(seq_len, dtype=bool)
    for offset in range(config.context_length):
        # Shift starts and mark as context
        indices = jnp.arange(seq_len) - offset
        valid = (indices >= 0) & (indices < seq_len)
        shifted_starts = jnp.where(
            valid, context_starts[jnp.clip(indices, 0, seq_len - 1)], False
        )
        context_mask = context_mask | shifted_starts

    return context_mask


def sample_target_mask(
    seq_len: int,
    context_mask: Bool[Array, " seq_len"],
    config: MaskingConfig,
    key: PRNGKeyArray,
) -> Bool[Array, " seq_len"]:
    """
    Sample target mask, ensuring no overlap with context.

    Args:
        seq_len: Length of sequence
        context_mask: Existing context mask
        config: Masking configuration
        key: PRNG key

    Returns:
        target_mask: Boolean mask [seq_len] where True = target position
    """
    # Sample start positions with probability target_prob
    target_starts = jax.random.bernoulli(key, config.target_prob, (seq_len,))

    # Expand each start to a block of length target_length
    target_mask = jnp.zeros(seq_len, dtype=bool)
    for offset in range(config.target_length):
        indices = jnp.arange(seq_len) - offset
        valid = (indices >= 0) & (indices < seq_len)
        shifted_starts = jnp.where(
            valid, target_starts[jnp.clip(indices, 0, seq_len - 1)], False
        )
        target_mask = target_mask | shifted_starts

    # Remove overlap with context
    target_mask = target_mask & ~context_mask

    return target_mask


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
        Training forward pass.

        JIT-compatible: uses fixed-size arrays with validity counts instead
        of dynamic boolean indexing.

        Args:
            waveform: Raw audio waveform [T] at 16kHz
            key: PRNG key

        Returns:
            Dictionary containing:
            - predictions: Predicted target representations [max_seq_len, 768]
            - targets: Actual target representations [max_seq_len, 768]
            - context_embeddings: Context representations at masked positions [max_seq_len, 768]
            - context_embeddings: Context representations at masked positions [max_seq_len, 768]
            - context_mask: Boolean context mask [seq_len]
            - target_mask: Boolean target mask [seq_len]
            - num_context: Number of valid context positions
            - num_targets: Number of valid target positions
        """
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)

        # 1. Extract waveform features
        features = self.waveform_encoder(waveform)  # [N, 768]
        seq_len = features.shape[0]

        # 2. Sample context and target masks
        context_mask = sample_context_mask(seq_len, self.config.masking, key2)
        target_mask = sample_target_mask(
            seq_len, context_mask, self.config.masking, key3
        )

        # 3. Get position indices (fixed size, padded with seq_len for invalid)
        # Use seq_len as fill value so invalid indices point to a valid position
        # (we'll use counts to ignore them in loss computation)
        context_positions, num_context = mask_to_indices(context_mask, seq_len)
        target_positions, num_targets = mask_to_indices(target_mask, seq_len)

        # Replace -1 padding with 0 (a valid index) - we use counts to mask later
        context_positions = jnp.where(context_positions >= 0, context_positions, 0)
        target_positions = jnp.where(target_positions >= 0, target_positions, 0)

        # 4. Encode with context masking (all positions, but attention restricted)
        context_output = self.context_encoder.forward_with_top_k(
            features,
            context_mask=context_mask,
            key=key4,
            inference=False,
        )
        # Gather context at positions (fixed size array)
        context_at_positions = context_output[context_positions]  # [seq_len, 768]

        # 5. Get target representations (encode full sequence without masking)
        full_output = self.context_encoder.forward_with_top_k(
            features,
            context_mask=None,  # No masking for targets
            key=key5,
            inference=False,
        )
        targets = full_output[target_positions]  # [seq_len, 768]

        # 6. Predict targets from context
        predictions = self.predictor(
            context_output=context_at_positions,
            context_positions=context_positions,
            target_positions=target_positions,
            inference=False,
        )  # [seq_len, 768]

        return {
            "predictions": predictions,
            "targets": targets,
            "context_embeddings": context_at_positions,
            "context_mask": context_mask,
            "target_mask": target_mask,
            "num_context": num_context,
            "num_targets": num_targets,
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
