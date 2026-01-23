"""
Predictor for WavLeJEPA.

Predicts target representations from context representations using
learnable mask tokens at target positions.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .context_encoder import SinusoidalPositionalEncoding, TransformerEncoderLayer


class Predictor(eqx.Module):
    """Predictor network for JEPA.

    Takes context encoder outputs and predicts target representations
    at masked positions. Uses learnable mask tokens for target positions.

    Architecture:
    1. Project context from context_dim (768) → predictor_dim (384)
    2. Insert learnable mask tokens at target positions
    3. Add positional encodings based on original sequence positions
    4. Process through transformer layers
    5. Project back predictor_dim (384) → output_dim (768)
    """

    context_dim: int = eqx.field(static=True)
    predictor_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)

    # Projections
    input_proj: eqx.nn.Linear  # context_dim → predictor_dim
    output_proj: eqx.nn.Linear  # predictor_dim → output_dim

    # Learnable mask token
    mask_token: Float[Array, " predictor_dim"]

    # Positional encoding
    pos_encoding: SinusoidalPositionalEncoding

    # Transformer layers
    layers: list[TransformerEncoderLayer]
    final_norm: eqx.nn.LayerNorm

    def __init__(
        self,
        context_dim: int = 768,
        predictor_dim: int = 384,
        output_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ffn_dim: int = 1536,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
        max_seq_len: int = 1000,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            context_dim: Dimension of context encoder output (default 768)
            predictor_dim: Internal dimension of predictor (default 384)
            output_dim: Output dimension, should match target dim (default 768)
            num_heads: Number of attention heads (default 12)
            num_layers: Number of transformer layers (default 12)
            ffn_dim: Feed-forward hidden dimension (default 1536 = 4 * 384)
            dropout: Dropout rate (default 0.0)
            layer_norm_eps: LayerNorm epsilon (default 1e-6)
            max_seq_len: Maximum sequence length for positional encoding
            key: JAX PRNG key
        """
        self.context_dim = context_dim
        self.predictor_dim = predictor_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        keys = jax.random.split(key, num_layers + 3)

        # Input projection: context_dim → predictor_dim
        self.input_proj = eqx.nn.Linear(
            in_features=context_dim,
            out_features=predictor_dim,
            use_bias=True,
            key=keys[0],
        )

        # Output projection: predictor_dim → output_dim
        self.output_proj = eqx.nn.Linear(
            in_features=predictor_dim,
            out_features=output_dim,
            use_bias=True,
            key=keys[1],
        )

        # Learnable mask token (initialized from normal distribution)
        self.mask_token = jax.random.normal(keys[2], (predictor_dim,)) * 0.02

        # Positional encoding at predictor dimension
        self.pos_encoding = SinusoidalPositionalEncoding(
            embed_dim=predictor_dim,
            max_len=max_seq_len,
        )

        # Transformer layers
        self.layers = [
            TransformerEncoderLayer(
                embed_dim=predictor_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                key=keys[i + 3],
            )
            for i in range(num_layers)
        ]

        # Final layer norm
        self.final_norm = eqx.nn.LayerNorm(
            shape=predictor_dim,
            eps=layer_norm_eps,
        )

    def __call__(
        self,
        context_output: Float[Array, "context_len context_dim"],
        context_positions: Int[Array, " context_len"],
        target_positions: Int[Array, " target_len"],
        num_context: Int[Array, ""],
        num_targets: Int[Array, ""],
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: bool = False,
    ) -> Float[Array, "target_len output_dim"]:
        """
        Predict target representations at specified positions.

        Args:
            context_output: Context encoder output [context_len, context_dim]
            context_positions: Indices of context positions in original sequence
            target_positions: Indices of target positions to predict
            num_context: Number of valid (non-padding) context positions
            num_targets: Number of valid (non-padding) target positions
            key: PRNG key for dropout
            inference: If True, disable dropout

        Returns:
            Predicted target representations [target_len, output_dim]
        """
        if key is not None:
            keys = jax.random.split(key, self.num_layers)
        else:
            keys = [None] * self.num_layers

        context_len = context_output.shape[0]
        target_len = target_positions.shape[0]

        # Project context to predictor dimension
        assert self.input_proj.bias is not None
        context_proj = (
            context_output @ self.input_proj.weight.T + self.input_proj.bias
        )  # [context_len, predictor_dim]

        # Create mask tokens for target positions
        mask_tokens = jnp.broadcast_to(
            self.mask_token, (target_len, self.predictor_dim)
        )  # [target_len, predictor_dim]

        # Concatenate context and mask tokens
        # Order: context tokens first, then mask tokens
        x = jnp.concatenate(
            [context_proj, mask_tokens], axis=0
        )  # [context_len + target_len, predictor_dim]

        # Build combined position indices for positional encoding
        all_positions = jnp.concatenate([context_positions, target_positions])

        # Add positional encodings based on original positions
        pos_encodings = self.pos_encoding.get_encoding(all_positions)
        x = x + pos_encodings

        # Build attention mask to exclude padded positions
        # Equinox uses True = can attend, so we mark valid positions as True
        ctx_valid = jnp.arange(context_len) < num_context
        tgt_valid = jnp.arange(target_len) < num_targets
        valid = jnp.concatenate([ctx_valid, tgt_valid])  # [context_len + target_len]
        # Key masking: queries can only attend to valid keys
        total_len = context_len + target_len
        attn_mask = jnp.broadcast_to(valid[None, :], (total_len, total_len))

        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=attn_mask, key=keys[i], inference=inference)

        # Final normalization
        x = jax.vmap(self.final_norm)(x)

        # Extract only the target positions (mask token outputs)
        target_output = x[context_len:]  # [target_len, predictor_dim]

        # Project to output dimension
        assert self.output_proj.bias is not None
        predictions = (
            target_output @ self.output_proj.weight.T + self.output_proj.bias
        )  # [target_len, output_dim]

        return predictions
