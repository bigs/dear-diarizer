"""
Context Encoder for WavLeJEPA.

Standard transformer encoder with:
- Sinusoidal positional encodings
- Pre-LN transformer blocks (using Equinox built-in components)
- Top-K layer averaging for output representations
- Support for context masking
"""

from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Bool, Int, PRNGKeyArray


class SinusoidalPositionalEncoding(eqx.Module):
    """Fixed sinusoidal positional encodings.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    max_len: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    pe: Float[Array, "max_len embed_dim"]

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 5000,
    ):
        self.max_len = max_len
        self.embed_dim = embed_dim

        # Precompute positional encodings
        position = jnp.arange(max_len)[:, None]  # [max_len, 1]
        div_term = jnp.exp(
            jnp.arange(0, embed_dim, 2) * (-jnp.log(10000.0) / embed_dim)
        )  # [embed_dim/2]

        pe = jnp.zeros((max_len, embed_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        self.pe = pe

    def __call__(
        self,
        x: Float[Array, "seq_len embed_dim"],
    ) -> Float[Array, "seq_len embed_dim"]:
        """Add positional encoding to input embeddings."""
        seq_len = x.shape[0]
        return x + self.pe[:seq_len]

    def get_encoding(
        self,
        positions: Int[Array, " num_positions"],
    ) -> Float[Array, "num_positions embed_dim"]:
        """Get positional encodings for specific positions."""
        return self.pe[positions]


class TransformerEncoderLayer(eqx.Module):
    """Standard transformer encoder layer with Pre-LN architecture.

    Uses Equinox built-in components:
    - eqx.nn.MultiheadAttention for self-attention
    - eqx.nn.MLP for feed-forward network

    Architecture:
    - LayerNorm -> MultiHeadAttention -> Dropout -> Residual
    - LayerNorm -> MLP -> Dropout -> Residual
    """

    embed_dim: int = eqx.field(static=True)

    self_attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP

    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
        *,
        key: PRNGKeyArray,
    ):
        self.embed_dim = embed_dim

        keys = jax.random.split(key, 2)

        # Use Equinox's built-in MultiheadAttention
        self.self_attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embed_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=dropout,
            key=keys[0],
        )

        # Use Equinox's built-in MLP
        # depth=1 means: input -> hidden -> output (one hidden layer)
        self.mlp = eqx.nn.MLP(
            in_size=embed_dim,
            out_size=embed_dim,
            width_size=ffn_dim,
            depth=1,
            activation=jax.nn.gelu,
            key=keys[1],
        )

        self.norm1 = eqx.nn.LayerNorm(shape=embed_dim, eps=layer_norm_eps)
        self.norm2 = eqx.nn.LayerNorm(shape=embed_dim, eps=layer_norm_eps)

        self.dropout1 = eqx.nn.Dropout(p=dropout)
        self.dropout2 = eqx.nn.Dropout(p=dropout)

    def __call__(
        self,
        x: Float[Array, "seq_len embed_dim"],
        mask: Optional[Bool[Array, "seq_len seq_len"]] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: bool = False,
    ) -> Float[Array, "seq_len embed_dim"]:
        """
        Args:
            x: Input embeddings [seq_len, embed_dim]
            mask: Attention mask [seq_len, seq_len]. True = attend, False = mask out.
            key: PRNG key for dropout
            inference: If True, disable dropout

        Returns:
            Output embeddings [seq_len, embed_dim]
        """
        if key is not None:
            key1, key2, key3 = jax.random.split(key, 3)
        else:
            key1 = key2 = key3 = None

        # Pre-LN Self-Attention
        residual = x
        x = jax.vmap(self.norm1)(x)
        # eqx.nn.MultiheadAttention expects (query, key, value) - for self-attn all same
        x = self.self_attn(x, x, x, mask=mask, key=key1, inference=inference)
        x = self.dropout1(x, key=key2, inference=inference)
        x = residual + x

        # Pre-LN MLP
        residual = x
        x = jax.vmap(self.norm2)(x)
        x = jax.vmap(self.mlp)(x)
        x = self.dropout2(x, key=key3, inference=inference)
        x = residual + x

        return x


class ContextEncoder(eqx.Module):
    """Transformer encoder for processing context blocks.

    Features:
    - Standard transformer encoder architecture using Equinox components
    - Top-K layer averaging for output representations
    - Support for context masking (attend only to context positions)
    """

    embed_dim: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    top_k_layers: int = eqx.field(static=True)
    top_k_norm: str = eqx.field(static=True)

    pos_encoding: SinusoidalPositionalEncoding
    layers: list[TransformerEncoderLayer]
    final_norm: eqx.nn.LayerNorm

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
        max_seq_len: int = 1000,
        top_k_layers: int = 8,
        top_k_norm: str = "instance",
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            embed_dim: Embedding dimension (default 768)
            num_heads: Number of attention heads (default 12)
            num_layers: Number of transformer layers (default 12)
            ffn_dim: Feed-forward hidden dimension (default 3072 = 4 * 768)
            dropout: Dropout rate (default 0.0)
            layer_norm_eps: LayerNorm epsilon (default 1e-6)
            max_seq_len: Maximum sequence length for positional encoding
            top_k_layers: Number of top layers to average for output (default 8)
            top_k_norm: Normalization for top-k averaging ("instance", "layer", "none")
            key: JAX PRNG key
        """
        if top_k_norm not in {"instance", "layer", "none"}:
            raise ValueError(
                f"top_k_norm must be one of 'instance', 'layer', or 'none' "
                f"(got {top_k_norm!r})"
            )
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.top_k_layers = top_k_layers
        self.top_k_norm = top_k_norm

        keys = jax.random.split(key, num_layers)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            embed_dim=embed_dim,
            max_len=max_seq_len,
        )

        # Transformer layers
        self.layers = [
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                key=keys[i],
            )
            for i in range(num_layers)
        ]

        # Final layer norm
        self.final_norm = eqx.nn.LayerNorm(
            shape=embed_dim,
            eps=layer_norm_eps,
        )

    def __call__(
        self,
        x: Float[Array, "seq_len embed_dim"],
        context_mask: Optional[Bool[Array, " seq_len"]] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: bool = False,
        return_all_layers: bool = False,
    ) -> (
        Float[Array, "seq_len embed_dim"]
        | tuple[
            Float[Array, "seq_len embed_dim"], list[Float[Array, "seq_len embed_dim"]]
        ]
    ):
        """
        Args:
            x: Input embeddings [seq_len, embed_dim]
            context_mask: Boolean mask [seq_len] indicating context positions.
                          If provided, attention is restricted to context positions only.
            key: PRNG key for dropout
            inference: If True, disable dropout
            return_all_layers: If True, return outputs from all layers

        Returns:
            If return_all_layers:
                (output, layer_outputs): Final output and list of all layer outputs
            Else:
                output: Final output embeddings [seq_len, embed_dim]
        """
        if key is not None:
            keys = jax.random.split(key, self.num_layers)
        else:
            keys = [None] * self.num_layers

        # Add positional encoding
        x = self.pos_encoding(x)

        # Build attention mask from context mask
        attn_mask = None
        if context_mask is not None:
            # Attention mask: query can attend to key if key is in context
            # Shape: [seq_len, seq_len] where [i, j] = True if j is context
            seq_len = x.shape[0]
            attn_mask = jnp.broadcast_to(context_mask[None, :], (seq_len, seq_len))

        # Process through transformer layers
        layer_outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=attn_mask, key=keys[i], inference=inference)
            layer_outputs.append(x)

        # Final normalization
        x = jax.vmap(self.final_norm)(x)

        if return_all_layers:
            return x, layer_outputs
        return x

    def forward_with_top_k(
        self,
        x: Float[Array, "seq_len embed_dim"],
        context_mask: Optional[Bool[Array, " seq_len"]] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: bool = False,
    ) -> Float[Array, "seq_len embed_dim"]:
        """Forward pass with Top-K layer averaging.

        Returns the average of instance-normalized outputs from the top K layers.
        This is the recommended method for extracting representations.
        """
        result = self.__call__(
            x,
            context_mask=context_mask,
            key=key,
            inference=inference,
            return_all_layers=True,
        )
        # Type narrowing: we know return_all_layers=True returns a tuple
        assert isinstance(result, tuple)
        _, layer_outputs = result

        # Get top K layers
        k = self.top_k_layers
        top_k_outputs = layer_outputs[-k:]  # Last K layers

        if self.top_k_norm == "instance":
            # Normalize each timestep across embedding dim
            def instance_norm(
                z: Float[Array, "seq_len embed_dim"],
            ) -> Float[Array, "seq_len embed_dim"]:
                mean = jnp.mean(z, axis=-1, keepdims=True)
                var = jnp.var(z, axis=-1, keepdims=True)
                return (z - mean) / jnp.sqrt(var + 1e-6)

            normalized_outputs = [instance_norm(out) for out in top_k_outputs]
        elif self.top_k_norm == "layer":
            # Use trained LayerNorm parameters (shared with final norm)
            normalized_outputs = [
                jax.vmap(self.final_norm)(out) for out in top_k_outputs
            ]
        else:
            normalized_outputs = top_k_outputs

        # Average across layers
        stacked = jnp.stack(normalized_outputs, axis=0)  # [K, seq_len, embed_dim]
        averaged = jnp.mean(stacked, axis=0)  # [seq_len, embed_dim]

        return averaged
