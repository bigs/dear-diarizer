"""Gated DeltaNet layer implementations.

Provides the main GatedDeltaNetLayer, Block, and Stack modules.
"""

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange, repeat
from jaxtyping import Array, Float, PRNGKeyArray

from .config import GatedDeltaNetConfig
from .conv import ShortConvolution
from .deltanet import gated_delta_rule, gated_delta_rule_recurrent, l2_normalize
from .norm import FusedRMSNormGated, RMSNorm


class GatedDeltaNetLayer(eqx.Module):
    """Single Gated DeltaNet layer.

    Architecture (following Qwen3-Next/FLA):
        1. Project input to Q, K, V, plus alpha (decay) and beta (learning rate)
        2. Apply short convolutions to Q, K, V (with SiLU activation)
        3. L2 normalize Q and K
        4. Apply grouped value attention (GVA) if num_v_heads > num_heads
        5. Compute gated delta rule recurrence
        6. Apply output gating with FusedRMSNormGated
        7. Project back to hidden_size
    """

    config: GatedDeltaNetConfig = eqx.field(static=True)

    # Input projections
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    a_proj: eqx.nn.Linear  # alpha/decay projection
    b_proj: eqx.nn.Linear  # beta/learning rate projection

    # Output projections
    g_proj: Optional[eqx.nn.Linear]  # gate projection (if use_gate)
    o_proj: eqx.nn.Linear

    # Short convolutions (if use_short_conv)
    q_conv: Optional[ShortConvolution]
    k_conv: Optional[ShortConvolution]
    v_conv: Optional[ShortConvolution]

    # Output normalization
    o_norm: RMSNorm | FusedRMSNormGated

    # Learnable parameters for decay computation
    A_log: Float[Array, " num_v_heads"]
    dt_bias: Float[Array, " num_v_heads"]

    def __init__(self, config: GatedDeltaNetConfig, *, key: PRNGKeyArray):
        self.config = config

        keys = jax.random.split(key, 12)

        # Q, K projections: hidden_size -> key_dim
        self.q_proj = eqx.nn.Linear(
            config.hidden_size, config.key_dim, use_bias=False, key=keys[0]
        )
        self.k_proj = eqx.nn.Linear(
            config.hidden_size, config.key_dim, use_bias=False, key=keys[1]
        )

        # V projection: hidden_size -> value_dim
        self.v_proj = eqx.nn.Linear(
            config.hidden_size, config.value_dim, use_bias=False, key=keys[2]
        )

        # Alpha (decay) and Beta (learning rate) projections
        # Both output num_v_heads values
        self.a_proj = eqx.nn.Linear(
            config.hidden_size, config.num_v_heads, use_bias=False, key=keys[3]
        )
        self.b_proj = eqx.nn.Linear(
            config.hidden_size, config.num_v_heads, use_bias=False, key=keys[4]
        )

        # Output projection: value_dim -> hidden_size
        self.o_proj = eqx.nn.Linear(
            config.value_dim, config.hidden_size, use_bias=False, key=keys[5]
        )

        # Gate projection (if use_gate)
        if config.use_gate:
            self.g_proj = eqx.nn.Linear(
                config.hidden_size, config.value_dim, use_bias=False, key=keys[6]
            )
            self.o_norm = FusedRMSNormGated(config.head_v_dim, eps=config.norm_eps, key=keys[7])
        else:
            self.g_proj = None
            self.o_norm = RMSNorm(config.head_v_dim, eps=config.norm_eps, key=keys[7])

        # Short convolutions (if use_short_conv)
        if config.use_short_conv:
            self.q_conv = ShortConvolution(
                config.key_dim, kernel_size=config.conv_size, key=keys[8]
            )
            self.k_conv = ShortConvolution(
                config.key_dim, kernel_size=config.conv_size, key=keys[9]
            )
            self.v_conv = ShortConvolution(
                config.value_dim, kernel_size=config.conv_size, key=keys[10]
            )
        else:
            self.q_conv = None
            self.k_conv = None
            self.v_conv = None

        # Initialize A_log (decay base) - uniform in [0, a_init_range[1]]
        A = jax.random.uniform(
            keys[11],
            (config.num_v_heads,),
            minval=config.a_init_range[0],
            maxval=config.a_init_range[1],
        )
        self.A_log = jnp.log(jnp.maximum(A, 1e-6))

        # Initialize dt_bias (inverse softplus of uniform in log space)
        dt = jnp.exp(
            jax.random.uniform(keys[11], (config.num_v_heads,))
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        )
        dt = jnp.maximum(dt, config.dt_init_floor)
        # Inverse softplus: x = y + log(-expm1(-y))
        self.dt_bias = dt + jnp.log(-jnp.expm1(-dt))

    def __call__(
        self,
        x: Float[Array, "seq_len hidden_size"],
        initial_state: Optional[Float[Array, "num_v_heads head_k head_v"]] = None,
        return_final_state: bool = False,
    ) -> Tuple[
        Float[Array, "seq_len hidden_size"],
        Optional[Float[Array, "num_v_heads head_k head_v"]],
    ]:
        """Forward pass through Gated DeltaNet layer.

        Args:
            x: Input tensor [seq_len, hidden_size]
            initial_state: Optional initial recurrent state
            return_final_state: Whether to return final state

        Returns:
            (output, final_state): Output and optional final state
        """
        cfg = self.config
        seq_len = x.shape[0]

        # Project to Q, K, V
        q = jax.vmap(self.q_proj)(x)  # [seq_len, key_dim]
        k = jax.vmap(self.k_proj)(x)  # [seq_len, key_dim]
        v = jax.vmap(self.v_proj)(x)  # [seq_len, value_dim]

        # Apply short convolutions (if enabled)
        if cfg.use_short_conv:
            q, _ = self.q_conv(q)
            k, _ = self.k_conv(k)
            v, _ = self.v_conv(v)
        else:
            # Apply SiLU directly if no conv
            q = jax.nn.silu(q)
            k = jax.nn.silu(k)
            v = jax.nn.silu(v)

        # Reshape to heads: [seq_len, num_heads, head_dim]
        q = rearrange(q, "s (h d) -> s h d", d=cfg.head_k_dim)
        k = rearrange(k, "s (h d) -> s h d", d=cfg.head_k_dim)
        v = rearrange(v, "s (h d) -> s h d", d=cfg.head_v_dim)

        # Apply GVA: repeat q, k for grouped value attention
        if cfg.num_v_heads > cfg.num_heads:
            q = repeat(q, "s h d -> s (h g) d", g=cfg.gva_groups)
            k = repeat(k, "s h d -> s (h g) d", g=cfg.gva_groups)

        # L2 normalize q and k (per the Gated DeltaNet spec)
        q = l2_normalize(q)
        k = l2_normalize(k)

        # Compute beta (learning rate) from b_proj
        beta = jax.nn.sigmoid(jax.vmap(self.b_proj)(x))  # [seq_len, num_v_heads]

        # Compute decay gate g in log space
        # g = -exp(A_log) * softplus(a_proj(x) + dt_bias)
        a = jax.vmap(self.a_proj)(x)  # [seq_len, num_v_heads]
        g = -jnp.exp(self.A_log) * jax.nn.softplus(a + self.dt_bias)

        # Add batch dimension for gated_delta_rule
        q_batch = q[None, ...]  # [1, seq_len, num_v_heads, head_k]
        k_batch = k[None, ...]
        v_batch = v[None, ...]
        g_batch = g[None, ...]
        beta_batch = beta[None, ...]

        if initial_state is not None:
            initial_state = initial_state[None, ...]  # [1, num_v_heads, head_k, head_v]

        # Run gated delta rule
        output, final_state = gated_delta_rule(
            q=q_batch,
            k=k_batch,
            v=v_batch,
            g=g_batch,
            beta=beta_batch,
            initial_state=initial_state,
            output_final_state=return_final_state,
        )

        # Remove batch dimension
        output = output[0]  # [seq_len, num_v_heads, head_v]

        # Apply output gating and normalization
        if cfg.use_gate:
            gate = jax.vmap(self.g_proj)(x)  # [seq_len, value_dim]
            gate = rearrange(gate, "s (h d) -> s h d", d=cfg.head_v_dim)
            # Apply FusedRMSNormGated per position and head
            output = jax.vmap(jax.vmap(self.o_norm))(output, gate)
        else:
            output = jax.vmap(jax.vmap(self.o_norm))(output)

        # Reshape back: [seq_len, num_v_heads, head_v] -> [seq_len, value_dim]
        output = rearrange(output, "s h d -> s (h d)")

        # Output projection
        output = jax.vmap(self.o_proj)(output)

        # Process final state
        if final_state is not None:
            final_state = final_state[0]  # Remove batch dim

        return output, final_state


class GatedDeltaNetBlock(eqx.Module):
    """Gated DeltaNet block with pre-norm and residual connection."""

    norm: RMSNorm
    deltanet: GatedDeltaNetLayer

    def __init__(self, config: GatedDeltaNetConfig, *, key: PRNGKeyArray):
        keys = jax.random.split(key, 2)
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps, key=keys[0])
        self.deltanet = GatedDeltaNetLayer(config, key=keys[1])

    def __call__(
        self,
        x: Float[Array, "seq_len hidden_size"],
        initial_state: Optional[Float[Array, "num_v_heads head_k head_v"]] = None,
        return_final_state: bool = False,
    ) -> Tuple[
        Float[Array, "seq_len hidden_size"],
        Optional[Float[Array, "num_v_heads head_k head_v"]],
    ]:
        """Forward with pre-norm and residual."""
        residual = x
        x = jax.vmap(self.norm)(x)
        x, final_state = self.deltanet(x, initial_state, return_final_state)
        return residual + x, final_state


class GatedDeltaNetStack(eqx.Module):
    """Stack of Gated DeltaNet blocks.

    Optionally includes input projection if input_dim != hidden_size.
    """

    config: GatedDeltaNetConfig = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)

    input_proj: Optional[eqx.nn.Linear]
    layers: list[GatedDeltaNetBlock]
    final_norm: RMSNorm

    def __init__(
        self,
        config: GatedDeltaNetConfig,
        input_dim: Optional[int] = None,
        *,
        key: PRNGKeyArray,
    ):
        self.config = config
        self.input_dim = input_dim if input_dim is not None else config.hidden_size

        keys = jax.random.split(key, config.num_layers + 2)

        # Optional input projection
        if self.input_dim != config.hidden_size:
            self.input_proj = eqx.nn.Linear(
                self.input_dim, config.hidden_size, use_bias=False, key=keys[0]
            )
        else:
            self.input_proj = None

        # Stack of blocks
        self.layers = [
            GatedDeltaNetBlock(config, key=keys[i + 1])
            for i in range(config.num_layers)
        ]

        # Final normalization
        self.final_norm = RMSNorm(config.hidden_size, eps=config.norm_eps, key=keys[-1])

    def __call__(
        self,
        x: Float[Array, "seq_len input_dim"],
    ) -> Float[Array, "seq_len hidden_size"]:
        """Process sequence through stack.

        Args:
            x: Input tensor [seq_len, input_dim]

        Returns:
            Output tensor [seq_len, hidden_size]
        """
        # Input projection
        if self.input_proj is not None:
            x = jax.vmap(self.input_proj)(x)

        # Process through blocks
        for layer in self.layers:
            x, _ = layer(x, return_final_state=False)

        # Final normalization
        x = jax.vmap(self.final_norm)(x)

        return x
