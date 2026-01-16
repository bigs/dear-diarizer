"""Linear attention layers for the Generator.

This module provides Equinox wrappers around SSM implementations:
- Mamba2 (via mamba2-jax SSD)
- Gated DeltaNet (via jax_gated_deltanet)

The SSM backend is selected based on GeneratorConfig.
"""

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from einops import rearrange

from jax_gated_deltanet import GatedDeltaNetBlock

from .config import GeneratorConfig
from .ssd import ssd_stable


class RMSNorm(eqx.Module):
    """Root Mean Square Layer Normalization."""

    weight: Float[Array, " dim"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-6, *, key: PRNGKeyArray):
        del key  # unused, but kept for consistent API
        self.weight = jnp.ones(dim)
        self.eps = eps

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(x_f32**2, axis=-1, keepdims=True)
        x_normed = x_f32 * jax.lax.rsqrt(variance + self.eps)
        return (x_normed * self.weight).astype(x.dtype)


class Mamba2Layer(eqx.Module):
    """Single Mamba2 layer (SSM-based linear attention).

    Wraps the mamba2-jax SSD implementation with Equinox parameter management.
    Simplified from the full Mamba2Mixer - no vocab embedding, no d_mlp split.
    """

    config: GeneratorConfig = eqx.field(static=True)

    # Projections
    in_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    # Depthwise conv1d (causal)
    conv1d: eqx.nn.Conv1d

    # SSM parameters
    dt_bias: Float[Array, " num_heads"]
    A_log: Float[Array, " num_heads"]
    D: Float[Array, " num_heads"]

    # Normalization
    norm: RMSNorm

    def __init__(self, config: GeneratorConfig, *, key: PRNGKeyArray):
        self.config = config
        m2 = config.mamba2  # Mamba2-specific config
        assert m2 is not None, "Mamba2Layer requires config.mamba2 to be set"

        keys = jax.random.split(key, 5)

        hidden_dim = config.hidden_dim
        intermediate_size = m2.intermediate_size(hidden_dim)
        state_size = m2.state_size
        num_heads = m2.num_heads(hidden_dim)

        # Input projection: hidden_dim -> intermediate_size + 2*state_size + num_heads
        # (x, B, C, dt) - no z/gate in simplified version
        in_proj_dim = intermediate_size + 2 * state_size + num_heads
        self.in_proj = eqx.nn.Linear(hidden_dim, in_proj_dim, use_bias=False, key=keys[0])

        # Conv1d over (x, B, C) - depthwise, causal
        conv1d_dim = intermediate_size + 2 * state_size
        self.conv1d = eqx.nn.Conv1d(
            in_channels=conv1d_dim,
            out_channels=conv1d_dim,
            kernel_size=m2.conv_kernel,
            padding=((m2.conv_kernel - 1, 0),),  # causal: left-pad only
            groups=conv1d_dim,  # depthwise
            use_bias=True,
            key=keys[1],
        )

        # dt_bias initialization (inverse softplus of uniform in log space)
        dt_min, dt_max = m2.time_step_min, m2.time_step_max
        dt_floor = m2.time_step_floor
        u = jax.random.uniform(keys[2], (num_heads,))
        dt = jnp.exp(u * (jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min))
        dt = jnp.maximum(dt, dt_floor)
        # inverse softplus: x = y + log(-expm1(-y))
        self.dt_bias = dt + jnp.log(-jnp.expm1(-dt))

        # A_log initialization (log of uniform in A_initializer_range)
        A_low, A_high = m2.A_initializer_range
        A = jax.random.uniform(keys[3], (num_heads,), minval=A_low, maxval=A_high)
        self.A_log = jnp.log(A)

        # D (residual connection weight)
        self.D = jnp.ones(num_heads)

        # RMSNorm before output projection
        self.norm = RMSNorm(intermediate_size, key=keys[4])

        # Output projection
        self.out_proj = eqx.nn.Linear(intermediate_size, hidden_dim, use_bias=False, key=keys[4])

    def __call__(
        self,
        x: Float[Array, "seq_len hidden_dim"],
        *,
        initial_state: Optional[Float[Array, "num_heads head_dim state_size"]] = None,
        return_final_state: bool = False,
    ) -> Tuple[Float[Array, "seq_len hidden_dim"], Optional[Float[Array, "num_heads head_dim state_size"]]]:
        """Forward pass through Mamba2 layer.

        Args:
            x: Input tensor [seq_len, hidden_dim]
            initial_state: Optional initial SSM state [num_heads, head_dim, state_size]
            return_final_state: Whether to return final SSM state

        Returns:
            (output, final_state): Output tensor and optional final state
        """
        cfg = self.config
        m2 = cfg.mamba2
        assert m2 is not None  # Guaranteed by __init__ assertion
        seq_len = x.shape[0]
        intermediate_size = cfg.intermediate_size
        state_size = m2.state_size

        # Project input
        xBC_dt = jax.vmap(self.in_proj)(x)  # [seq_len, in_proj_dim]

        # Split into components
        x_conv, B_t, C_t, dt = jnp.split(
            xBC_dt,
            [
                intermediate_size,
                intermediate_size + state_size,
                intermediate_size + 2 * state_size,
            ],
            axis=-1,
        )
        # x_conv: [seq_len, intermediate_size]
        # B_t, C_t: [seq_len, state_size]
        # dt: [seq_len, num_heads]

        # Causal conv1d over x, B, C
        xBC = jnp.concatenate([x_conv, B_t, C_t], axis=-1)  # [seq_len, conv1d_dim]
        xBC = rearrange(xBC, "l d -> d l")  # Conv1d expects [channels, length]
        xBC = self.conv1d(xBC)
        xBC = jax.nn.silu(xBC)
        xBC = rearrange(xBC, "d l -> l d")  # Back to [length, channels]

        # Split back
        x_ssm, B_ssm, C_ssm = jnp.split(
            xBC,
            [intermediate_size, intermediate_size + state_size],
            axis=-1,
        )

        # Prepare for SSD: add batch dimension
        x_batched = x_ssm[None, ...]  # [1, seq_len, intermediate_size]
        dt_batched = dt[None, ...]  # [1, seq_len, num_heads]
        B_batched = B_ssm[None, ...]  # [1, seq_len, state_size]
        C_batched = C_ssm[None, ...]  # [1, seq_len, state_size]

        # Reshape x for SSD: [batch, seq_len, num_heads, head_dim]
        num_heads = cfg.num_heads
        head_dim = m2.head_dim
        x_ssd = rearrange(x_batched, "b l (h p) -> b l h p", p=head_dim)

        # Broadcast B, C across heads: [batch, seq_len, num_heads, state_size]
        B_ssd = jnp.broadcast_to(
            B_batched[:, :, None, :],
            (1, seq_len, num_heads, state_size),
        )
        C_ssd = jnp.broadcast_to(
            C_batched[:, :, None, :],
            (1, seq_len, num_heads, state_size),
        )

        # Prepare initial state if provided
        init_state = None
        if initial_state is not None:
            init_state = initial_state[None, None, ...]  # [1, 1, num_heads, head_dim, state_size]

        # Run SSD (using our stable implementation)
        A = -jnp.exp(self.A_log.astype(jnp.float32))
        y, final_state = ssd_stable(
            x=x_ssd,
            dt=dt_batched,
            A=A,
            B=B_ssd,
            C=C_ssd,
            chunk_size=m2.chunk_size,
            D=self.D,
            dt_bias=self.dt_bias,
            dt_min=m2.time_step_limit[0],
            dt_max=m2.time_step_limit[1],
            initial_states=init_state,
            return_final_state=return_final_state,
        )

        # Reshape output: [1, seq_len, num_heads, head_dim] -> [seq_len, intermediate_size]
        y = rearrange(y, "b l h p -> b l (h p)")
        y = y[0]  # Remove batch dim

        # Normalize and project
        y = jax.vmap(self.norm)(y)
        y = jax.vmap(self.out_proj)(y)

        # Process final state if returned
        if final_state is not None:
            final_state = final_state[0]  # Remove batch dim: [num_heads, head_dim, state_size]

        return y, final_state


class Mamba2Block(eqx.Module):
    """Mamba2 block with residual connection and pre-norm."""

    norm: RMSNorm
    mamba: Mamba2Layer

    def __init__(self, config: GeneratorConfig, *, key: PRNGKeyArray):
        keys = jax.random.split(key, 2)
        self.norm = RMSNorm(config.hidden_dim, key=keys[0])
        self.mamba = Mamba2Layer(config, key=keys[1])

    def __call__(
        self,
        x: Float[Array, "seq_len hidden_dim"],
        *,
        initial_state: Optional[Float[Array, "num_heads head_dim state_size"]] = None,
        return_final_state: bool = False,
    ) -> Tuple[Float[Array, "seq_len hidden_dim"], Optional[Float[Array, "num_heads head_dim state_size"]]]:
        """Forward with pre-norm and residual."""
        residual = x
        x = jax.vmap(self.norm)(x)
        x, final_state = self.mamba(x, initial_state=initial_state, return_final_state=return_final_state)
        return residual + x, final_state


class LinearAttentionStack(eqx.Module):
    """Stack of SSM blocks for contextualizing frame embeddings.

    This is the first stage of the Generator pipeline:
    WavLeJEPA embeddings -> LinearAttentionStack -> Contextualized frames

    The SSM backend (Mamba2 or GatedDeltaNet) is selected based on
    which config is provided in GeneratorConfig.
    """

    config: GeneratorConfig = eqx.field(static=True)
    layers: list[Union[Mamba2Block, GatedDeltaNetBlock]]
    input_proj: Optional[eqx.nn.Linear]
    final_norm: RMSNorm

    def __init__(self, config: GeneratorConfig, *, key: PRNGKeyArray):
        self.config = config

        keys = jax.random.split(key, config.num_layers + 2)

        # Optional input projection if input_dim != hidden_dim
        if config.input_dim != config.hidden_dim:
            self.input_proj = eqx.nn.Linear(
                config.input_dim, config.hidden_dim, use_bias=False, key=keys[0]
            )
        else:
            self.input_proj = None

        # Instantiate SSM blocks based on config
        if config.mamba2 is not None:
            # Use Mamba2 backend
            self.layers = [
                Mamba2Block(config, key=keys[i + 1])
                for i in range(config.num_layers)
            ]
        else:
            # Use GatedDeltaNet backend
            assert config.deltanet is not None
            self.layers = [
                GatedDeltaNetBlock(config.deltanet, key=keys[i + 1])
                for i in range(config.num_layers)
            ]

        # Final normalization
        self.final_norm = RMSNorm(config.hidden_dim, key=keys[-1])

    def __call__(
        self,
        x: Float[Array, "seq_len input_dim"],
    ) -> Float[Array, "seq_len hidden_dim"]:
        """Contextualize frame embeddings.

        Args:
            x: Frame embeddings from WavLeJEPA [seq_len, input_dim]

        Returns:
            Contextualized representations [seq_len, hidden_dim]
        """
        # Optional input projection
        if self.input_proj is not None:
            x = jax.vmap(self.input_proj)(x)

        # Process through SSM blocks (both backends have same interface)
        for layer in self.layers:
            x, _ = layer(x, return_final_state=False)

        # Final normalization
        x = jax.vmap(self.final_norm)(x)

        return x
