"""Attractor Generator module.

Generates speaker attractors from contextualized frame embeddings using
an iterative GRU-based approach with cross-attention.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from .config import GeneratorConfig
from .mamba import LinearAttentionStack


class AttractorGenerator(eqx.Module):
    """Generate speaker attractors from frame embeddings.

    Architecture:
    1. LinearAttentionStack contextualizes frame embeddings
    2. GRU generates attractors iteratively
    3. Cross-attention queries frames at each GRU step
    4. Confidence head determines when to stop

    The generation loop uses jax.lax.while_loop for dynamic stopping.
    Output is padded to max_attractors with valid_count for masking.
    """

    config: GeneratorConfig = eqx.field(static=True)

    # Frame processing
    linear_attn_stack: LinearAttentionStack

    # Cross-attention (GRU hidden queries contextualized frames)
    cross_attn: eqx.nn.MultiheadAttention

    # GRU generator
    # Input: [prev_attractor; cross_attn_output] = 2 * hidden_dim
    gru_cell: eqx.nn.GRUCell

    # Learned start token (used as prev_attractor at step 0)
    start_token: Float[Array, " attractor_dim"]

    # Output heads
    attractor_head: eqx.nn.Linear  # hidden_dim -> attractor_dim
    confidence_head: eqx.nn.Linear  # hidden_dim -> 1

    def __init__(self, config: GeneratorConfig, *, key: PRNGKeyArray):
        self.config = config

        keys = jax.random.split(key, 6)

        # Linear attention stack for contextualizing frames
        self.linear_attn_stack = LinearAttentionStack(config, key=keys[0])

        # Cross-attention: query from GRU hidden, kv from contextualized frames
        self.cross_attn = eqx.nn.MultiheadAttention(
            num_heads=config.num_cross_attn_heads,
            query_size=config.hidden_dim,
            key_size=config.hidden_dim,
            value_size=config.hidden_dim,
            output_size=config.hidden_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=keys[1],
        )

        # GRU cell
        # Input size: attractor_dim (prev attractor) + hidden_dim (cross-attn output)
        gru_input_size = config.attractor_dim + config.hidden_dim
        self.gru_cell = eqx.nn.GRUCell(
            input_size=gru_input_size,
            hidden_size=config.hidden_dim,
            use_bias=True,
            key=keys[2],
        )

        # Learned start token
        self.start_token = jax.random.normal(keys[3], (config.attractor_dim,)) * 0.02

        # Output heads
        self.attractor_head = eqx.nn.Linear(
            config.hidden_dim, config.attractor_dim, use_bias=True, key=keys[4]
        )
        self.confidence_head = eqx.nn.Linear(
            config.hidden_dim, 1, use_bias=True, key=keys[5]
        )

    def __call__(
        self,
        frame_embeddings: Float[Array, "num_frames input_dim"],
        *,
        key: PRNGKeyArray,
    ) -> Tuple[
        Float[Array, "max_attractors attractor_dim"],
        Float[Array, " max_attractors"],
        Float[Array, ""],  # scalar valid_count (Array for JAX compatibility)
    ]:
        """Generate speaker attractors from frame embeddings.

        Args:
            frame_embeddings: [N, input_dim] from frozen WavLeJEPA encoder
            key: PRNG key (unused currently, but kept for API consistency)

        Returns:
            attractors: [max_attractors, attractor_dim] padded array
            confidences: [max_attractors] padded with 0 for invalid entries
            valid_count: number of valid attractors (for downstream masking)
        """
        del key  # Currently unused, but may be needed for dropout etc.
        cfg = self.config
        max_K = cfg.max_attractors

        # Contextualize frames through linear attention stack (done once)
        contextualized = self.linear_attn_stack(frame_embeddings)  # [N, hidden_dim]

        # Initialize GRU hidden state from mean-pooled contextualized frames
        h_init = jnp.mean(contextualized, axis=0)  # [hidden_dim]

        # While loop state tuple:
        # (prev_attractor, h, attractors_buffer, confidences_buffer, step, continue_flag)
        init_state = (
            self.start_token,  # prev_attractor: [attractor_dim]
            h_init,  # h: GRU hidden [hidden_dim]
            jnp.zeros((max_K, cfg.attractor_dim)),  # attractors buffer
            jnp.zeros((max_K,)),  # confidences buffer
            jnp.array(0),  # step counter
            jnp.array(True),  # continue flag
        )

        def cond_fn(state):
            _, _, _, _, step, cont = state
            return cont & (step < max_K)

        def body_fn(state):
            prev_attractor, h, attractors, confs, step, _ = state

            # Cross-attend: GRU hidden queries contextualized frames
            # Query shape: [1, hidden_dim], KV shape: [N, hidden_dim]
            query = h[None, :]  # [1, hidden_dim]
            context = self.cross_attn(query, contextualized, contextualized)
            context = context[0]  # [hidden_dim] - squeeze query dim

            # GRU input: concatenate previous attractor with cross-attention context
            x = jnp.concatenate([prev_attractor, context])  # [attractor_dim + hidden_dim]

            # GRU step
            h_new = self.gru_cell(x, h)

            # Generate attractor and confidence
            a = self.attractor_head(h_new)  # [attractor_dim]
            c = jax.nn.sigmoid(self.confidence_head(h_new)[0])  # scalar

            # Update buffers
            attractors = attractors.at[step].set(a)
            confs = confs.at[step].set(c)

            # Continue if confidence > threshold
            cont = c > cfg.confidence_threshold

            return (a, h_new, attractors, confs, step + 1, cont)

        final = jax.lax.while_loop(cond_fn, body_fn, init_state)
        _, _, attractors, confidences, valid_count, _ = final

        return attractors, confidences, valid_count

    def generate_fixed(
        self,
        frame_embeddings: Float[Array, "num_frames input_dim"],
        num_attractors: int,
    ) -> Tuple[
        Float[Array, "num_attractors attractor_dim"],
        Float[Array, " num_attractors"],
    ]:
        """Generate a fixed number of attractors (no early stopping).

        Useful for training where we want gradients through all steps.

        Args:
            frame_embeddings: [N, input_dim] from frozen WavLeJEPA encoder
            num_attractors: Exact number of attractors to generate

        Returns:
            attractors: [num_attractors, attractor_dim]
            confidences: [num_attractors]
        """
        # Contextualize frames
        contextualized = self.linear_attn_stack(frame_embeddings)  # [N, hidden_dim]

        # Initialize GRU hidden state
        h_init = jnp.mean(contextualized, axis=0)  # [hidden_dim]

        def step_fn(carry, _):
            prev_attractor, h = carry

            # Cross-attend
            query = h[None, :]
            context = self.cross_attn(query, contextualized, contextualized)
            context = context[0]

            # GRU input and step
            x = jnp.concatenate([prev_attractor, context])
            h_new = self.gru_cell(x, h)

            # Generate outputs
            a = self.attractor_head(h_new)
            c = jax.nn.sigmoid(self.confidence_head(h_new)[0])

            return (a, h_new), (a, c)

        init_carry = (self.start_token, h_init)
        _, (attractors, confidences) = jax.lax.scan(
            step_fn, init_carry, None, length=num_attractors
        )

        return attractors, confidences
