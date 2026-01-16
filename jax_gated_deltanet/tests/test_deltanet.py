"""Tests for Gated DeltaNet implementation."""

import jax
import jax.numpy as jnp
import pytest

from jax_gated_deltanet import (
    GatedDeltaNetConfig,
    GatedDeltaNetLayer,
    GatedDeltaNetBlock,
    GatedDeltaNetStack,
    gated_delta_rule,
    gated_delta_rule_recurrent,
    l2_normalize,
    RMSNorm,
    FusedRMSNormGated,
    ShortConvolution,
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def small_config():
    """Small config for fast testing."""
    return GatedDeltaNetConfig(
        hidden_size=128,
        num_heads=2,
        num_v_heads_=4,  # GVA
        head_k_dim=32,
        expand_v=2.0,
        num_layers=2,
    )


class TestConfig:
    def test_default_config(self):
        config = GatedDeltaNetConfig()
        assert config.hidden_size == 2048
        assert config.num_heads == 6
        assert config.num_v_heads == 6  # defaults to num_heads
        assert config.head_k_dim == 256
        assert config.head_v_dim == 512  # 256 * 2.0
        assert config.key_dim == 1536  # 6 * 256
        assert config.value_dim == 3072  # 6 * 512

    def test_gva_config(self):
        config = GatedDeltaNetConfig(num_heads=4, num_v_heads_=8)
        assert config.num_v_heads == 8
        assert config.gva_groups == 2

    def test_invalid_gva_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            GatedDeltaNetConfig(num_heads=4, num_v_heads_=7)

    def test_invalid_expand_v_raises(self):
        with pytest.raises(ValueError, match="non-integer"):
            GatedDeltaNetConfig(head_k_dim=100, expand_v=1.5)  # 150 is ok, but test edge


class TestNorm:
    def test_rmsnorm_shape(self, key):
        norm = RMSNorm(64, key=key)
        x = jax.random.normal(key, (64,))
        y = norm(x)
        assert y.shape == x.shape

    def test_rmsnorm_normalized(self, key):
        norm = RMSNorm(64, key=key)
        x = jax.random.normal(key, (64,)) * 10  # large values
        y = norm(x)
        # RMS should be close to 1
        rms = jnp.sqrt(jnp.mean(y**2))
        assert jnp.abs(rms - 1.0) < 0.1

    def test_fused_rmsnorm_gated_shape(self, key):
        norm = FusedRMSNormGated(64, key=key)
        x = jax.random.normal(key, (64,))
        gate = jax.random.normal(key, (64,))
        y = norm(x, gate)
        assert y.shape == x.shape

    def test_fused_rmsnorm_gated_zero_gate(self, key):
        norm = FusedRMSNormGated(64, key=key)
        x = jax.random.normal(key, (64,))
        gate = jnp.full((64,), -100.0)  # sigmoid(-100) ~ 0
        y = norm(x, gate)
        assert jnp.allclose(y, 0, atol=1e-5)


class TestConv:
    def test_short_conv_shape(self, key):
        conv = ShortConvolution(64, kernel_size=4, key=key)
        x = jax.random.normal(key, (16, 64))
        y, _ = conv(x)
        assert y.shape == x.shape

    def test_short_conv_causal(self, key):
        """Verify causality: output[t] shouldn't depend on input[t+1:]"""
        conv = ShortConvolution(64, kernel_size=4, key=key)
        x = jax.random.normal(key, (16, 64))

        # Full sequence output
        y_full, _ = conv(x)

        # Output for first 8 positions should be same with truncated input
        y_partial, _ = conv(x[:8])

        assert jnp.allclose(y_full[:8], y_partial, atol=1e-5)

    def test_short_conv_with_cache(self, key):
        conv = ShortConvolution(64, kernel_size=4, key=key)
        x = jax.random.normal(key, (16, 64))

        # Process in chunks with caching
        y1, cache = conv(x[:8], output_cache=True)
        y2, _ = conv(x[8:], cache=cache)
        y_chunked = jnp.concatenate([y1, y2], axis=0)

        # Should match full processing
        y_full, _ = conv(x)

        assert jnp.allclose(y_chunked, y_full, atol=1e-5)


class TestDeltaRule:
    def test_gated_delta_rule_shape(self, key):
        batch, seq, heads, head_k, head_v = 2, 16, 4, 32, 64

        q = l2_normalize(jax.random.normal(key, (batch, seq, heads, head_k)))
        k = l2_normalize(jax.random.normal(key, (batch, seq, heads, head_k)))
        v = jax.random.normal(key, (batch, seq, heads, head_v))
        g = jax.nn.log_sigmoid(jax.random.normal(key, (batch, seq, heads)))
        beta = jax.nn.sigmoid(jax.random.normal(key, (batch, seq, heads)))

        output, state = gated_delta_rule(q, k, v, g, beta, output_final_state=True)

        assert output.shape == (batch, seq, heads, head_v)
        assert state.shape == (batch, heads, head_k, head_v)

    def test_gated_delta_rule_with_initial_state(self, key):
        batch, seq, heads, head_k, head_v = 2, 16, 4, 32, 64

        q = l2_normalize(jax.random.normal(key, (batch, seq, heads, head_k)))
        k = l2_normalize(jax.random.normal(key, (batch, seq, heads, head_k)))
        v = jax.random.normal(key, (batch, seq, heads, head_v))
        g = jax.nn.log_sigmoid(jax.random.normal(key, (batch, seq, heads)))
        beta = jax.nn.sigmoid(jax.random.normal(key, (batch, seq, heads)))
        h0 = jax.random.normal(key, (batch, heads, head_k, head_v)) * 0.1

        output, _ = gated_delta_rule(q, k, v, g, beta, initial_state=h0)

        assert output.shape == (batch, seq, heads, head_v)

    def test_recurrent_matches_scan(self, key):
        """Verify single-step recurrent matches scan implementation."""
        batch, heads, head_k, head_v = 2, 4, 32, 64

        # Single timestep
        q = l2_normalize(jax.random.normal(key, (batch, 1, heads, head_k)))
        k = l2_normalize(jax.random.normal(key, (batch, 1, heads, head_k)))
        v = jax.random.normal(key, (batch, 1, heads, head_v))
        g = jax.nn.log_sigmoid(jax.random.normal(key, (batch, 1, heads)))
        beta = jax.nn.sigmoid(jax.random.normal(key, (batch, 1, heads)))
        h0 = jax.random.normal(key, (batch, heads, head_k, head_v)) * 0.1

        # Using scan
        out_scan, state_scan = gated_delta_rule(
            q, k, v, g, beta, initial_state=h0, output_final_state=True
        )

        # Using recurrent
        out_rec, state_rec = gated_delta_rule_recurrent(q, k, v, g, beta, h0)

        assert jnp.allclose(out_scan, out_rec, atol=1e-5)
        assert jnp.allclose(state_scan, state_rec, atol=1e-5)


class TestLayer:
    def test_layer_shape(self, key, small_config):
        layer = GatedDeltaNetLayer(small_config, key=key)
        x = jax.random.normal(key, (16, small_config.hidden_size))

        output, state = layer(x, return_final_state=True)

        assert output.shape == x.shape
        assert state.shape == (
            small_config.num_v_heads,
            small_config.head_k_dim,
            small_config.head_v_dim,
        )

    def test_layer_no_gate(self, key):
        config = GatedDeltaNetConfig(
            hidden_size=128,
            num_heads=2,
            head_k_dim=32,
            use_gate=False,
        )
        layer = GatedDeltaNetLayer(config, key=key)
        x = jax.random.normal(key, (16, config.hidden_size))

        output, _ = layer(x)
        assert output.shape == x.shape

    def test_layer_no_conv(self, key):
        config = GatedDeltaNetConfig(
            hidden_size=128,
            num_heads=2,
            head_k_dim=32,
            use_short_conv=False,
        )
        layer = GatedDeltaNetLayer(config, key=key)
        x = jax.random.normal(key, (16, config.hidden_size))

        output, _ = layer(x)
        assert output.shape == x.shape


class TestBlock:
    def test_block_residual(self, key, small_config):
        block = GatedDeltaNetBlock(small_config, key=key)
        x = jax.random.normal(key, (16, small_config.hidden_size))

        output, _ = block(x)

        # Output should be different from input (residual + transformed)
        assert not jnp.allclose(output, x)
        assert output.shape == x.shape


class TestStack:
    def test_stack_shape(self, key, small_config):
        stack = GatedDeltaNetStack(small_config, key=key)
        x = jax.random.normal(key, (16, small_config.hidden_size))

        output = stack(x)

        assert output.shape == x.shape

    def test_stack_with_input_proj(self, key, small_config):
        stack = GatedDeltaNetStack(small_config, input_dim=64, key=key)
        x = jax.random.normal(key, (16, 64))

        output = stack(x)

        assert output.shape == (16, small_config.hidden_size)


class TestGradients:
    def test_layer_gradient_flow(self, key, small_config):
        """Verify gradients flow through the layer."""
        layer = GatedDeltaNetLayer(small_config, key=key)
        x = jax.random.normal(key, (8, small_config.hidden_size))

        def loss_fn(layer, x):
            output, _ = layer(x)
            return jnp.mean(output**2)

        grads = jax.grad(loss_fn)(layer, x)

        # Check that gradients exist and are finite
        q_grad = grads.q_proj.weight
        assert q_grad is not None
        assert jnp.all(jnp.isfinite(q_grad))

    def test_stack_gradient_flow(self, key, small_config):
        """Verify gradients flow through the full stack."""
        stack = GatedDeltaNetStack(small_config, key=key)
        x = jax.random.normal(key, (8, small_config.hidden_size))

        def loss_fn(stack, x):
            output = stack(x)
            return jnp.mean(output**2)

        grads = jax.grad(loss_fn)(stack, x)

        # Check gradients in first and last layer
        first_layer_grad = grads.layers[0].deltanet.q_proj.weight
        last_layer_grad = grads.layers[-1].deltanet.q_proj.weight

        assert jnp.all(jnp.isfinite(first_layer_grad))
        assert jnp.all(jnp.isfinite(last_layer_grad))


class TestChunkwise:
    """Tests for chunkwise parallel implementation."""

    def test_chunk_output_matches_naive(self, key):
        """Verify chunkwise output matches naive scan."""
        from jax_gated_deltanet.ops.chunk import gated_delta_rule_chunk

        batch, seq, heads, head_k, head_v = 2, 64, 4, 32, 64

        q = l2_normalize(jax.random.normal(key, (batch, seq, heads, head_k)))
        k = l2_normalize(jax.random.normal(key, (batch, seq, heads, head_k)))
        v = jax.random.normal(key, (batch, seq, heads, head_v))
        g = jax.nn.log_sigmoid(jax.random.normal(key, (batch, seq, heads)))
        beta = jax.nn.sigmoid(jax.random.normal(key, (batch, seq, heads)))

        output_naive, _ = gated_delta_rule(q, k, v, g, beta)
        output_chunk, _ = gated_delta_rule_chunk(q, k, v, g, beta, chunk_size=16)

        assert jnp.allclose(output_naive, output_chunk, atol=1e-5)

    def test_chunk_state_matches_naive(self, key):
        """Verify chunkwise final state matches naive scan."""
        from jax_gated_deltanet.ops.chunk import gated_delta_rule_chunk

        batch, seq, heads, head_k, head_v = 2, 64, 4, 32, 64

        q = l2_normalize(jax.random.normal(key, (batch, seq, heads, head_k)))
        k = l2_normalize(jax.random.normal(key, (batch, seq, heads, head_k)))
        v = jax.random.normal(key, (batch, seq, heads, head_v))
        g = jax.nn.log_sigmoid(jax.random.normal(key, (batch, seq, heads)))
        beta = jax.nn.sigmoid(jax.random.normal(key, (batch, seq, heads)))

        _, state_naive = gated_delta_rule(q, k, v, g, beta, output_final_state=True)
        _, state_chunk = gated_delta_rule_chunk(
            q, k, v, g, beta, output_final_state=True, chunk_size=16
        )

        assert jnp.allclose(state_naive, state_chunk, atol=1e-5)

    def test_chunk_with_initial_state(self, key):
        """Verify chunkwise works with initial state."""
        from jax_gated_deltanet.ops.chunk import gated_delta_rule_chunk

        batch, seq, heads, head_k, head_v = 2, 64, 4, 32, 64

        q = l2_normalize(jax.random.normal(key, (batch, seq, heads, head_k)))
        k = l2_normalize(jax.random.normal(key, (batch, seq, heads, head_k)))
        v = jax.random.normal(key, (batch, seq, heads, head_v))
        g = jax.nn.log_sigmoid(jax.random.normal(key, (batch, seq, heads)))
        beta = jax.nn.sigmoid(jax.random.normal(key, (batch, seq, heads)))
        h0 = jax.random.normal(key, (batch, heads, head_k, head_v)) * 0.1

        output_naive, state_naive = gated_delta_rule(
            q, k, v, g, beta, initial_state=h0, output_final_state=True
        )
        output_chunk, state_chunk = gated_delta_rule_chunk(
            q, k, v, g, beta, initial_state=h0, output_final_state=True, chunk_size=16
        )

        assert jnp.allclose(output_naive, output_chunk, atol=1e-5)
        assert jnp.allclose(state_naive, state_chunk, atol=1e-5)

    def test_chunk_non_divisible_length(self, key):
        """Verify chunkwise handles non-divisible sequence lengths."""
        from jax_gated_deltanet.ops.chunk import gated_delta_rule_chunk

        batch, seq, heads, head_k, head_v = 2, 50, 4, 32, 64  # 50 not divisible by 16

        q = l2_normalize(jax.random.normal(key, (batch, seq, heads, head_k)))
        k = l2_normalize(jax.random.normal(key, (batch, seq, heads, head_k)))
        v = jax.random.normal(key, (batch, seq, heads, head_v))
        g = jax.nn.log_sigmoid(jax.random.normal(key, (batch, seq, heads)))
        beta = jax.nn.sigmoid(jax.random.normal(key, (batch, seq, heads)))

        output_naive, _ = gated_delta_rule(q, k, v, g, beta)
        output_chunk, _ = gated_delta_rule_chunk(q, k, v, g, beta, chunk_size=16)

        assert output_chunk.shape == output_naive.shape
        assert jnp.allclose(output_naive, output_chunk, atol=1e-5)

    def test_chunk_gradient_flow(self, key):
        """Verify gradients flow through chunkwise implementation."""
        from jax_gated_deltanet.ops.chunk import gated_delta_rule_chunk

        batch, seq, heads, head_k, head_v = 2, 32, 4, 32, 64

        q = l2_normalize(jax.random.normal(key, (batch, seq, heads, head_k)))
        k = l2_normalize(jax.random.normal(key, (batch, seq, heads, head_k)))
        v = jax.random.normal(key, (batch, seq, heads, head_v))
        g = jax.nn.log_sigmoid(jax.random.normal(key, (batch, seq, heads)))
        beta = jax.nn.sigmoid(jax.random.normal(key, (batch, seq, heads)))

        def loss_fn(q, k, v, g, beta):
            output, _ = gated_delta_rule_chunk(q, k, v, g, beta, chunk_size=16)
            return jnp.mean(output**2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)

        for i, grad in enumerate(grads):
            assert jnp.all(jnp.isfinite(grad)), f"Gradient {i} has non-finite values"
