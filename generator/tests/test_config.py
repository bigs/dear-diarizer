"""Tests for GeneratorConfig and SSM backend selection."""

import pytest
import jax
import jax.numpy as jnp

from generator.config import GeneratorConfig, Mamba2Config
from generator.mamba import LinearAttentionStack, Mamba2Block
from jax_gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetBlock


class TestConfigValidation:
    """Tests for config validation."""

    def test_mamba2_config_valid(self):
        """Mamba2-only config should be valid."""
        config = GeneratorConfig(mamba2=Mamba2Config())
        assert config.ssm_type == "mamba2"
        assert config.mamba2 is not None
        assert config.deltanet is None

    def test_deltanet_config_valid(self):
        """DeltaNet-only config should be valid."""
        deltanet_cfg = GatedDeltaNetConfig(hidden_size=768)
        config = GeneratorConfig(deltanet=deltanet_cfg)
        assert config.ssm_type == "deltanet"
        assert config.deltanet is not None
        assert config.mamba2 is None

    def test_both_configs_error(self):
        """Both mamba2 and deltanet configs should raise error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            GeneratorConfig(
                mamba2=Mamba2Config(),
                deltanet=GatedDeltaNetConfig(hidden_size=768),
            )

    def test_neither_config_error(self):
        """Neither mamba2 nor deltanet config should raise error."""
        with pytest.raises(ValueError, match="Must specify either"):
            GeneratorConfig()

    def test_deltanet_hidden_size_mismatch_error(self):
        """DeltaNet hidden_size must match hidden_dim."""
        with pytest.raises(ValueError, match="must match"):
            GeneratorConfig(
                hidden_dim=768,
                deltanet=GatedDeltaNetConfig(hidden_size=512),  # Mismatch!
            )

    def test_deltanet_hidden_size_match_valid(self):
        """DeltaNet hidden_size matching hidden_dim should be valid."""
        config = GeneratorConfig(
            hidden_dim=512,
            deltanet=GatedDeltaNetConfig(hidden_size=512),
        )
        assert config.deltanet.hidden_size == config.hidden_dim


class TestMamba2Config:
    """Tests for Mamba2Config."""

    def test_intermediate_size(self):
        """Test intermediate_size computation."""
        cfg = Mamba2Config(expand=2)
        assert cfg.intermediate_size(768) == 1536
        assert cfg.intermediate_size(512) == 1024

    def test_num_heads(self):
        """Test num_heads computation."""
        cfg = Mamba2Config(head_dim=64, expand=2)
        # intermediate_size = 2 * 768 = 1536
        # num_heads = 1536 / 64 = 24
        assert cfg.num_heads(768) == 24


class TestLinearAttentionStackFactory:
    """Tests for LinearAttentionStack SSM backend selection."""

    def test_mamba2_backend_selection(self):
        """LinearAttentionStack should use Mamba2Block with mamba2 config."""
        config = GeneratorConfig(mamba2=Mamba2Config())
        key = jax.random.PRNGKey(0)
        stack = LinearAttentionStack(config, key=key)

        # Check that layers are Mamba2Block instances
        assert len(stack.layers) == config.num_layers
        assert all(isinstance(layer, Mamba2Block) for layer in stack.layers)

    def test_deltanet_backend_selection(self):
        """LinearAttentionStack should use GatedDeltaNetBlock with deltanet config."""
        config = GeneratorConfig(
            hidden_dim=768,
            deltanet=GatedDeltaNetConfig(hidden_size=768, num_layers=4),
        )
        key = jax.random.PRNGKey(0)
        stack = LinearAttentionStack(config, key=key)

        # Check that layers are GatedDeltaNetBlock instances
        assert len(stack.layers) == config.num_layers
        assert all(isinstance(layer, GatedDeltaNetBlock) for layer in stack.layers)


class TestBackendOutputs:
    """Tests that both backends produce valid outputs."""

    @pytest.fixture
    def sample_input(self):
        """Generate sample input for testing."""
        key = jax.random.PRNGKey(42)
        seq_len = 100
        input_dim = 768
        return jax.random.normal(key, (seq_len, input_dim))

    def test_mamba2_forward_pass(self, sample_input):
        """Mamba2 backend should produce valid output."""
        config = GeneratorConfig(mamba2=Mamba2Config())
        key = jax.random.PRNGKey(0)
        stack = LinearAttentionStack(config, key=key)

        output = stack(sample_input)

        # Check output shape
        assert output.shape == (sample_input.shape[0], config.hidden_dim)
        # Check no NaNs
        assert not jnp.any(jnp.isnan(output))

    def test_deltanet_forward_pass(self, sample_input):
        """DeltaNet backend should produce valid output."""
        config = GeneratorConfig(
            hidden_dim=768,
            deltanet=GatedDeltaNetConfig(hidden_size=768),
        )
        key = jax.random.PRNGKey(0)
        stack = LinearAttentionStack(config, key=key)

        output = stack(sample_input)

        # Check output shape
        assert output.shape == (sample_input.shape[0], config.hidden_dim)
        # Check no NaNs
        assert not jnp.any(jnp.isnan(output))

    def test_both_backends_same_output_shape(self, sample_input):
        """Both backends should produce same output shape."""
        mamba2_config = GeneratorConfig(mamba2=Mamba2Config())
        deltanet_config = GeneratorConfig(
            hidden_dim=768,
            deltanet=GatedDeltaNetConfig(hidden_size=768),
        )

        key = jax.random.PRNGKey(0)
        mamba2_stack = LinearAttentionStack(mamba2_config, key=key)
        deltanet_stack = LinearAttentionStack(deltanet_config, key=key)

        mamba2_out = mamba2_stack(sample_input)
        deltanet_out = deltanet_stack(sample_input)

        assert mamba2_out.shape == deltanet_out.shape


class TestGradientFlow:
    """Tests that gradients flow through both backends."""

    def test_mamba2_gradient_flow(self):
        """Gradients should flow through Mamba2 backend."""
        config = GeneratorConfig(mamba2=Mamba2Config())
        key = jax.random.PRNGKey(0)
        stack = LinearAttentionStack(config, key=key)

        x = jax.random.normal(jax.random.PRNGKey(42), (50, 768))

        def loss_fn(model, x):
            return jnp.mean(model(x) ** 2)

        grads = jax.grad(loss_fn)(stack, x)

        # Check that gradients exist and are not all zero
        grad_norm = sum(
            jnp.sum(jnp.abs(g)) for g in jax.tree.leaves(grads) if g is not None
        )
        assert grad_norm > 0

    def test_deltanet_gradient_flow(self):
        """Gradients should flow through DeltaNet backend."""
        config = GeneratorConfig(
            hidden_dim=768,
            deltanet=GatedDeltaNetConfig(hidden_size=768),
        )
        key = jax.random.PRNGKey(0)
        stack = LinearAttentionStack(config, key=key)

        x = jax.random.normal(jax.random.PRNGKey(42), (50, 768))

        def loss_fn(model, x):
            return jnp.mean(model(x) ** 2)

        grads = jax.grad(loss_fn)(stack, x)

        # Check that gradients exist and are not all zero
        grad_norm = sum(
            jnp.sum(jnp.abs(g)) for g in jax.tree.leaves(grads) if g is not None
        )
        assert grad_norm > 0
