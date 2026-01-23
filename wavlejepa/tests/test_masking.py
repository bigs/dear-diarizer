"""Tests for masking utilities."""

import jax
import jax.numpy as jnp
import pytest

from wavlejepa import (
    MaskingConfig,
    sample_context_mask,
    sample_target_mask,
    sample_masks,
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def default_config():
    return MaskingConfig()


class TestMaskingConfig:
    def test_default_values(self):
        config = MaskingConfig()
        assert config.context_ratio == 0.35
        assert config.target_ratio == 0.23
        assert config.context_block_length == 10
        assert config.target_block_length == 10
        assert config.min_context_ratio == 0.10
        assert config.max_retries == 10

    def test_custom_values(self):
        config = MaskingConfig(
            context_ratio=0.40,
            target_ratio=0.20,
            context_block_length=5,
            target_block_length=15,
        )
        assert config.context_ratio == 0.40
        assert config.target_ratio == 0.20
        assert config.context_block_length == 5
        assert config.target_block_length == 15


class TestSampleContextMask:
    def test_output_shape(self, key, default_config):
        seq_len = 100
        mask = sample_context_mask(seq_len, default_config, key)
        assert mask.shape == (seq_len,)
        assert mask.dtype == jnp.bool_

    def test_ratio_within_tolerance(self, key, default_config):
        seq_len = 200
        mask = sample_context_mask(seq_len, default_config, key)
        actual_ratio = jnp.sum(mask) / seq_len
        # Allow 20% tolerance due to block overlaps
        assert 0.20 <= actual_ratio <= 0.50

    def test_deterministic_with_same_key(self, default_config):
        seq_len = 100
        key1 = jax.random.PRNGKey(123)
        key2 = jax.random.PRNGKey(123)
        mask1 = sample_context_mask(seq_len, default_config, key1)
        mask2 = sample_context_mask(seq_len, default_config, key2)
        assert jnp.array_equal(mask1, mask2)

    def test_different_keys_produce_different_masks(self, default_config):
        seq_len = 100
        key1 = jax.random.PRNGKey(1)
        key2 = jax.random.PRNGKey(2)
        mask1 = sample_context_mask(seq_len, default_config, key1)
        mask2 = sample_context_mask(seq_len, default_config, key2)
        assert not jnp.array_equal(mask1, mask2)


class TestSampleTargetMask:
    def test_output_shape(self, key, default_config):
        seq_len = 100
        context_mask = sample_context_mask(seq_len, default_config, key)
        key2 = jax.random.PRNGKey(99)
        target_mask = sample_target_mask(seq_len, context_mask, default_config, key2)
        assert target_mask.shape == (seq_len,)
        assert target_mask.dtype == jnp.bool_

    def test_disjoint_from_context(self, key, default_config):
        """Target mask must not overlap with context mask."""
        seq_len = 200
        context_mask = sample_context_mask(seq_len, default_config, key)
        key2 = jax.random.PRNGKey(99)
        target_mask = sample_target_mask(seq_len, context_mask, default_config, key2)
        overlap = jnp.sum(context_mask & target_mask)
        assert overlap == 0

    def test_ratio_within_tolerance(self, key, default_config):
        seq_len = 200
        context_mask = sample_context_mask(seq_len, default_config, key)
        key2 = jax.random.PRNGKey(99)
        target_mask = sample_target_mask(seq_len, context_mask, default_config, key2)
        actual_ratio = jnp.sum(target_mask) / seq_len
        # Allow tolerance due to block overlaps and context exclusion
        assert 0.05 <= actual_ratio <= 0.35


class TestSampleMasks:
    def test_output_shapes(self, key, default_config):
        seq_len = 100
        context_mask, target_mask = sample_masks(seq_len, default_config, key)
        assert context_mask.shape == (seq_len,)
        assert target_mask.shape == (seq_len,)

    def test_masks_are_disjoint(self, key, default_config):
        seq_len = 200
        context_mask, target_mask = sample_masks(seq_len, default_config, key)
        overlap = jnp.sum(context_mask & target_mask)
        assert overlap == 0

    def test_ratios_approximate_targets(self, key, default_config):
        """Verify ratios are close to configured targets."""
        seq_len = 500
        context_mask, target_mask = sample_masks(seq_len, default_config, key)

        context_ratio = jnp.sum(context_mask) / seq_len
        target_ratio = jnp.sum(target_mask) / seq_len

        # Context should be ~35% (allow 15% tolerance for block effects)
        assert 0.20 <= context_ratio <= 0.50
        # Target should be ~23% (more variance due to context exclusion)
        assert 0.05 <= target_ratio <= 0.35

    def test_retry_triggers_on_low_coverage(self, key):
        """Verify retry mechanism activates when initial coverage is low."""
        # Use very small context ratio to occasionally trigger low coverage
        config = MaskingConfig(
            context_ratio=0.05,  # Very low target
            min_context_ratio=0.10,  # But require at least 10%
            max_retries=10,
        )
        seq_len = 100

        # Run multiple times to check retry logic works
        for i in range(10):
            key_i = jax.random.PRNGKey(i)
            context_mask, _ = sample_masks(seq_len, config, key_i)
            # With retries, we should still get some coverage
            # (may not always hit min due to max_retries limit)
            assert jnp.sum(context_mask) > 0

    def test_min_context_ratio_enforced(self, key):
        """Verify minimum context ratio is respected when possible."""
        config = MaskingConfig(
            context_ratio=0.35,
            min_context_ratio=0.10,
            max_retries=10,
        )
        seq_len = 200
        context_mask, _ = sample_masks(seq_len, config, key)
        context_ratio = jnp.sum(context_mask) / seq_len
        assert context_ratio >= config.min_context_ratio

    def test_jit_compatible(self, key, default_config):
        """Verify sample_masks works under JIT compilation."""
        seq_len = 100

        @jax.jit
        def jitted_sample(key):
            return sample_masks(seq_len, default_config, key)

        context_mask, target_mask = jitted_sample(key)
        assert context_mask.shape == (seq_len,)
        assert target_mask.shape == (seq_len,)
        # Verify no overlap after JIT
        assert jnp.sum(context_mask & target_mask) == 0

    def test_jit_with_retry(self, key):
        """Verify JIT compatibility with retry mechanism (while_loop)."""
        config = MaskingConfig(
            context_ratio=0.15,
            min_context_ratio=0.10,
            max_retries=5,
        )
        seq_len = 100

        @jax.jit
        def jitted_sample(key):
            return sample_masks(seq_len, config, key)

        # Should compile and run without errors
        context_mask, target_mask = jitted_sample(key)
        assert context_mask.shape == (seq_len,)
        assert target_mask.shape == (seq_len,)


class TestGradients:
    def test_masks_do_not_require_gradients(self, key, default_config):
        """Verify masking functions work in gradient context."""
        seq_len = 50

        def loss_fn(x, key):
            context_mask, target_mask = sample_masks(seq_len, default_config, key)
            # Use masks to select values
            context_sum = jnp.sum(jnp.where(context_mask, x, 0.0))
            target_sum = jnp.sum(jnp.where(target_mask, x, 0.0))
            return context_sum + target_sum

        x = jax.random.normal(key, (seq_len,))
        # Should not raise - masks are not differentiable but that's expected
        grad = jax.grad(loss_fn)(x, key)
        assert grad.shape == (seq_len,)
        assert jnp.all(jnp.isfinite(grad))
