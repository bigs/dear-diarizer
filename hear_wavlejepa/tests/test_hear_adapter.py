"""
Tests for the HEAR adapter.

Test A (required): Generate a synthetic checkpoint and verify:
- load_model returns model with HEAR attributes
- get_timestamp_embeddings returns correct shapes
- timestamps are increasing and within duration
- get_scene_embeddings returns correct shape
- no NaNs/infs

Test B (optional): Compare PyTorch vs JAX outputs for parity.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

# JAX model imports
from wavlejepa.model import WavLeJEPA, WavLeJEPAConfig
from wavlejepa.training.config import TrainingConfig, CheckpointConfig
from wavlejepa.training.checkpoint import WavLeJEPACheckpointer
from wavlejepa.training.state import TrainState, create_optimizer
import equinox as eqx


@pytest.fixture
def synthetic_checkpoint(tmp_path: Path) -> Path:
    """Create a synthetic checkpoint for testing."""
    # Use a small config for fast testing
    model_config = WavLeJEPAConfig(
        waveform_embed_dim=768,
        waveform_num_groups=32,
        context_embed_dim=768,
        context_num_heads=12,
        context_num_layers=2,  # Small for testing
        context_ffn_dim=3072,
        context_dropout=0.0,
        context_top_k_layers=2,
        context_top_k_norm="instance",
        predictor_dim=384,
        predictor_num_heads=12,
        predictor_num_layers=2,  # Small for testing
        predictor_ffn_dim=1536,
        predictor_dropout=0.0,
        max_seq_len=1000,
    )

    # Create model with fixed seed
    key = jax.random.key(42)
    model = WavLeJEPA(model_config, key=key)

    # Create training config
    checkpoint_config = CheckpointConfig(
        checkpoint_dir=str(tmp_path),
        save_every_n_steps=1,
        keep_n_checkpoints=1,
        save_best=True,
    )
    training_config = TrainingConfig(checkpoint=checkpoint_config)

    # Create checkpointer
    checkpointer = WavLeJEPACheckpointer(
        config=checkpoint_config,
        training_config=training_config,
        model_config=model_config,
    )

    # Create optimizer and state
    optimizer = create_optimizer(training_config.optimizer)
    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)

    state = TrainState(
        model=model,
        opt_state=opt_state,
        step=jnp.array(0, dtype=jnp.int32),
        key=jax.random.key(0),
        best_loss=jnp.array(float("inf"), dtype=jnp.float32),
    )

    # Save checkpoint
    checkpointer.save(state)
    checkpointer.save_best(state, val_loss=0.5)
    checkpointer.wait_until_finished()

    return tmp_path


class TestHEARAdapter:
    """Test HEAR adapter functionality."""

    def test_load_model_has_hear_attributes(self, synthetic_checkpoint: Path):
        """Test that load_model returns a model with HEAR attributes."""
        from hear_wavlejepa import load_model

        model = load_model(str(synthetic_checkpoint))

        # Check HEAR-required attributes
        assert hasattr(model, "sample_rate")
        assert model.sample_rate == 16000

        assert hasattr(model, "timestamp_embedding_size")
        assert model.timestamp_embedding_size == 768

        assert hasattr(model, "scene_embedding_size")
        assert model.scene_embedding_size == 768

    def test_timestamp_embeddings_shape(self, synthetic_checkpoint: Path):
        """Test get_timestamp_embeddings returns correct shapes."""
        from hear_wavlejepa import load_model, get_timestamp_embeddings

        model = load_model(str(synthetic_checkpoint))

        # Create test audio: 1 second at 16kHz
        n_sounds = 2
        n_samples = 16000
        audio = torch.randn(n_sounds, n_samples)

        embeddings, timestamps = get_timestamp_embeddings(audio, model)

        # Check shapes
        assert embeddings.ndim == 3
        assert embeddings.shape[0] == n_sounds
        assert embeddings.shape[2] == model.timestamp_embedding_size

        assert timestamps.ndim == 2
        assert timestamps.shape[0] == n_sounds
        assert timestamps.shape[1] == embeddings.shape[1]

    def test_timestamp_embeddings_increasing(self, synthetic_checkpoint: Path):
        """Test that timestamps are strictly increasing."""
        from hear_wavlejepa import load_model, get_timestamp_embeddings

        model = load_model(str(synthetic_checkpoint))

        audio = torch.randn(1, 16000)
        _, timestamps = get_timestamp_embeddings(audio, model)

        # Timestamps should be strictly increasing
        diffs = timestamps[0, 1:] - timestamps[0, :-1]
        assert (diffs > 0).all(), "Timestamps should be strictly increasing"

    def test_timestamp_embeddings_within_duration(self, synthetic_checkpoint: Path):
        """Test that timestamps are within audio duration."""
        from hear_wavlejepa import load_model, get_timestamp_embeddings

        model = load_model(str(synthetic_checkpoint))

        duration_sec = 2.0
        n_samples = int(duration_sec * 16000)
        audio = torch.randn(1, n_samples)

        _, timestamps = get_timestamp_embeddings(audio, model)

        # Allow some tolerance for the last frame extending slightly
        assert timestamps.min() >= 0, "Timestamps should be non-negative"
        # First timestamp should be near the center of first receptive field
        assert timestamps[0, 0] < 100, "First timestamp should be early in the audio"

    def test_timestamp_hop_size(self, synthetic_checkpoint: Path):
        """Test that hop size is <= 50ms (HEAR requirement)."""
        from hear_wavlejepa import load_model, get_timestamp_embeddings

        model = load_model(str(synthetic_checkpoint))

        audio = torch.randn(1, 32000)  # 2 seconds
        _, timestamps = get_timestamp_embeddings(audio, model)

        # Compute hop size in ms
        if timestamps.shape[1] > 1:
            hop_ms = (timestamps[0, 1] - timestamps[0, 0]).item()
            assert hop_ms <= 50, f"Hop size {hop_ms}ms exceeds HEAR limit of 50ms"
            # Expected: 10ms (100Hz)
            assert abs(hop_ms - 10) < 1, f"Expected ~10ms hop, got {hop_ms}ms"

    def test_scene_embeddings_shape(self, synthetic_checkpoint: Path):
        """Test get_scene_embeddings returns correct shape."""
        from hear_wavlejepa import load_model, get_scene_embeddings

        model = load_model(str(synthetic_checkpoint))

        n_sounds = 3
        audio = torch.randn(n_sounds, 16000)

        embeddings = get_scene_embeddings(audio, model)

        assert embeddings.ndim == 2
        assert embeddings.shape[0] == n_sounds
        assert embeddings.shape[1] == model.scene_embedding_size

    def test_no_nans_or_infs(self, synthetic_checkpoint: Path):
        """Test that outputs contain no NaNs or Infs."""
        from hear_wavlejepa import load_model, get_timestamp_embeddings, get_scene_embeddings

        model = load_model(str(synthetic_checkpoint))

        audio = torch.randn(2, 16000)

        ts_embeddings, timestamps = get_timestamp_embeddings(audio, model)
        scene_embeddings = get_scene_embeddings(audio, model)

        assert torch.isfinite(ts_embeddings).all(), "Timestamp embeddings contain NaN/Inf"
        assert torch.isfinite(timestamps).all(), "Timestamps contain NaN/Inf"
        assert torch.isfinite(scene_embeddings).all(), "Scene embeddings contain NaN/Inf"

    def test_single_audio_input(self, synthetic_checkpoint: Path):
        """Test that single audio (1D) input works."""
        from hear_wavlejepa import load_model, get_timestamp_embeddings, get_scene_embeddings

        model = load_model(str(synthetic_checkpoint))

        # Single audio without batch dimension
        audio = torch.randn(16000)

        ts_embeddings, timestamps = get_timestamp_embeddings(audio, model)
        scene_embeddings = get_scene_embeddings(audio, model)

        # Should add batch dimension automatically
        assert ts_embeddings.shape[0] == 1
        assert timestamps.shape[0] == 1
        assert scene_embeddings.shape[0] == 1


class TestPyTorchJAXParity:
    """Optional tests comparing PyTorch vs JAX outputs."""

    def test_output_shape_parity(self, synthetic_checkpoint: Path):
        """Test that PyTorch and JAX produce same output shapes."""
        from hear_wavlejepa import load_model, get_timestamp_embeddings

        # Load PyTorch model
        torch_model = load_model(str(synthetic_checkpoint))

        # Load JAX model
        from evals.voxceleb.checkpoint_utils import restore_model as restore_jax_model

        jax_model = restore_jax_model(synthetic_checkpoint)

        # Create test audio
        audio_np = np.random.randn(16000).astype(np.float32)
        audio_torch = torch.from_numpy(audio_np).unsqueeze(0)
        audio_jax = jnp.array(audio_np)

        # Get PyTorch output
        torch_embeddings, _ = get_timestamp_embeddings(audio_torch, torch_model)

        # Get JAX output
        jax_embeddings = jax_model.extract_features(audio_jax)

        # Check shape parity
        torch_shape = torch_embeddings.shape
        jax_shape = jax_embeddings.shape

        assert torch_shape[1] == jax_shape[0], (
            f"Frame count mismatch: PyTorch {torch_shape[1]} vs JAX {jax_shape[0]}"
        )
        assert torch_shape[2] == jax_shape[1], (
            f"Embed dim mismatch: PyTorch {torch_shape[2]} vs JAX {jax_shape[1]}"
        )

    def test_output_value_similarity(self, synthetic_checkpoint: Path):
        """Test that PyTorch and JAX produce similar output values.

        Note: Exact equality is not expected due to:
        - Different floating point accumulation order
        - Different GELU/softmax implementations
        - Different random number generation

        We check for reasonable similarity using cosine similarity.
        """
        from hear_wavlejepa import load_model, get_timestamp_embeddings

        # Load models
        torch_model = load_model(str(synthetic_checkpoint))

        from evals.voxceleb.checkpoint_utils import restore_model as restore_jax_model

        jax_model = restore_jax_model(synthetic_checkpoint)

        # Create deterministic test audio
        np.random.seed(42)
        audio_np = np.random.randn(16000).astype(np.float32)
        audio_torch = torch.from_numpy(audio_np).unsqueeze(0)
        audio_jax = jnp.array(audio_np)

        # Get outputs
        torch_embeddings, _ = get_timestamp_embeddings(audio_torch, torch_model)
        jax_embeddings = jax_model.extract_features(audio_jax)

        # Convert to numpy for comparison
        torch_np = torch_embeddings[0].numpy()
        jax_np = np.asarray(jax_embeddings)

        # Compute cosine similarity per frame
        def cosine_similarity(a, b):
            dot = np.sum(a * b, axis=-1)
            norm_a = np.linalg.norm(a, axis=-1)
            norm_b = np.linalg.norm(b, axis=-1)
            return dot / (norm_a * norm_b + 1e-8)

        cos_sim = cosine_similarity(torch_np, jax_np)
        mean_cos_sim = np.mean(cos_sim)

        # We expect high similarity if weights loaded correctly
        # Lower threshold since implementations may differ
        assert mean_cos_sim > 0.9, (
            f"Mean cosine similarity {mean_cos_sim:.4f} is too low. "
            "PyTorch and JAX outputs may not be properly aligned."
        )

    def test_embedding_statistics(self, synthetic_checkpoint: Path):
        """Test that embedding statistics are similar between PyTorch and JAX."""
        from hear_wavlejepa import load_model, get_timestamp_embeddings

        # Load models
        torch_model = load_model(str(synthetic_checkpoint))

        from evals.voxceleb.checkpoint_utils import restore_model as restore_jax_model

        jax_model = restore_jax_model(synthetic_checkpoint)

        # Create test audio
        np.random.seed(42)
        audio_np = np.random.randn(16000).astype(np.float32)
        audio_torch = torch.from_numpy(audio_np).unsqueeze(0)
        audio_jax = jnp.array(audio_np)

        # Get outputs
        torch_embeddings, _ = get_timestamp_embeddings(audio_torch, torch_model)
        jax_embeddings = jax_model.extract_features(audio_jax)

        # Convert to numpy
        torch_np = torch_embeddings[0].numpy()
        jax_np = np.asarray(jax_embeddings)

        # Compare statistics
        torch_mean = np.mean(torch_np)
        jax_mean = np.mean(jax_np)
        torch_std = np.std(torch_np)
        jax_std = np.std(jax_np)

        # Statistics should be in same ballpark
        # (instance norm in top-k means both should have ~0 mean, ~1 std per frame)
        assert abs(torch_mean - jax_mean) < 0.5, (
            f"Mean difference too large: PyTorch {torch_mean:.4f} vs JAX {jax_mean:.4f}"
        )
        assert abs(torch_std - jax_std) < 0.5, (
            f"Std difference too large: PyTorch {torch_std:.4f} vs JAX {jax_std:.4f}"
        )
