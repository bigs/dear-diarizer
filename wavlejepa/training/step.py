"""
Training and evaluation step functions for WavLeJEPA.

Supports both single-GPU and multi-GPU data parallelism.
Includes mixed precision (bfloat16) support for H100/GB10.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import jax.sharding as jshard
import equinox as eqx
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from ..losses import compute_loss
from .state import TrainState
from .config import LossConfig, PrecisionConfig
from .precision import get_compute_dtype, cast_to_float32, cast_model_to_compute


# Type alias for sharding configuration
ShardingConfig = Optional[tuple[jshard.Mesh, jshard.NamedSharding, jshard.NamedSharding]]


def init_sharding() -> ShardingConfig:
    """
    Initialize sharding for multi-GPU data parallelism.

    Returns None for single GPU (no sharding needed).
    For multi-GPU, returns (mesh, data_sharding, model_sharding).

    Data is sharded across batch dimension.
    Model is replicated on all devices.
    """
    num_devices = len(jax.devices())

    if num_devices == 1:
        return None

    mesh = jax.make_mesh((num_devices,), ("batch",))
    data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
    model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())

    return mesh, data_sharding, model_sharding


def shard_batch(batch: Array, sharding: ShardingConfig) -> Array:
    """Shard a batch across devices (no-op for single GPU)."""
    if sharding is None:
        return batch
    _, data_sharding, _ = sharding
    return jax.device_put(batch, data_sharding)


def shard_state(state: TrainState, sharding: ShardingConfig) -> TrainState:
    """Replicate state across devices (no-op for single GPU)."""
    if sharding is None:
        return state
    _, _, model_sharding = sharding
    return eqx.filter_shard(state, model_sharding)


def make_train_step(
    optimizer: optax.GradientTransformation,
    loss_config: LossConfig,
    sharding: ShardingConfig = None,
    precision_config: Optional[PrecisionConfig] = None,
):
    """
    Create a JIT-compiled training step function.

    Args:
        optimizer: Optax optimizer
        loss_config: Loss configuration
        sharding: Sharding config from init_sharding() (None for single GPU)
        precision_config: Mixed precision config (None defaults to bfloat16)

    Returns:
        train_step function: (state, batch) -> (new_state, metrics)
    """
    if precision_config is None:
        precision_config = PrecisionConfig()

    compute_dtype = get_compute_dtype(precision_config)
    loss_in_fp32 = precision_config.loss_in_float32

    @eqx.filter_jit(donate="all")
    def train_step(
        state: TrainState,
        batch: Float[Array, "batch time"],
    ) -> tuple[TrainState, dict[str, Array]]:
        """
        Single training step.

        Args:
            state: Current training state
            batch: Audio waveforms [batch, time] at 16kHz

        Returns:
            Tuple of (updated state, metrics dict)
        """
        # Enforce sharding constraints for multi-GPU
        if sharding is not None:
            _, data_sharding, model_sharding = sharding
            state = eqx.filter_shard(state, model_sharding)
            batch = eqx.filter_shard(batch, data_sharding)

        # Cast to compute dtype for mixed precision
        batch = batch.astype(compute_dtype)

        # Split PRNG key for this step
        key, loss_key, forward_key = jax.random.split(state.key, 3)

        # Per-sample keys for vmap (each sample needs different masking)
        batch_size = batch.shape[0]
        forward_keys = jax.random.split(forward_key, batch_size)

        def loss_fn(model):
            """Compute loss over batch with vmap."""
            # Cast model to compute dtype for mixed precision forward pass
            model_compute = cast_model_to_compute(model, compute_dtype)

            def single_forward(waveform, k):
                return model_compute.forward_train(waveform, key=k)

            # vmap forward_train over batch dimension
            outputs = jax.vmap(single_forward)(batch, forward_keys)

            # Cast outputs to float32 for stable loss computation
            if loss_in_fp32:
                outputs = jax.tree.map(
                    lambda x: cast_to_float32(x) if x.dtype == compute_dtype else x,
                    outputs,
                )

            # Compute loss
            # outputs has shape [batch, ...] for each field
            total_loss, metrics = compute_loss(
                outputs,
                loss_key,
                sigreg_weight=loss_config.sigreg_weight,
                sigreg_encoder_weight=loss_config.sigreg_encoder_weight,
                num_slices=loss_config.num_slices,
            )
            return total_loss, metrics

        # Compute gradients with respect to model parameters
        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            state.model
        )

        # Partition model into params and static for optimizer update
        params = eqx.filter(state.model, eqx.is_array)
        grad_params = eqx.filter(grads, eqx.is_array)

        # Apply optimizer
        updates, new_opt_state = optimizer.update(grad_params, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Reconstruct model with updated params
        new_model = eqx.combine(new_params, state.model)

        # Build new state
        new_state = TrainState(
            model=new_model,
            opt_state=new_opt_state,
            step=state.step + 1,
            key=key,
            best_loss=state.best_loss,
        )

        # Enforce output sharding for multi-GPU
        if sharding is not None:
            _, _, model_sharding = sharding
            new_state = eqx.filter_shard(new_state, model_sharding)

        return new_state, metrics

    return train_step


def make_eval_step(
    loss_config: LossConfig,
    sharding: ShardingConfig = None,
    precision_config: Optional[PrecisionConfig] = None,
):
    """
    Create a JIT-compiled evaluation step function.

    Args:
        loss_config: Loss configuration
        sharding: Sharding config from init_sharding() (None for single GPU)
        precision_config: Mixed precision config (None defaults to bfloat16)

    Returns:
        eval_step function: (state, batch) -> metrics
    """
    if precision_config is None:
        precision_config = PrecisionConfig()

    compute_dtype = get_compute_dtype(precision_config)
    loss_in_fp32 = precision_config.loss_in_float32

    @eqx.filter_jit
    def eval_step(
        state: TrainState,
        batch: Float[Array, "batch time"],
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        """
        Single evaluation step (no gradient computation).

        Args:
            state: Current training state
            batch: Audio waveforms [batch, time] at 16kHz
            key: PRNG key for evaluation

        Returns:
            Metrics dict
        """
        # Enforce sharding constraints for multi-GPU
        if sharding is not None:
            _, data_sharding, model_sharding = sharding
            state = eqx.filter_shard(state, model_sharding)
            batch = eqx.filter_shard(batch, data_sharding)

        # Cast to compute dtype for mixed precision
        batch = batch.astype(compute_dtype)

        loss_key, forward_key = jax.random.split(key)
        batch_size = batch.shape[0]
        forward_keys = jax.random.split(forward_key, batch_size)

        # Cast model to compute dtype for mixed precision forward pass
        model_compute = cast_model_to_compute(state.model, compute_dtype)

        def single_forward(waveform, k):
            return model_compute.forward_train(waveform, key=k)

        outputs = jax.vmap(single_forward)(batch, forward_keys)

        # Cast outputs to float32 for stable loss computation
        if loss_in_fp32:
            outputs = jax.tree.map(
                lambda x: cast_to_float32(x) if x.dtype == compute_dtype else x,
                outputs,
            )

        _, metrics = compute_loss(
            outputs,
            loss_key,
            sigreg_weight=loss_config.sigreg_weight,
            num_slices=loss_config.num_slices,
        )

        return metrics

    return eval_step
