"""
Training state management for WavLeJEPA.

TrainState is a PyTree containing model, optimizer state, step, and PRNG key.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array, PRNGKeyArray

from ..model import WavLeJEPA, WavLeJEPAConfig
from .config import OptimizerConfig


class TrainState(eqx.Module):
    """
    Training state as an Equinox Module (proper PyTree).

    All fields are JAX-traceable pytrees:
    - model: Equinox module (pytree of arrays + static fields)
    - opt_state: Optax optimizer state (pytree of arrays)
    - step: Scalar int32 array for step count
    - key: PRNG key for reproducibility
    - best_loss: Best validation loss seen (for best checkpoint)
    """

    model: eqx.Module
    opt_state: optax.OptState
    step: Array
    key: PRNGKeyArray
    best_loss: Array


def create_optimizer(config: OptimizerConfig) -> optax.GradientTransformation:
    """
    Create optimizer with warmup + cosine decay schedule.

    Schedule:
    - Linear warmup from 0 to peak_lr over warmup_steps
    - Cosine decay from peak_lr to 0 over remaining steps
    """
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=config.peak_lr,
                transition_steps=config.warmup_steps,
            ),
            optax.cosine_decay_schedule(
                init_value=config.peak_lr,
                decay_steps=config.total_steps - config.warmup_steps,
            ),
        ],
        boundaries=[config.warmup_steps],
    )

    return optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=config.weight_decay,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
        ),
    )


def create_train_state(
    model_config: WavLeJEPAConfig,
    optimizer_config: OptimizerConfig,
    key: PRNGKeyArray,
) -> tuple[TrainState, optax.GradientTransformation]:
    """
    Initialize training state from configs.

    Args:
        model_config: WavLeJEPA model configuration
        optimizer_config: Optimizer configuration
        key: PRNG key for initialization

    Returns:
        Tuple of (TrainState, optimizer)
    """
    key, model_key, state_key = jax.random.split(key, 3)

    # Initialize model
    model = WavLeJEPA(model_config, key=model_key)

    # Initialize optimizer
    optimizer = create_optimizer(optimizer_config)

    # Get trainable params for optimizer init
    # eqx.filter separates arrays from static fields
    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)

    state = TrainState(
        model=model,
        opt_state=opt_state,
        step=jnp.array(0, dtype=jnp.int32),
        key=state_key,
        best_loss=jnp.array(float("inf"), dtype=jnp.float32),
    )

    return state, optimizer


def get_lr_at_step(config: OptimizerConfig, step: int) -> float:
    """Get learning rate at a given step (for logging)."""
    if step < config.warmup_steps:
        # Linear warmup
        return config.peak_lr * step / config.warmup_steps
    else:
        # Cosine decay
        decay_steps = config.total_steps - config.warmup_steps
        progress = (step - config.warmup_steps) / decay_steps
        return config.peak_lr * 0.5 * (1 + jnp.cos(jnp.pi * progress))
