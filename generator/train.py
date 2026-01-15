"""Training loop for the Speaker Attractor Generator.

Implements:
- Full training step with energy + confidence + regularization losses
- Temperature annealing (deterministic annealing)
- Support for supervised (speaker count known) and unsupervised modes
"""

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from .attractor import AttractorGenerator
from .config import GeneratorConfig
from .energy import total_energy
from .losses import confidence_loss
from .regularization import compute_regularization


@dataclass
class TrainConfig:
    """Training configuration."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Temperature annealing
    tau_start: float = 1.0  # Initial temperature (soft assignments)
    tau_end: float = 0.1  # Final temperature (hard assignments)
    anneal_steps: int = 10000  # Steps to anneal over

    # Loss weights
    lambda_separation: float = 1.0
    lambda_coverage: float = 0.1
    lambda_confidence: float = 1.0
    lambda_cardinality: float = 1.0
    lambda_spread: float = 0.1

    # Energy function params
    separation_margin: float = 1.0
    min_usage: float = 1.0

    # Regularization params
    over_penalty: float = 2.0  # Asymmetry for cardinality loss
    num_segments: int = 10  # Temporal segments for spread
    min_segments: int = 3  # Minimum presence for spread
    presence_threshold: float = 1.0

    # Cardinality
    max_speakers: int = 4  # Soft cap when unsupervised

    # Training
    num_attractors: int = 10  # Fixed generation count for training


def get_temperature(step: int, config: TrainConfig) -> Float[Array, ""]:
    """Compute temperature for current step (linear annealing)."""
    progress = jnp.clip(step / config.anneal_steps, 0.0, 1.0)
    return config.tau_start + progress * (config.tau_end - config.tau_start)


def compute_loss(
    model: AttractorGenerator,
    frames: Float[Array, "num_frames input_dim"],
    tau: float | Float[Array, ""],
    config: TrainConfig,
    target_count: Float[Array, ""] | None = None,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """Compute full training loss.

    Args:
        model: AttractorGenerator model
        frames: [N, input_dim] frame embeddings from WavLeJEPA
        tau: Current temperature for soft assignment
        config: Training configuration
        target_count: Ground truth speaker count (None for unsupervised)

    Returns:
        (total_loss, loss_dict): Total loss and breakdown for logging
    """
    # Generate attractors (fixed count for differentiability)
    # Also get contextualized frames for energy computation (same embedding space as attractors)
    attractors, confidences, contextualized = model.generate_fixed(
        frames, config.num_attractors
    )

    # Build mask from confidences (all are "valid" during training, loss handles it)
    K = attractors.shape[0]
    attractor_mask = jnp.ones(K)

    # Energy loss (use contextualized frames, not raw input)
    e_total = total_energy(
        frames=contextualized,
        attractors=attractors,
        tau=tau,
        lambda_separation=config.lambda_separation,
        lambda_coverage=config.lambda_coverage,
        separation_margin=config.separation_margin,
        min_usage=config.min_usage,
        attractor_mask=attractor_mask,
    )

    # Confidence loss
    l_conf = confidence_loss(
        confidences=confidences,
        frames=contextualized,
        attractors=attractors,
        tau=tau,
        usage_threshold=config.min_usage,
        attractor_mask=attractor_mask,
    )

    # Regularization (cardinality + temporal spread)
    l_reg, reg_dict = compute_regularization(
        frames=contextualized,
        attractors=attractors,
        confidences=confidences,
        tau=tau,
        target_count=target_count,
        max_count=config.max_speakers if target_count is None else None,
        lambda_cardinality=config.lambda_cardinality,
        lambda_spread=config.lambda_spread,
        over_penalty=config.over_penalty,
        num_segments=config.num_segments,
        min_segments=config.min_segments,
        presence_threshold=config.presence_threshold,
        attractor_mask=attractor_mask,
    )

    # Total loss
    total = e_total + config.lambda_confidence * l_conf + l_reg

    loss_dict = {
        "total": total,
        "energy": e_total,
        "confidence": l_conf,
        "cardinality": reg_dict["cardinality"],
        "temporal_spread": reg_dict["temporal_spread"],
        "tau": jnp.array(tau),
    }

    return total, loss_dict


def make_train_step(
    config: TrainConfig,
) -> Callable:
    """Create a JIT-compiled training step function.

    Args:
        config: Training configuration

    Returns:
        train_step function: (model, opt_state, frames, step, target_count, key) -> (model, opt_state, loss_dict)
    """

    @eqx.filter_jit
    def train_step(
        model: AttractorGenerator,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        frames: Float[Array, "num_frames input_dim"],
        step: int,
        target_count: Float[Array, ""] | None = None,
        key: PRNGKeyArray | None = None,
    ) -> tuple[AttractorGenerator, optax.OptState, dict[str, Float[Array, ""]]]:
        """Single training step.

        Args:
            model: AttractorGenerator model
            opt_state: Optimizer state
            optimizer: Optax optimizer
            frames: [N, input_dim] frame embeddings
            step: Current training step (for temperature annealing)
            target_count: Ground truth speaker count (None for unsupervised)
            key: PRNG key (unused currently)

        Returns:
            (updated_model, updated_opt_state, loss_dict)
        """
        del key  # Currently unused

        # Get current temperature
        tau = get_temperature(step, config)

        # Compute loss and gradients
        def loss_fn(model):
            loss, loss_dict = compute_loss(model, frames, tau, config, target_count)
            return loss, loss_dict

        (loss, loss_dict), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model
        )

        # Apply gradient clipping
        grads = jax.tree.map(
            lambda g: jnp.clip(g, -config.max_grad_norm, config.max_grad_norm), grads
        )

        # Update model
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss_dict

    return train_step


def create_optimizer(config: TrainConfig) -> optax.GradientTransformation:
    """Create optimizer with learning rate and weight decay."""
    return optax.adamw(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def init_training(
    model_config: GeneratorConfig,
    train_config: TrainConfig,
    key: PRNGKeyArray,
) -> tuple[AttractorGenerator, optax.OptState, optax.GradientTransformation]:
    """Initialize model and optimizer for training.

    Args:
        model_config: Generator model configuration
        train_config: Training configuration
        key: PRNG key

    Returns:
        (model, opt_state, optimizer)
    """
    # Initialize model
    model = AttractorGenerator(model_config, key=key)

    # Initialize optimizer
    optimizer = create_optimizer(train_config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    return model, opt_state, optimizer


def train_epoch(
    model: AttractorGenerator,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    train_step_fn: Callable,
    data_iterator,  # Iterator yielding (frames, target_count) or just frames
    start_step: int,
    key: PRNGKeyArray,
) -> tuple[AttractorGenerator, optax.OptState, int, list[dict]]:
    """Train for one epoch.

    Args:
        model: Current model
        opt_state: Current optimizer state
        optimizer: Optimizer
        train_step_fn: JIT-compiled train step
        data_iterator: Iterator over training data
        start_step: Starting step number
        key: PRNG key

    Returns:
        (model, opt_state, final_step, epoch_losses)
    """
    step = start_step
    epoch_losses = []

    for batch in data_iterator:
        # Handle both (frames, target_count) and just frames
        if isinstance(batch, tuple):
            frames, target_count = batch
        else:
            frames = batch
            target_count = None

        key, subkey = jax.random.split(key)

        model, opt_state, loss_dict = train_step_fn(
            model, opt_state, optimizer, frames, step, target_count, subkey
        )

        epoch_losses.append(loss_dict)
        step += 1

    return model, opt_state, step, epoch_losses
