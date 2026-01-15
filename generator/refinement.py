"""Test-time optimization for attractor refinement.

Refines initial attractors from the generator by minimizing the energy
function via gradient descent.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .energy import total_energy


def refine_attractors(
    attractors: Float[Array, "num_attractors attractor_dim"],
    frames: Float[Array, "num_frames hidden_dim"],
    valid_count: int | Float[Array, ""],
    num_steps: int = 50,
    lr: float = 0.01,
    tau: float = 0.1,
    lambda_separation: float = 1.0,
    lambda_coverage: float = 0.1,
    separation_margin: float = 1.0,
    min_usage: float = 1.0,
    early_stop_threshold: float | None = 1e-4,
) -> Float[Array, "num_attractors attractor_dim"]:
    """Refine attractors by minimizing energy function.

    Uses gradient descent on the energy function to polish initial
    attractors from the generator.

    Args:
        attractors: [K, D] initial attractors from generator
        frames: [N, D] contextualized frame embeddings (from LinearAttentionStack)
        valid_count: Number of valid attractors (rest are padding)
        num_steps: Maximum optimization steps
        lr: Learning rate for gradient descent
        tau: Temperature for soft assignment (low = hard)
        lambda_separation: Weight for separation term
        lambda_coverage: Weight for coverage term
        separation_margin: Margin for separation hinge loss
        min_usage: Minimum usage for coverage penalty
        early_stop_threshold: Stop if energy change < this (None to disable)

    Returns:
        Refined attractors [K, D]
    """
    K = attractors.shape[0]

    # Build attractor mask from valid_count
    attractor_mask = (jnp.arange(K) < valid_count).astype(jnp.float32)

    def energy_fn(A: Float[Array, "K D"]) -> Float[Array, ""]:
        return total_energy(
            frames=frames,
            attractors=A,
            tau=tau,
            lambda_separation=lambda_separation,
            lambda_coverage=lambda_coverage,
            separation_margin=separation_margin,
            min_usage=min_usage,
            attractor_mask=attractor_mask,
        )

    grad_fn = jax.grad(energy_fn)

    if early_stop_threshold is not None:
        # Use while_loop with early stopping
        def cond_fn(state):
            A, prev_energy, step, continue_flag = state
            return continue_flag & (step < num_steps)

        def body_fn(state):
            A, prev_energy, step, _ = state

            # Compute gradient and update
            grad_A = grad_fn(A)

            # Mask gradients for invalid attractors
            grad_A = grad_A * attractor_mask[:, None]

            A_new = A - lr * grad_A

            # Compute new energy
            new_energy = energy_fn(A_new)

            # Check convergence
            energy_change = jnp.abs(prev_energy - new_energy)
            continue_flag = energy_change > early_stop_threshold

            return (A_new, new_energy, step + 1, continue_flag)

        init_energy = energy_fn(attractors)
        init_state = (attractors, init_energy, jnp.array(0), jnp.array(True))

        final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
        refined, _, _, _ = final_state

    else:
        # Use fori_loop for fixed number of steps
        def step_fn(_, A):
            grad_A = grad_fn(A)
            grad_A = grad_A * attractor_mask[:, None]
            return A - lr * grad_A

        refined = jax.lax.fori_loop(0, num_steps, step_fn, attractors)

    return refined


def refine_attractors_with_trace(
    attractors: Float[Array, "num_attractors attractor_dim"],
    frames: Float[Array, "num_frames hidden_dim"],
    valid_count: int | Float[Array, ""],
    num_steps: int = 50,
    lr: float = 0.01,
    tau: float = 0.1,
    lambda_separation: float = 1.0,
    lambda_coverage: float = 0.1,
    separation_margin: float = 1.0,
    min_usage: float = 1.0,
) -> Tuple[
    Float[Array, "num_attractors attractor_dim"],
    Float[Array, " num_steps"],
]:
    """Refine attractors and return energy trace for debugging.

    Same as refine_attractors but also returns energy at each step.
    Useful for visualizing convergence. Always runs full num_steps.

    Args:
        attractors: [K, D] initial attractors from generator
        frames: [N, D] contextualized frame embeddings
        valid_count: Number of valid attractors
        num_steps: Number of optimization steps
        lr: Learning rate
        tau: Temperature for soft assignment
        lambda_separation: Weight for separation term
        lambda_coverage: Weight for coverage term
        separation_margin: Margin for separation hinge loss
        min_usage: Minimum usage for coverage penalty

    Returns:
        (refined_attractors, energy_trace): Refined attractors and energy per step
    """
    K = attractors.shape[0]
    attractor_mask = (jnp.arange(K) < valid_count).astype(jnp.float32)

    def energy_fn(A):
        return total_energy(
            frames=frames,
            attractors=A,
            tau=tau,
            lambda_separation=lambda_separation,
            lambda_coverage=lambda_coverage,
            separation_margin=separation_margin,
            min_usage=min_usage,
            attractor_mask=attractor_mask,
        )

    grad_fn = jax.grad(energy_fn)

    def step_fn(A, _):
        energy = energy_fn(A)
        grad_A = grad_fn(A)
        grad_A = grad_A * attractor_mask[:, None]
        A_new = A - lr * grad_A
        return A_new, energy

    refined, energy_trace = jax.lax.scan(step_fn, attractors, None, length=num_steps)

    return refined, energy_trace
