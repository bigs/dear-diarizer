import jax
import jax.numpy as jnp


def sigreg(
    x: jax.Array,
    key: jax.Array,
    num_slices: int = 256,
    axis_name: str | None = None,
) -> jax.Array:
    """
    SIGReg implementation in JAX.

    Args:
        x: Input array of shape (batch, features)
        key: JAX PRNGKey for random projection
        num_slices: Number of random projections
        axis_name: Optional axis name for pmean (for distributed training)
    """
    n_batch, n_features = x.shape

    # Slice sampling -- synced across devices if the same key is used
    A = jax.random.normal(key, (n_features, num_slices))
    A = A / jnp.linalg.norm(A, ord=2, axis=0, keepdims=True)

    # Project x onto the random slices
    x_proj = x @ A  # (batch, num_slices)

    # Integration points
    t = jnp.linspace(-5, 5, 17)

    # Theoretical CF for N(0, 1) and Gauss. window
    exp_f = jnp.exp(-0.5 * t**2)

    # Empirical CF -- gathered across devices if axis_name is provided
    # x_proj: (N, M), t: (T,) -> x_t: (N, M, T)
    x_t = x_proj[:, :, jnp.newaxis] * t[jnp.newaxis, jnp.newaxis, :]

    # ecf = mean over batch dimension (N)
    # Resulting ecf shape: (num_slices, 17)
    ecf = jnp.mean(jnp.exp(1j * x_t), axis=0)

    # all_reduce if axis_name is provided (equivalent to AVG)
    if axis_name is not None:
        ecf = jax.lax.pmean(ecf, axis_name=axis_name)
        world_size = jax.lax.psum(1.0, axis_name=axis_name)
    else:
        world_size = 1.0

    # Weighted L2 distance
    # ecf: (M, T), exp_f: (T,)
    err = jnp.square(jnp.abs(ecf - exp_f)) * exp_f

    global_n = n_batch * world_size

    # T = trapz(err, t, dim=1) * N
    # Note: jnp.trapezoid is the modern version of jnp.trapz
    res = jnp.trapezoid(err, x=t, axis=1) * global_n

    return res
