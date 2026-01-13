"""
Projector for WavLeJEPA.

MLP that maps representations to a lower-dimensional space
where SIGReg enforces isotropic Gaussian distribution.
"""

import jax
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class Projector(eqx.Module):
    """MLP projector for SIGReg regularization.

    Maps encoder/predictor outputs to a lower-dimensional space where
    SIGReg enforces isotropic Gaussian distribution.

    Architecture: Linear → GELU → LayerNorm → ... → Linear (no norm on last)

    At inference time, this module is discarded - only used for training.
    """

    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)

    layers: list[eqx.nn.Linear]
    norms: list[eqx.nn.LayerNorm]

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: tuple[int, ...] = (2048, 2048),
        output_dim: int = 256,
        layer_norm_eps: float = 1e-6,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            input_dim: Input dimension (default 768, matches encoder output)
            hidden_dims: Hidden layer dimensions (default (2048, 2048))
            output_dim: Output dimension for SIGReg (default 256)
            layer_norm_eps: LayerNorm epsilon (default 1e-6)
            key: JAX PRNG key
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        dims = [input_dim, *hidden_dims, output_dim]
        num_layers = len(dims) - 1

        keys = jax.random.split(key, num_layers)

        self.layers = []
        self.norms = []

        for i in range(num_layers):
            self.layers.append(
                eqx.nn.Linear(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    use_bias=True,
                    key=keys[i],
                )
            )
            # No norm on last layer
            if i < num_layers - 1:
                self.norms.append(
                    eqx.nn.LayerNorm(shape=dims[i + 1], eps=layer_norm_eps)
                )

    def __call__(
        self,
        x: Float[Array, "*batch input_dim"],
    ) -> Float[Array, "*batch output_dim"]:
        """Project input to SIGReg space.

        Args:
            x: Input tensor [..., input_dim]

        Returns:
            Projected tensor [..., output_dim]
        """
        # Handle arbitrary batch dimensions by reshaping
        original_shape = x.shape
        batch_shape = original_shape[:-1]
        x = x.reshape(-1, self.input_dim)  # [batch, input_dim]

        for i, layer in enumerate(self.layers):
            assert layer.bias is not None
            x = x @ layer.weight.T + layer.bias

            # Apply norm and activation (except last layer)
            if i < len(self.norms):
                x = jax.vmap(self.norms[i])(x)
                x = jax.nn.gelu(x)

        # Restore original batch shape
        x = x.reshape(*batch_shape, self.output_dim)
        return x
