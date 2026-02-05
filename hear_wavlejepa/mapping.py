"""
Parameter name mapping utilities from JAX/Equinox to PyTorch.

Maps the Equinox pytree parameter paths to PyTorch state_dict keys.
"""

from typing import Any

import jax
import jax.tree_util
import numpy as np


def flatten_pytree(pytree: Any) -> dict[str, np.ndarray]:
    """Flatten a JAX pytree into a flat dict of path -> array.

    Uses jax.tree_util for proper flattening of JAX pytrees,
    filtering to only include actual array parameters (not scalars).

    Args:
        pytree: A nested JAX pytree (e.g., eqx.filter(model, eqx.is_array))

    Returns:
        Dict mapping dot-separated paths to numpy arrays (only non-scalar arrays)
    """
    flat = {}

    leaves_with_path = jax.tree_util.tree_leaves_with_path(pytree)

    for path, leaf in leaves_with_path:
        # Convert path keys to string
        path_parts = []
        for key in path:
            if hasattr(key, "key"):
                # GetAttrKey or similar
                path_parts.append(str(key.key))
            elif hasattr(key, "idx"):
                # SequenceKey
                path_parts.append(str(key.idx))
            else:
                path_parts.append(str(key))

        # Join and clean up path (remove leading dots, collapse double dots)
        path_str = ".".join(path_parts)
        path_str = path_str.lstrip(".")
        while ".." in path_str:
            path_str = path_str.replace("..", ".")

        # Convert to numpy
        arr = np.asarray(leaf)

        # Only include non-scalar arrays (actual weight matrices/vectors)
        if arr.ndim > 0:
            flat[path_str] = arr

    return flat


def map_waveform_encoder_params(
    jax_flat: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Map waveform encoder parameters from JAX to PyTorch.

    JAX paths:
        waveform_encoder.conv_blocks.{i}.conv.weight  # (out, in, kernel)
        waveform_encoder.conv_blocks.{i}.conv.bias    # (out, 1) in JAX
        waveform_encoder.conv_blocks.{i}.norm.weight
        waveform_encoder.conv_blocks.{i}.norm.bias
        waveform_encoder.proj.weight  # (out, in)
        waveform_encoder.proj.bias

    PyTorch paths:
        waveform_encoder.conv_blocks.{i}.conv.weight  # (out, in, kernel)
        waveform_encoder.conv_blocks.{i}.conv.bias    # (out,) in PyTorch
        waveform_encoder.conv_blocks.{i}.norm.weight
        waveform_encoder.conv_blocks.{i}.norm.bias
        waveform_encoder.proj.weight  # (out, in)
        waveform_encoder.proj.bias
    """
    torch_params = {}

    for jax_path, arr in jax_flat.items():
        if not jax_path.startswith("waveform_encoder."):
            continue

        # Handle Conv1d bias shape difference
        # JAX Conv1d uses (out_channels, 1) for bias, PyTorch uses (out_channels,)
        if ".conv.bias" in jax_path and arr.ndim == 2:
            arr = arr.squeeze(-1)

        torch_params[jax_path] = arr

    return torch_params


def map_context_encoder_params(
    jax_flat: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Map context encoder parameters from JAX to PyTorch.

    JAX structure (after proper flattening with tree_util):
        context_encoder.layers.{i}.self_attn.query_proj.weight  # (embed_dim, embed_dim)
        context_encoder.layers.{i}.self_attn.query_proj.bias    # (embed_dim,)
        context_encoder.layers.{i}.mlp.layers.0.weight  # (ffn_dim, embed_dim)
        context_encoder.layers.{i}.mlp.layers.1.weight  # (embed_dim, ffn_dim)
        context_encoder.layers.{i}.norm1.weight
        context_encoder.layers.{i}.norm2.weight

    PyTorch structure:
        context_encoder.layers.{i}.query_proj.weight
        context_encoder.layers.{i}.mlp_linear1.weight
        context_encoder.layers.{i}.mlp_linear2.weight
        context_encoder.layers.{i}.norm1.weight
        context_encoder.layers.{i}.norm2.weight
    """
    torch_params = {}

    for jax_path, arr in jax_flat.items():
        if not jax_path.startswith("context_encoder."):
            continue

        # Remove context_encoder prefix for parsing
        rel_path = jax_path[len("context_encoder.") :]

        # Positional encoding buffer
        if rel_path.startswith("pos_encoding."):
            torch_params[jax_path] = arr
            continue

        # Final norm
        if rel_path.startswith("final_norm."):
            torch_params[jax_path] = arr
            continue

        # Transformer layers
        if rel_path.startswith("layers."):
            # Parse layer index: layers.{i}.{rest}
            parts = rel_path.split(".", 2)  # ['layers', '{i}', '{rest}']
            if len(parts) < 3:
                continue
            layer_idx = parts[1]
            rest = parts[2]

            # Attention projections
            if rest.startswith("self_attn."):
                attn_rest = rest[len("self_attn.") :]
                # JAX and PyTorch both have (embed_dim, embed_dim) for weights now
                # Just need to rename: self_attn.query_proj -> query_proj
                proj_name = attn_rest.split(".")[0]  # query_proj, key_proj, etc.
                param_type = attn_rest.split(".")[1]  # weight or bias

                torch_key = f"context_encoder.layers.{layer_idx}.{proj_name}.{param_type}"
                torch_params[torch_key] = arr

            # MLP
            elif rest.startswith("mlp."):
                mlp_rest = rest[len("mlp.") :]
                # JAX: mlp.layers.0.{weight,bias} -> PyTorch: mlp_linear1.{weight,bias}
                # JAX: mlp.layers.1.{weight,bias} -> PyTorch: mlp_linear2.{weight,bias}
                if mlp_rest.startswith("layers.0."):
                    param_type = mlp_rest.split(".")[-1]
                    torch_key = f"context_encoder.layers.{layer_idx}.mlp_linear1.{param_type}"
                    torch_params[torch_key] = arr
                elif mlp_rest.startswith("layers.1."):
                    param_type = mlp_rest.split(".")[-1]
                    torch_key = f"context_encoder.layers.{layer_idx}.mlp_linear2.{param_type}"
                    torch_params[torch_key] = arr

            # Layer norms
            elif rest.startswith("norm1.") or rest.startswith("norm2."):
                torch_params[jax_path] = arr

    return torch_params


def jax_params_to_torch_state_dict(
    jax_params: Any,
) -> dict[str, np.ndarray]:
    """Convert JAX model params pytree to PyTorch state_dict.

    Args:
        jax_params: The model_params pytree restored from Orbax

    Returns:
        Dict mapping PyTorch state_dict keys to numpy arrays
    """
    # Flatten the JAX pytree
    jax_flat = flatten_pytree(jax_params)

    torch_state = {}

    # Map waveform encoder params
    torch_state.update(map_waveform_encoder_params(jax_flat))

    # Map context encoder params
    torch_state.update(map_context_encoder_params(jax_flat))

    return torch_state
