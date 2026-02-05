"""HEAR-compatible adapter for WavLeJEPA."""

from .hear_api import (
    load_model,
    get_scene_embeddings,
    get_timestamp_embeddings,
)

__all__ = [
    "load_model",
    "get_scene_embeddings",
    "get_timestamp_embeddings",
]
