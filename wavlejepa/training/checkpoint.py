"""
Orbax checkpointing for WavLeJEPA.

Handles the Equinox + Orbax integration:
- Separates array params from static model structure
- Async checkpointing for performance
- Seamless resume with full state restoration
"""

from pathlib import Path
from typing import Optional
import json

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import orbax.checkpoint as ocp
from jaxtyping import PRNGKeyArray

from ..model import WavLeJEPA, WavLeJEPAConfig
from .state import TrainState, create_optimizer
from .config import TrainingConfig, CheckpointConfig


class WavLeJEPACheckpointer:
    """
    Checkpoint manager for WavLeJEPA training.

    Key design decisions:
    1. Separates array params from static model structure for Orbax
    2. Stores TrainingConfig as JSON for model reconstruction on restore
    3. Uses async checkpointing to not block training
    4. Tracks best model by validation loss
    """

    def __init__(
        self,
        config: CheckpointConfig,
        training_config: TrainingConfig,
        model_config: WavLeJEPAConfig,
    ):
        self.config = config
        self.training_config = training_config
        self.model_config = model_config
        self.checkpoint_dir = Path(config.checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save configs for reconstruction on restore
        self._save_configs()

        # Orbax checkpoint manager
        options = ocp.CheckpointManagerOptions(
            max_to_keep=config.keep_n_checkpoints,
            create=True,
        )

        self.manager = ocp.CheckpointManager(
            self.checkpoint_dir / "checkpoints",
            options=options,
        )

        # Separate manager for best model
        if config.save_best:
            best_options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
            self.best_manager = ocp.CheckpointManager(
                self.checkpoint_dir / "best",
                options=best_options,
            )
        else:
            self.best_manager = None

        # Cache for optimizer (needed for restore)
        self._optimizer: Optional[optax.GradientTransformation] = None

    def _save_configs(self) -> None:
        """Save configs for reconstruction on restore."""
        config_path = self.checkpoint_dir / "training_config.json"
        self.training_config.save(config_path)

        # Save model config separately (it's a dataclass with nested MaskingConfig)
        model_config_path = self.checkpoint_dir / "model_config.json"
        with open(model_config_path, "w") as f:
            json.dump(
                {
                    "waveform_embed_dim": self.model_config.waveform_embed_dim,
                    "waveform_num_groups": self.model_config.waveform_num_groups,
                    "context_embed_dim": self.model_config.context_embed_dim,
                    "context_num_heads": self.model_config.context_num_heads,
                    "context_num_layers": self.model_config.context_num_layers,
                    "context_ffn_dim": self.model_config.context_ffn_dim,
                    "context_dropout": self.model_config.context_dropout,
                    "context_top_k_layers": self.model_config.context_top_k_layers,
                    "predictor_dim": self.model_config.predictor_dim,
                    "predictor_num_heads": self.model_config.predictor_num_heads,
                    "predictor_num_layers": self.model_config.predictor_num_layers,
                    "predictor_ffn_dim": self.model_config.predictor_ffn_dim,
                    "predictor_dropout": self.model_config.predictor_dropout,
                    "projector_hidden_dims": list(
                        self.model_config.projector_hidden_dims
                    ),
                    "projector_output_dim": self.model_config.projector_output_dim,
                    "max_seq_len": self.model_config.max_seq_len,
                },
                f,
                indent=2,
            )

    def _state_to_saveable(self, state: TrainState) -> dict:
        """
        Convert TrainState to Orbax-saveable format.

        Equinox modules contain both arrays (parameters) and non-arrays
        (configs, activation functions). We only save the arrays.
        """
        # Extract only array leaves from model
        model_params = eqx.filter(state.model, eqx.is_array)

        return {
            "model_params": model_params,
            "opt_state": state.opt_state,
            "step": state.step,
            "key": state.key,
            "best_loss": state.best_loss,
        }

    def _saveable_to_state(
        self,
        saveable: dict,
        model_template: WavLeJEPA,
    ) -> TrainState:
        """
        Reconstruct TrainState from saved checkpoint.

        Args:
            saveable: Loaded checkpoint dict
            model_template: Fresh model with correct static structure
        """
        # Combine loaded params with model structure
        model = eqx.combine(saveable["model_params"], model_template)

        return TrainState(
            model=model,
            opt_state=saveable["opt_state"],
            step=saveable["step"],
            key=saveable["key"],
            best_loss=saveable["best_loss"],
        )

    def save(self, state: TrainState, metrics: Optional[dict] = None) -> None:
        """
        Save checkpoint (async).

        Args:
            state: Current training state
            metrics: Optional metrics dict for metadata
        """
        step = int(state.step)
        saveable = self._state_to_saveable(state)

        self.manager.save(
            step,
            args=ocp.args.StandardSave(saveable),
        )

    def save_best(self, state: TrainState, val_loss: float) -> TrainState:
        """
        Save best model if validation loss improved.

        Args:
            state: Current training state
            val_loss: Current validation loss

        Returns:
            Updated state with new best_loss if improved, else original state
        """
        if self.best_manager is None:
            return state

        current_best = float(state.best_loss)
        if val_loss >= current_best:
            return state

        # Update best_loss in state
        new_state = TrainState(
            model=state.model,
            opt_state=state.opt_state,
            step=state.step,
            key=state.key,
            best_loss=jnp.array(val_loss, dtype=jnp.float32),
        )

        saveable = self._state_to_saveable(new_state)
        self.best_manager.save(
            int(state.step),
            args=ocp.args.StandardSave(saveable),
        )

        return new_state

    def restore(
        self,
        step: Optional[int] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[tuple[TrainState, optax.GradientTransformation]]:
        """
        Restore training state from checkpoint.

        Args:
            step: Specific step to restore (None = latest)
            key: PRNG key for model template initialization

        Returns:
            Tuple of (TrainState, optimizer) or None if no checkpoint exists
        """
        if step is None:
            step = self.manager.latest_step()

        if step is None:
            return None

        # Create model template for structure
        if key is None:
            key = jax.random.key(0)
        model_template = WavLeJEPA(self.model_config, key=key)

        # Create optimizer
        optimizer = create_optimizer(self.training_config.optimizer)
        self._optimizer = optimizer

        # Create abstract target for efficient restore
        params_template = eqx.filter(model_template, eqx.is_array)
        opt_state_template = optimizer.init(params_template)

        abstract_target = {
            "model_params": params_template,
            "opt_state": opt_state_template,
            "step": jnp.array(0, dtype=jnp.int32),
            "key": jax.random.key(0),
            "best_loss": jnp.array(0.0, dtype=jnp.float32),
        }

        # Restore checkpoint
        restored = self.manager.restore(
            step,
            args=ocp.args.StandardRestore(abstract_target),
        )

        state = self._saveable_to_state(restored, model_template)

        return state, optimizer

    def restore_best(
        self,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[tuple[TrainState, optax.GradientTransformation]]:
        """Restore best model checkpoint."""
        if self.best_manager is None:
            return None

        step = self.best_manager.latest_step()
        if step is None:
            return None

        # Create model template
        if key is None:
            key = jax.random.key(0)
        model_template = WavLeJEPA(self.model_config, key=key)

        # Create optimizer
        optimizer = create_optimizer(self.training_config.optimizer)

        # Create abstract target
        params_template = eqx.filter(model_template, eqx.is_array)
        opt_state_template = optimizer.init(params_template)

        abstract_target = {
            "model_params": params_template,
            "opt_state": opt_state_template,
            "step": jnp.array(0, dtype=jnp.int32),
            "key": jax.random.key(0),
            "best_loss": jnp.array(0.0, dtype=jnp.float32),
        }

        restored = self.best_manager.restore(
            step,
            args=ocp.args.StandardRestore(abstract_target),
        )

        state = self._saveable_to_state(restored, model_template)

        return state, optimizer

    def wait_until_finished(self) -> None:
        """Block until all async saves complete."""
        self.manager.wait_until_finished()
        if self.best_manager:
            self.best_manager.wait_until_finished()

    @property
    def latest_step(self) -> Optional[int]:
        """Get latest checkpoint step."""
        return self.manager.latest_step()
