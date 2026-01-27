#!/usr/bin/env python3
"""
WavLeJEPA training script.

Usage:
    # With real data (WebDataset shards)
    python -m wavlejepa.train --shards "s3://bucket/audio-{000000..000999}.tar"

    # With dummy data (for testing)
    python -m wavlejepa.train --dummy

    # With config file
    python -m wavlejepa.train --config config.yaml --shards "path/to/shards-*.tar"

Supports:
- Single GPU and multi-GPU (data parallelism)
- Checkpoint resume
- Wandb logging
- Streaming from S3/GCS via WebDataset
"""

import argparse
import signal
from typing import Optional

import jax
from tqdm import tqdm

from .model import WavLeJEPAConfig, MaskingConfig
from .data import DataConfig, AudioDataLoader
from .training import (
    TrainingConfig,
    create_train_state,
    get_lr_at_step,
    init_sharding,
    shard_batch,
    shard_state,
    make_train_step,
    make_eval_step,
    WavLeJEPACheckpointer,
    WandBLogger,
    NoOpLogger,
)


class GracefulShutdown:
    """Handles graceful shutdown on SIGINT."""

    def __init__(self):
        self.shutdown_requested = False
        self._original_handler = None

    def request_shutdown(self, _signum, _frame):
        """Signal handler that requests graceful shutdown."""
        if self.shutdown_requested:
            # Second SIGINT - force exit
            tqdm.write("\nForced exit requested, terminating immediately...")
            raise KeyboardInterrupt
        self.shutdown_requested = True
        tqdm.write("\nShutdown requested, finishing current step...")

    def install(self):
        """Install the signal handler."""
        self._original_handler = signal.signal(signal.SIGINT, self.request_shutdown)

    def uninstall(self):
        """Restore original signal handler."""
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)


def create_dummy_batch(
    batch_size: int,
    duration_samples: int,
    key: jax.Array,
) -> jax.Array:
    """Create dummy batch for testing."""
    return jax.random.normal(key, (batch_size, duration_samples))


def main(
    config_path: Optional[str] = None,
    shards_path: Optional[str] = None,
    use_wandb: bool = True,
    use_dummy_data: bool = False,
    model_config: Optional[WavLeJEPAConfig] = None,
    resume_wandb_id: Optional[str] = None,
):
    """
    Main training loop.

    Args:
        config_path: Path to YAML config file (uses defaults if None)
        shards_path: Path to WebDataset shards (e.g., "s3://bucket/train-{000..999}.tar")
        use_wandb: Whether to enable wandb logging
        use_dummy_data: Use random dummy data instead of real data
        model_config: Optional model config (uses defaults if None)
        resume_wandb_id: Optional wandb run ID to resume logging to
    """
    # Load config
    if config_path:
        config = TrainingConfig.from_yaml(config_path)
    else:
        config = TrainingConfig()

    if model_config is None:
        # Build masking config from overrides (if any)
        masking_overrides = config.masking
        masking_kwargs = {}
        if masking_overrides.context_ratio is not None:
            masking_kwargs["context_ratio"] = masking_overrides.context_ratio
        if masking_overrides.target_ratio is not None:
            masking_kwargs["target_ratio"] = masking_overrides.target_ratio
        if masking_overrides.context_block_length is not None:
            masking_kwargs["context_block_length"] = masking_overrides.context_block_length
        if masking_overrides.target_block_length is not None:
            masking_kwargs["target_block_length"] = masking_overrides.target_block_length
        if masking_overrides.num_target_groups is not None:
            masking_kwargs["num_target_groups"] = masking_overrides.num_target_groups
        if masking_overrides.min_context_ratio is not None:
            masking_kwargs["min_context_ratio"] = masking_overrides.min_context_ratio

        masking_config = MaskingConfig(**masking_kwargs) if masking_kwargs else None
        model_config = WavLeJEPAConfig(masking=masking_config) if masking_config else WavLeJEPAConfig()

    # Initialize multi-GPU sharding
    sharding = init_sharding()
    num_devices = len(jax.devices())
    print(f"Training on {num_devices} device(s)")

    # Global batch size
    global_batch_size = config.data.batch_size * num_devices
    duration_samples = int(config.data.crop_duration * config.data.sample_rate)
    print(f"Global batch size: {global_batch_size}")
    print(f"Audio duration: {config.data.crop_duration}s ({duration_samples} samples)")

    # Initialize data loader
    if use_dummy_data or shards_path is None:
        print("Using dummy data (random noise)")
        data_loader = None
    else:
        # Detect HuggingFace dataset path (hf://dataset/name or hf://dataset/name:subset)
        if shards_path.startswith("hf://"):
            hf_path = shards_path[5:]  # Remove "hf://" prefix
            # Parse optional subset: "dataset/name:subset" -> ("dataset/name", "subset")
            if ":" in hf_path:
                hf_dataset, hf_subset = hf_path.rsplit(":", 1)
            else:
                hf_dataset = hf_path
                hf_subset = "unbalanced"  # Default for AudioSet
            print(f"Loading data from HuggingFace: {hf_dataset} (subset: {hf_subset})")
            data_config = DataConfig(
                hf_dataset=hf_dataset,
                hf_subset=hf_subset,
                sample_rate=config.data.sample_rate,
                crop_duration=config.data.crop_duration,
                crops_per_audio=config.data.crops_per_audio,
                batch_size=global_batch_size,
                shuffle_buffer=1000,
                num_workers=config.data.num_workers,
                prefetch_batches=config.data.prefetch_batches,
            )
        else:
            # WebDataset shards (local, s3://, gs://)
            print(f"Loading data from: {shards_path}")
            data_config = DataConfig(
                shards_path=shards_path,
                sample_rate=config.data.sample_rate,
                crop_duration=config.data.crop_duration,
                crops_per_audio=config.data.crops_per_audio,
                batch_size=global_batch_size,
                shuffle_buffer=1000,
                num_workers=config.data.num_workers,
                prefetch_batches=config.data.prefetch_batches,
            )
        data_loader = iter(AudioDataLoader(data_config, seed=config.seed, infinite=True))

    # Initialize checkpointer
    checkpointer = WavLeJEPACheckpointer(
        config.checkpoint,
        config,
        model_config,
    )

    # Try to resume from checkpoint
    key = jax.random.key(config.seed)
    restored = checkpointer.restore(key=key)

    if restored is not None:
        state, optimizer = restored
        start_step = int(state.step)
        print(f"Resumed from step {start_step}")
    else:
        print("Starting fresh training")
        state, optimizer = create_train_state(
            model_config,
            config.optimizer,
            key,
        )
        start_step = 0

    # Print model info
    param_counts = state.model.count_params()
    print(f"Model parameters: {param_counts['total']:,}")

    # Shard state across devices
    state = shard_state(state, sharding)

    # Create training step with mixed precision
    train_step_fn = make_train_step(
        optimizer, config.loss, sharding, config.precision
    )
    eval_step_fn = make_eval_step(config.loss, sharding, config.precision)
    print(f"Compute dtype: {config.precision.compute_dtype}")

    # Initialize logger
    if use_wandb:
        logger = WandBLogger(config.logging, config, resume_run_id=resume_wandb_id)
    else:
        logger = NoOpLogger()

    # Training loop
    pbar = tqdm(
        range(start_step, config.optimizer.total_steps),
        initial=start_step,
        total=config.optimizer.total_steps,
        desc="Training",
    )

    eval_key = jax.random.key(config.seed + 1)

    # Set up graceful shutdown handler
    shutdown = GracefulShutdown()
    shutdown.install()

    interrupted = False
    try:
        for step in pbar:
            # Check for graceful shutdown request
            if shutdown.shutdown_requested:
                interrupted = True
                break

            # Get batch
            if data_loader is not None:
                batch = next(data_loader)
            else:
                key, batch_key = jax.random.split(key)
                batch = create_dummy_batch(global_batch_size, duration_samples, batch_key)

            batch = shard_batch(batch, sharding)

            # Training step
            state, metrics = train_step_fn(state, batch)

            # Logging
            current_lr = get_lr_at_step(config.optimizer, step)
            metrics["lr"] = current_lr
            logger.log_step(step, metrics)

            # Update progress bar
            loss_val = float(metrics["loss/total"])
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{current_lr:.2e}")

            # Evaluation
            if step % config.logging.eval_every_n_steps == 0 and step > 0:
                if data_loader is not None:
                    val_batch = next(data_loader)
                else:
                    eval_key, eval_batch_key = jax.random.split(eval_key)
                    val_batch = create_dummy_batch(
                        global_batch_size, duration_samples, eval_batch_key
                    )
                val_batch = shard_batch(val_batch, sharding)

                eval_key, eval_step_key = jax.random.split(eval_key)
                val_metrics = eval_step_fn(state, val_batch, eval_step_key)
                logger.log_eval(step, val_metrics)

                # Save best model
                val_loss = float(val_metrics["loss/total"])
                state = checkpointer.save_best(state, val_loss)

            # Checkpointing
            if step % config.checkpoint.save_every_n_steps == 0 and step > 0:
                checkpointer.save(state, metrics)
                tqdm.write(f"Checkpoint saved at step {step}")

            # Check again after step completes (in case signal arrived during step)
            if shutdown.shutdown_requested:
                interrupted = True
                break

    except KeyboardInterrupt:
        # Second Ctrl+C during graceful shutdown triggers immediate exit
        interrupted = True
        tqdm.write("Training interrupted")

    finally:
        shutdown.uninstall()

    # Finalization
    pbar.close()

    if interrupted:
        tqdm.write("Saving checkpoint before exit...")

    checkpointer.save(state)
    tqdm.write("Waiting for checkpoint save to complete...")
    checkpointer.wait_until_finished()
    tqdm.write("Checkpoint saved.")

    tqdm.write("Finalizing wandb run...")
    logger.finish()

    if interrupted:
        tqdm.write("Training interrupted at step {}.".format(int(state.step)))
    else:
        tqdm.write("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WavLeJEPA")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--shards",
        type=str,
        default=None,
        help="Data source: WebDataset shards (local/s3://bucket/train-{000..999}.tar) "
             "or HuggingFace dataset (hf://agkphysics/AudioSet:unbalanced)",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy data (random noise) for testing",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--resume-wandb-id",
        type=str,
        default=None,
        help="Wandb run ID to resume logging to (found in run URL)",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        shards_path=args.shards,
        use_wandb=not args.no_wandb,
        use_dummy_data=args.dummy,
        resume_wandb_id=args.resume_wandb_id,
    )
