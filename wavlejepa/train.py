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
from typing import Optional

import jax
from tqdm import tqdm

from .model import WavLeJEPAConfig
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
):
    """
    Main training loop.

    Args:
        config_path: Path to YAML config file (uses defaults if None)
        shards_path: Path to WebDataset shards (e.g., "s3://bucket/train-{000..999}.tar")
        use_wandb: Whether to enable wandb logging
        use_dummy_data: Use random dummy data instead of real data
        model_config: Optional model config (uses defaults if None)
    """
    # Load config
    if config_path:
        config = TrainingConfig.from_yaml(config_path)
    else:
        config = TrainingConfig()

    if model_config is None:
        model_config = WavLeJEPAConfig()

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
        print(f"Loading data from: {shards_path}")
        data_config = DataConfig(
            shards_path=shards_path,
            sample_rate=config.data.sample_rate,
            crop_duration=config.data.crop_duration,
            batch_size=global_batch_size,
            shuffle_buffer=1000,
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

    # Create training step
    train_step_fn = make_train_step(optimizer, config.loss, sharding)
    eval_step_fn = make_eval_step(config.loss, sharding)

    # Initialize logger
    if use_wandb:
        logger = WandBLogger(config.logging, config)
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

    try:
        for step in pbar:
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
                print(f"\nCheckpoint saved at step {step}")

    except KeyboardInterrupt:
        print("\nTraining interrupted")

    # Final save
    checkpointer.save(state)
    checkpointer.wait_until_finished()
    logger.finish()

    print("Training complete!")


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
        help="Path to WebDataset shards (e.g., 's3://bucket/train-{000..999}.tar')",
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
    args = parser.parse_args()

    main(
        config_path=args.config,
        shards_path=args.shards,
        use_wandb=not args.no_wandb,
        use_dummy_data=args.dummy,
    )
