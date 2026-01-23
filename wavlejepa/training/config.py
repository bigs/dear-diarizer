"""
Training configuration for WavLeJEPA.

Frozen dataclasses for all training hyperparameters.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
from pathlib import Path


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer hyperparameters following WavJEPA spec."""

    peak_lr: float = 2e-4
    warmup_steps: int = 100_000
    total_steps: int = 375_000
    weight_decay: float = 0.04
    grad_clip_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass(frozen=True)
class LossConfig:
    """Loss function weights."""

    sigreg_weight: float = 0.02
    num_slices: int = 256


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpointing configuration."""

    checkpoint_dir: str = "./checkpoints"
    save_every_n_steps: int = 5000
    keep_n_checkpoints: int = 5
    save_best: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    """Wandb logging configuration."""

    project: str = "wavlejepa"
    entity: Optional[str] = None
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 1000


@dataclass(frozen=True)
class DataConfig:
    """Data pipeline configuration."""

    batch_size: int = 32  # per-device batch size
    crop_duration: float = 2.0  # seconds
    sample_rate: int = 16000
    num_workers: int = 4
    prefetch_batches: int = 2


@dataclass(frozen=True)
class PrecisionConfig:
    """Mixed precision configuration.

    For H100/GB10 (Blackwell), bfloat16 gives ~2x throughput with same
    dynamic range as float32. Master weights stay in float32 automatically.
    """

    # Compute dtype for forward/backward pass activations
    # Options: "float32", "bfloat16", "float16"
    compute_dtype: str = "bfloat16"

    # Whether to keep loss computation in float32 (recommended for stability)
    loss_in_float32: bool = True


@dataclass(frozen=True)
class MaskingConfigOverride:
    """Masking configuration overrides for sweeps.

    These override the defaults in model.MaskingConfig.
    Only specify values you want to change from defaults.
    """

    context_ratio: Optional[float] = None
    target_ratio: Optional[float] = None
    context_block_length: Optional[int] = None
    target_block_length: Optional[int] = None
    num_target_groups: Optional[int] = None
    min_context_ratio: Optional[float] = None


@dataclass(frozen=True)
class TrainingConfig:
    """Complete training configuration."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    masking: MaskingConfigOverride = field(default_factory=MaskingConfigOverride)

    seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dict for wandb config tracking."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        loss_data = data.get("loss", {})
        loss_filtered = {
            k: v for k, v in loss_data.items() if k in LossConfig.__dataclass_fields__
        }
        masking_data = data.get("masking", {})
        masking_filtered = {
            k: v
            for k, v in masking_data.items()
            if k in MaskingConfigOverride.__dataclass_fields__
        }

        return cls(
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            loss=LossConfig(**loss_filtered),
            checkpoint=CheckpointConfig(**data.get("checkpoint", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            data=DataConfig(**data.get("data", {})),
            precision=PrecisionConfig(**data.get("precision", {})),
            masking=MaskingConfigOverride(**masking_filtered),
            seed=data.get("seed", 42),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        loss_data = data.get("loss", {})
        loss_filtered = {
            k: v for k, v in loss_data.items() if k in LossConfig.__dataclass_fields__
        }
        masking_data = data.get("masking", {})
        masking_filtered = {
            k: v
            for k, v in masking_data.items()
            if k in MaskingConfigOverride.__dataclass_fields__
        }

        return cls(
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            loss=LossConfig(**loss_filtered),
            checkpoint=CheckpointConfig(**data.get("checkpoint", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            data=DataConfig(**data.get("data", {})),
            precision=PrecisionConfig(**data.get("precision", {})),
            masking=MaskingConfigOverride(**masking_filtered),
            seed=data.get("seed", 42),
        )
