"""Run configuration shared by the learner, sampler, and orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import tomllib

from parallax.sampler.sampling_params import SamplingParams


@dataclass(slots=True)
class ModelConfig:
    name: str


@dataclass(slots=True)
class LearnerConfig:
    fsdp_size: int
    tensor_parallel_size: int
    learning_rate: float
    param_dtype: str
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        assert self.fsdp_size > 0
        assert self.tensor_parallel_size > 0
        assert self.learning_rate > 0
        assert self.param_dtype in {"float32", "bfloat16", "float16"}

    @property
    def num_gpus(self) -> int:
        return self.fsdp_size * self.tensor_parallel_size


@dataclass(slots=True)
class SamplerConfig:
    num_replicas: int
    tensor_parallel_size: int
    max_model_len: int

    def __post_init__(self) -> None:
        assert self.num_replicas > 0
        assert self.tensor_parallel_size > 0
        assert self.max_model_len > 0


@dataclass(slots=True)
class AlgorithmConfig:
    name: str
    advantage_epsilon: float = 1e-8
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.28
    importance_ratio_cap: float = 5.0


@dataclass(slots=True)
class AsyncRLConfig:
    num_env_workers: int
    num_generations: int
    learner_batch_size: int
    rollout_queue_capacity: int
    sampling: SamplingParams
    num_eval_generations: int = 8
    num_eval_examples: int = 32
    num_eval_workers: int = 8
    eval_every_steps: int = 1
    enable_mixed_rollouts: bool = False

    @property
    def num_groups_per_batch(self) -> int:
        return self.learner_batch_size // self.num_generations


@dataclass(slots=True)
class TransportConfig:
    kind: str


@dataclass(slots=True)
class WandbConfig:
    project: str
    entity: str | None = None
    mode: str = "online"
    base_url: str | None = None

    def __post_init__(self) -> None:
        assert self.mode in {"online", "offline", "disabled"}


@dataclass(slots=True)
class SpeedrunConfig:
    steps: int = 100
    eval_last_n: int = 10
    target_pass_at_1: float = 0.9


@dataclass(slots=True)
class Config:
    model: ModelConfig
    learner: LearnerConfig
    sampler: SamplerConfig
    algorithm: AlgorithmConfig
    async_rl: AsyncRLConfig
    transport: TransportConfig | None = None
    wandb: WandbConfig | None = None
    speedrun: SpeedrunConfig = field(default_factory=SpeedrunConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        with open(path, "rb") as file:
            data = tomllib.load(file)

        transport = data.get("transport")
        wandb = data.get("wandb")
        speedrun = data.get("speedrun")
        async_rl = data["async_rl"]
        sampling = async_rl.pop("sampling")
        return cls(
            model=ModelConfig(**data["model"]),
            learner=LearnerConfig(**data["learner"]),
            sampler=SamplerConfig(**data["sampler"]),
            algorithm=AlgorithmConfig(**data["algorithm"]),
            async_rl=AsyncRLConfig(
                **async_rl,
                sampling=SamplingParams(**sampling),
            ),
            transport=TransportConfig(**transport) if transport else None,
            wandb=WandbConfig(**wandb) if wandb else None,
            speedrun=SpeedrunConfig(**(speedrun or {})),
        )
