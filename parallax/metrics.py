"""Metric logging backends."""

from __future__ import annotations

from typing import Protocol

import wandb
from wandb.sdk.wandb_run import Run

from parallax.envs.base import RolloutGroup

class MetricsLogger(Protocol):
    """Accept scalar metrics for one learner update."""

    def log(self, metrics: dict[str, float], step: int) -> None: ...

    def log_eval_samples(self, groups: list[RolloutGroup], step: int) -> None: ...


class WandbMetricsLogger:
    """Log learner-update metrics to an existing Weights & Biases run."""

    def __init__(self, run: Run) -> None:
        self.run = run

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.run.log(metrics, step=step)

    def log_eval_samples(self, groups: list[RolloutGroup], step: int) -> None:
        """Log every sampled evaluation completion in a step-scoped table."""
        table = wandb.Table(
            columns=["learner_step", "example_index", "generation_index", "reward", "completion"],
            data=[
                [step, example_index, generation_index, reward, rollout.text]
                for example_index, group in enumerate(groups)
                for generation_index, (rollout, reward) in enumerate(
                    zip(group.rollouts, group.rewards, strict=True)
                )
            ],
        )
        self.run.log({"eval/samples": table}, step=step)
