"""Async RL rollout and training pipeline."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Iterable

from parallax.checkpoint import CheckpointWriter
from parallax.envs.base import Env
from parallax.envs.base import RolloutGroup
from parallax.evaluation import evaluate_envs
from parallax.evaluation import reward_metrics
from parallax.evaluation import rollout_group
from parallax.learner.algorithms import Algorithm
from parallax.learner.learner import Learner
from parallax.metrics import MetricsLogger
from parallax.runtime import RuntimeStats
from parallax.sampler.client import SamplerClient
from parallax.transport import WeightTransport
from parallax.utils.config import AsyncRLConfig


class AsyncRLOrchestrator:
    """Runs environments concurrently and feeds fixed rollout batches to training."""

    def __init__(
        self,
        envs: Iterable[Env],
        sampler: SamplerClient,
        algorithm: Algorithm,
        learner: Learner,
        transport: WeightTransport,
        config: AsyncRLConfig,
        metrics_logger: MetricsLogger | None = None,
        eval_envs: Iterable[Env] = (),
        max_steps: int | None = None,
        checkpoint_writer: CheckpointWriter | None = None,
    ) -> None:
        self.envs = iter(envs)
        self.eval_envs = eval_envs
        self.sampler = sampler
        self.sample_fn = partial(
            sampler.generate,
            sampling_params=config.sampling,
        )
        self.algorithm = algorithm
        self.learner = learner
        self.transport = transport
        self.config = config
        self.metrics_logger = metrics_logger
        self.max_steps = max_steps
        self.checkpoint_writer = checkpoint_writer
        self.update_step = 0
        self.runtime_stats = RuntimeStats()
        self._rollout_queue_closed = False
        self.rollout_queue: asyncio.Queue[RolloutGroup | None] = asyncio.Queue(
            maxsize=config.rollout_queue_capacity
        )

    async def run(self) -> None:
        async with asyncio.TaskGroup() as task_group:
            learner_ready = task_group.create_task(self.learner.initialize())
            await self.sampler.wait_until_ready()
            workers = [
                task_group.create_task(self._run_envs())
                for _ in range(self.config.num_env_workers)
            ]
            close_queue = task_group.create_task(
                self._close_rollout_queue(workers)
            )
            await self._train(learner_ready)
            if self.update_step == self.max_steps:
                for worker in workers:
                    worker.cancel()
                close_queue.cancel()
        if self.max_steps is not None and self.update_step < self.max_steps:
            raise RuntimeError(
                f"Training data exhausted at step {self.update_step}; "
                f"expected {self.max_steps} steps"
            )

    async def _run_envs(self) -> None:
        for env in self.envs:
            rollouts = await rollout_group(
                env.rollout,
                self.sample_fn,
                self.config.num_generations,
            )
            group = env.score(rollouts)
            await self.rollout_queue.put(group)

    async def _close_rollout_queue(
        self,
        workers: list[asyncio.Task[None]],
    ) -> None:
        await asyncio.gather(*workers)
        await self.rollout_queue.put(None)
        self._rollout_queue_closed = True

    async def _train(self, learner_ready: asyncio.Task[None]) -> None:
        # Evaluate the initial sampler policy while the learner initializes.
        if self.eval_envs:
            with self.runtime_stats.timer("evaluation"):
                eval_groups = await self._evaluate()
            if self.metrics_logger is not None:
                self.metrics_logger.log_eval_samples(eval_groups, step=0)
                self.metrics_logger.log(
                    reward_metrics("eval", eval_groups),
                    step=0,
                )
        await learner_ready
        self.runtime_stats.record_setup_complete()
        self.runtime_stats.set_phase("sampling")
        training_groups: list[RolloutGroup] = []
        sampled_groups: list[RolloutGroup] = []
        while (group := await self.rollout_queue.get()) is not None:
            sampled_groups.append(group)
            if not group.is_valid:
                continue
            training_groups.append(group)
            if len(training_groups) == self.config.num_groups_per_batch:
                await self._update(training_groups, sampled_groups)
                if self.update_step == self.max_steps:
                    return
                training_groups = []
                sampled_groups = []
        if training_groups:
            await self._update(training_groups, sampled_groups)

    async def _update(
        self,
        training_groups: list[RolloutGroup],
        sampled_groups: list[RolloutGroup],
    ) -> None:
        batch = self.algorithm.prepare(training_groups)
        with self.runtime_stats.timer("learner_update"):
            learner_metrics = await self.learner.update(batch)
        self.update_step += 1
        checkpoint_writer = self.checkpoint_writer
        if checkpoint_writer is not None:
            checkpoint_elapsed_seconds = checkpoint_writer.elapsed_seconds()
        with self.runtime_stats.timer("weight_transfer"):
            await self.transport.transfer(self.learner, self.sampler)
        self.runtime_stats.record_step()
        if checkpoint_writer is not None:
            with self.runtime_stats.timer("checkpoint"):
                await checkpoint_writer.save(
                    self.learner,
                    checkpoint_elapsed_seconds,
                )
        eval_groups = []
        if (
            self.eval_envs
            and self.update_step % self.config.eval_every_steps == 0
        ):
            with self.runtime_stats.timer("evaluation"):
                eval_groups = await self._evaluate()
            if self.metrics_logger is not None and eval_groups:
                self.metrics_logger.log_eval_samples(eval_groups, step=self.update_step)
        self.runtime_stats.set_phase("sampling")
        if self.metrics_logger is not None:
            self.metrics_logger.log(
                {
                    **reward_metrics("rollout", sampled_groups),
                    **reward_metrics("eval", eval_groups),
                    **{
                        f"runtime/{name}": value
                        for name, value in self.runtime_stats.step_metrics().items()
                    },
                    "train/clip_fraction": float(
                        learner_metrics["clip_fraction"]
                    ),
                    "train/grad_norm": float(learner_metrics["grad_norm"]),
                    "train/kl": float(learner_metrics["kl"]),
                    "train/mean_ratio": float(learner_metrics["mean_ratio"]),
                },
                step=self.update_step,
            )

    async def runtime_snapshot(self) -> dict[str, object]:
        rollout_queue_size = self.rollout_queue.qsize()
        if self._rollout_queue_closed and rollout_queue_size:
            rollout_queue_size -= 1
        return {
            **self.runtime_stats.snapshot(),
            **await self.sampler.stats(),
            "rollout_queue_size": rollout_queue_size,
            "client_inflight_requests": self.sampler.inflight_requests,
            "learner_step": self.update_step,
        }

    async def _evaluate(self) -> list[RolloutGroup]:
        return await evaluate_envs(
            self.eval_envs,
            self.sample_fn,
            self.config.num_eval_generations,
            self.config.num_eval_workers,
        )
