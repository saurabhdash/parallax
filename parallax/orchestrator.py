"""Async RL rollout and training pipeline."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Awaitable, Callable, Iterable

from parallax.envs.base import Env
from parallax.envs.base import Rollout
from parallax.envs.base import RolloutGroup
from parallax.envs.base import SamplerFn
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
            task_group.create_task(self._close_rollout_queue(workers))
            task_group.create_task(self._train(learner_ready))

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
                    _reward_metrics("eval", eval_groups),
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
        with self.runtime_stats.timer("weight_transfer"):
            await self.transport.transfer(self.learner, self.sampler)
        self.runtime_stats.record_step()
        eval_groups = []
        if self.update_step % self.config.eval_every_steps == 0:
            with self.runtime_stats.timer("evaluation"):
                eval_groups = await self._evaluate()
            if self.metrics_logger is not None and eval_groups:
                self.metrics_logger.log_eval_samples(eval_groups, step=self.update_step)
        self.runtime_stats.set_phase("sampling")
        if self.metrics_logger is not None:
            self.metrics_logger.log(
                {
                    **_reward_metrics("rollout", sampled_groups),
                    **_reward_metrics("eval", eval_groups),
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
        if not self.eval_envs:
            return []
        queue: asyncio.Queue[Env | None] = asyncio.Queue()
        for env in self.eval_envs:
            queue.put_nowait(env)
        for _ in range(self.config.num_eval_workers):
            queue.put_nowait(None)
        worker_groups = await asyncio.gather(
            *(
                self._eval_worker(queue)
                for _ in range(self.config.num_eval_workers)
            )
        )
        return [
            group
            for groups in worker_groups
            for group in groups
        ]

    async def _eval_worker(
        self,
        queue: asyncio.Queue[Env | None],
    ) -> list[RolloutGroup]:
        groups = []
        while (env := await queue.get()) is not None:
            rollouts = await rollout_group(
                env.rollout,
                self.sample_fn,
                self.config.num_eval_generations,
            )
            groups.append(env.score(rollouts))
        return groups


async def rollout_group(
    rollout: Callable[[SamplerFn], Awaitable[Rollout]],
    sample_fn: SamplerFn,
    num_generations: int,
) -> list[Rollout]:
    return await asyncio.gather(
        *(rollout(sample_fn) for _ in range(num_generations))
    )


def _reward_metrics(
    prefix: str,
    groups: list[RolloutGroup],
) -> dict[str, float]:
    if not groups:
        return {}
    rewards = [reward for group in groups for reward in group.rewards]
    group_size = len(groups[0].rewards)
    return {
        f"{prefix}/reward_mean": sum(rewards) / len(rewards),
        f"{prefix}/pass@1": sum(reward == 1.0 for reward in rewards)
        / len(rewards),
        f"{prefix}/pass@{group_size}": sum(
            any(reward == 1.0 for reward in group.rewards)
            for group in groups
        )
        / len(groups),
    }
