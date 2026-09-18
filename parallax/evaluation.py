"""Shared rollout evaluation helpers."""

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterable

from parallax.envs.base import Env
from parallax.envs.base import Rollout
from parallax.envs.base import RolloutGroup
from parallax.envs.base import SamplerFn


async def rollout_group(
    rollout: Callable[[SamplerFn], Awaitable[Rollout]],
    sample_fn: SamplerFn,
    num_generations: int,
) -> list[Rollout]:
    return await asyncio.gather(
        *(rollout(sample_fn) for _ in range(num_generations))
    )


async def evaluate_envs(
    envs: Iterable[Env],
    sample_fn: SamplerFn,
    num_generations: int,
    num_workers: int,
) -> list[RolloutGroup]:
    queue: asyncio.Queue[Env | None] = asyncio.Queue()
    for env in envs:
        queue.put_nowait(env)
    for _ in range(num_workers):
        queue.put_nowait(None)
    worker_groups = await asyncio.gather(
        *(
            _eval_worker(queue, sample_fn, num_generations)
            for _ in range(num_workers)
        )
    )
    return [group for groups in worker_groups for group in groups]


async def _eval_worker(
    queue: asyncio.Queue[Env | None],
    sample_fn: SamplerFn,
    num_generations: int,
) -> list[RolloutGroup]:
    groups = []
    while (env := await queue.get()) is not None:
        rollouts = await rollout_group(
            env.rollout,
            sample_fn,
            num_generations,
        )
        groups.append(env.score(rollouts))
    return groups


def reward_metrics(
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
