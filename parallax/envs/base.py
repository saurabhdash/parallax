"""Interfaces for scored RL environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterator, TypeAlias

SamplerFn: TypeAlias = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class Rollout:
    text: str
    token_ids: list[int]
    train_mask: list[bool]
    sampler_logprobs: list[float]


@dataclass
class RolloutGroup:
    rollouts: list[Rollout]
    rewards: list[float]
    is_valid: bool = True


class Env:
    async def rollout(
        self,
        sample_fn: SamplerFn,
    ) -> Rollout:
        raise NotImplementedError

    def score(self, rollouts: list[Rollout]) -> RolloutGroup:
        raise NotImplementedError


class EnvDataset:
    def __iter__(self) -> Iterator[Env]:
        raise NotImplementedError
