"""RL algorithm interfaces shared by the orchestrator and learner."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from parallax.envs.base import RolloutGroup
from parallax.utils.config import AlgorithmConfig

Array = jax.Array | np.ndarray
# Pad batches to sequence lengths supported by cuDNN flash attention.
SEQUENCE_LENGTH_MULTIPLE = 64


def _kl3_with_kl2_gradient(log_ratio: jax.Array) -> jax.Array:
    kl2 = 0.5 * jnp.square(log_ratio)
    kl3 = jnp.exp(log_ratio) - log_ratio - 1.0
    return jax.lax.stop_gradient(kl3) + kl2 - jax.lax.stop_gradient(kl2)


@dataclass(slots=True)
class TrainingBatch:
    """The token-level data an algorithm sends from orchestrator to learner."""

    token_ids: Array  # [batch, sequence]
    attention_mask: Array  # [batch, sequence]
    train_mask: Array  # [batch, sequence - 1]
    sampler_logprobs: Array  # [batch, sequence - 1]
    advantages: Array  # [batch, sequence - 1]


class Algorithm:
    """A complete RL recipe: rollout preparation plus a differentiable loss."""

    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: AlgorithmConfig) -> Algorithm:
        assert config.name in ALGORITHMS
        return ALGORITHMS[config.name](config)

    def prepare(self, groups: list[RolloutGroup]) -> TrainingBatch:
        """Turn scored rollouts into a learner batch."""
        raise NotImplementedError

    def loss(
        self,
        learner_logprobs: jax.Array,
        batch: TrainingBatch,
    ) -> tuple[jax.Array, dict[str, jax.Array], jax.Array]:
        """Return a normalized loss, metrics, and loss normalizer."""
        raise NotImplementedError


class GRPO(Algorithm):
    """Group Relative Policy Optimization."""

    def prepare(self, groups: list[RolloutGroup]) -> TrainingBatch:
        rollouts = [
            rollout
            for group in groups
            for rollout in group.rollouts
        ]
        max_rollout_length = max(len(rollout.token_ids) for rollout in rollouts)
        multiple = SEQUENCE_LENGTH_MULTIPLE
        max_length = (
            (max_rollout_length + multiple - 1) // multiple * multiple
        )
        num_rollouts = len(rollouts)
        token_ids = np.zeros((num_rollouts, max_length), dtype=np.int32)
        attention_mask = np.zeros((num_rollouts, max_length), dtype=bool)
        train_mask = np.zeros((num_rollouts, max_length - 1), dtype=bool)
        sampler_logprobs = np.zeros(
            (num_rollouts, max_length - 1),
            dtype=np.float32,
        )
        advantages = np.zeros((num_rollouts, max_length - 1), dtype=np.float32)

        row = 0
        for group in groups:
            rewards = np.asarray(group.rewards, dtype=np.float32)
            group_advantages = (rewards - rewards.mean()) / (
                rewards.std() + self.config.advantage_epsilon
            )
            for rollout, advantage in zip(group.rollouts, group_advantages):
                length = len(rollout.token_ids)
                target_length = length - 1
                token_ids[row, :length] = rollout.token_ids
                attention_mask[row, :length] = True
                train_mask[row, :target_length] = rollout.train_mask
                sampler_logprobs[row, :target_length] = rollout.sampler_logprobs
                advantages[row, :target_length] = advantage
                row += 1

        return TrainingBatch(
            token_ids=token_ids,
            attention_mask=attention_mask,
            train_mask=train_mask,
            sampler_logprobs=sampler_logprobs,
            advantages=advantages,
        )

    def loss(
        self,
        learner_logprobs: jax.Array,
        batch: TrainingBatch,
    ) -> tuple[jax.Array, dict[str, jax.Array], jax.Array]:
        log_ratio = learner_logprobs - batch.sampler_logprobs
        ratio = jnp.exp(log_ratio)
        clipped_ratio = jnp.clip(
            ratio,
            1.0 - self.config.clip_epsilon_low,
            1.0 + self.config.clip_epsilon_high,
        )
        surrogate = jnp.minimum(
            ratio * batch.advantages,
            clipped_ratio * batch.advantages,
        )
        mask = batch.train_mask
        token_count = jnp.maximum(mask.sum(), 1.0)
        sequence_lengths = jnp.maximum(mask.sum(axis=-1), 1.0)
        sequence_count = jnp.asarray(mask.shape[0], dtype=jnp.float32)
        loss = -jnp.mean((surrogate * mask).sum(axis=-1) / sequence_lengths)
        kl = _kl3_with_kl2_gradient(log_ratio)
        clipped = ratio != clipped_ratio
        return (
            loss,
            {
                "clip_fraction": (clipped * mask).sum() / token_count,
                "kl": (kl * mask).sum() / token_count,
                "mean_ratio": (ratio * mask).sum() / token_count,
            },
            sequence_count,
        )


class CISPO(Algorithm):
    """Clipped Importance Sampling Policy Optimization."""

    def prepare(self, groups: list[RolloutGroup]) -> TrainingBatch:
        return GRPO.prepare(self, groups)

    def loss(
        self,
        learner_logprobs: jax.Array,
        batch: TrainingBatch,
    ) -> tuple[jax.Array, dict[str, jax.Array], jax.Array]:
        log_ratio = learner_logprobs - batch.sampler_logprobs
        ratio = jnp.exp(log_ratio)
        importance_weight = jax.lax.stop_gradient(
            jnp.minimum(ratio, self.config.importance_ratio_cap)
        )
        mask = batch.train_mask
        token_count = jnp.maximum(mask.sum(), 1.0)
        weighted_advantages = importance_weight * batch.advantages
        loss = -jnp.sum(weighted_advantages * learner_logprobs * mask) / token_count
        kl = _kl3_with_kl2_gradient(log_ratio)
        clipped = ratio > self.config.importance_ratio_cap

        return (
            loss,
            {
                "clip_fraction": (clipped * mask).sum() / token_count,
                "kl": (kl * mask).sum() / token_count,
                "mean_ratio": (ratio * mask).sum() / token_count,
            },
            token_count,
        )


ALGORITHMS: dict[str, type[Algorithm]] = {
    "grpo": GRPO,
    "cispo": CISPO,
}
