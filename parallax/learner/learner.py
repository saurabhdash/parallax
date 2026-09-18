"""JAX training loop for the learner model."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from einops import einsum
from huggingface_hub import snapshot_download
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from parallax.learner.algorithms import Algorithm
from parallax.learner.algorithms import TrainingBatch
from parallax.learner.configs import LearnerModelConfig
from parallax.learner.model import LearnerModel
from parallax.learner.model import Params
from parallax.learner.weights import load_hf_weights
from parallax.utils.config import LearnerConfig

VOCAB_CHUNK_SIZE = 4096


@dataclass(frozen=True, slots=True)
class TrainState:
    """Trainable parameters and optimizer state."""

    params: Params
    optimizer_state: optax.OptState
    step: jax.Array


@dataclass(frozen=True, slots=True)
class WeightSnapshot:
    """A learner parameter version held stable for transport."""

    version: int
    params: Params
    config: LearnerModelConfig


class Learner:
    """Owns one local model, optimizer, and compiled update function."""

    def __init__(
        self,
        model_name: str,
        algorithm: Algorithm,
        config: LearnerConfig,
    ) -> None:
        self.model_name = model_name
        self.algorithm = algorithm
        self.config = config
        self.state = None
        self._weight_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Load and shard model weights and initialize optimizer state."""
        self.state = await asyncio.to_thread(self._initialize)

    def _initialize(self) -> TrainState:
        self.model_config = LearnerModelConfig.from_pretrained(
            self.model_name,
            self.config.param_dtype,
        )
        self.model = LearnerModel(self.model_config)
        devices = np.asarray(jax.devices())
        assert devices.size == self.config.num_gpus
        assert self.model_config.embed_dim % self.config.fsdp_size == 0
        assert self.model_config.hidden_dim % self.config.tensor_parallel_size == 0
        assert self.model_config.num_heads % self.config.tensor_parallel_size == 0
        assert self.model_config.num_kv_heads % self.config.tensor_parallel_size == 0
        assert self.model_config.vocab_size % self.config.tensor_parallel_size == 0
        self.mesh = Mesh(
            devices.reshape(
                self.config.fsdp_size,
                self.config.tensor_parallel_size,
            ),
            ("fsdp", "tp"),
        )
        transforms = []
        if self.config.max_grad_norm is not None:
            transforms.append(
                optax.clip_by_global_norm(self.config.max_grad_norm)
            )
        transforms.append(
            optax.adamw(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        )
        self.optimizer = optax.chain(*transforms)
        self.param_shardings = jax.tree.map(
            lambda spec: NamedSharding(self.mesh, spec),
            self.model.partition_specs(),
        )
        self.batch_sharding = NamedSharding(
            self.mesh,
            P(None, "fsdp", None),
        )
        self._initialize_optimizer = jax.jit(self.optimizer.init)
        # TODO: Compile ahead of time from the configured batch shapes.
        self._compiled_update = jax.jit(
            self._update,
            donate_argnums=(0, 1),
        )

        model_path = snapshot_download(repo_id=self.model_name)
        host_params = load_hf_weights(model_path, self.model_config)
        assert jax.tree.structure(host_params) == jax.tree.structure(
            self.param_shardings
        )
        params = jax.tree.map(
            jax.device_put,
            host_params,
            self.param_shardings,
        )
        return TrainState(
            params=params,
            optimizer_state=self._initialize_optimizer(params),
            step=jnp.array(0, dtype=jnp.int32),
        )

    async def update(self, batch: TrainingBatch) -> dict[str, jax.Array]:
        """Update the learner without blocking the orchestrator event loop."""
        async with self._weight_lock:
            return await asyncio.to_thread(self._update_state, batch)

    @asynccontextmanager
    async def weights(self) -> AsyncIterator[WeightSnapshot]:
        """Hold the current parameter buffers stable during snapshot creation."""
        async with self._weight_lock:
            yield WeightSnapshot(
                version=int(jax.device_get(self.state.step)),
                params=self.state.params,
                config=self.model_config,
            )

    def _update_state(self, batch: TrainingBatch) -> dict[str, jax.Array]:
        assert self.state is not None
        state, metrics = self.step(self.state, batch)
        jax.block_until_ready(state.params) # Making sure that we are measuring the correct time for the update, not just dispatch.
        self.state = state
        return metrics

    def step(
        self,
        state: TrainState,
        batch: TrainingBatch,
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        """Apply one optimizer update to the RL batch."""
        self._validate_batch(batch)
        accumulation_steps = self.config.gradient_accumulation_steps
        batch_size = batch.token_ids.shape[0]
        assert batch_size % accumulation_steps == 0
        microbatch_size = batch_size // accumulation_steps
        assert microbatch_size % self.config.fsdp_size == 0

        def microbatches(
            array: jax.Array | np.ndarray,
        ) -> jax.Array | np.ndarray:
            return array.reshape(
                accumulation_steps,
                microbatch_size,
                *array.shape[1:],
            )

        batch = TrainingBatch(
            token_ids=jax.device_put(
                microbatches(batch.token_ids),
                self.batch_sharding,
            ),
            attention_mask=jax.device_put(
                microbatches(batch.attention_mask),
                self.batch_sharding,
            ),
            train_mask=jax.device_put(
                microbatches(batch.train_mask),
                self.batch_sharding,
            ),
            sampler_logprobs=jax.device_put(
                microbatches(batch.sampler_logprobs),
                self.batch_sharding,
            ),
            advantages=jax.device_put(
                microbatches(batch.advantages),
                self.batch_sharding,
            ),
        )
        with jax.set_mesh(self.mesh):
            params, optimizer_state, step, metrics = self._compiled_update(
                state.params,
                state.optimizer_state,
                state.step,
                batch.token_ids,
                batch.attention_mask,
                batch.train_mask,
                batch.sampler_logprobs,
                batch.advantages,
            )
        return (
            TrainState(
                params=params,
                optimizer_state=optimizer_state,
                step=step,
            ),
            metrics,
        )

    def _update(
        self,
        params: Params,
        optimizer_state: optax.OptState,
        step: jax.Array,
        token_ids: jax.Array,
        attention_mask: jax.Array,
        train_mask: jax.Array,
        sampler_logprobs: jax.Array,
        advantages: jax.Array,
    ) -> tuple[Params, optax.OptState, jax.Array, dict[str, jax.Array]]:
        def accumulate(
            gradient_sum: Params,
            arrays: tuple[jax.Array, ...],
        ):
            (
                microbatch_token_ids,
                microbatch_attention_mask,
                microbatch_train_mask,
                microbatch_sampler_logprobs,
                microbatch_advantages,
            ) = arrays
            batch = TrainingBatch(
                token_ids=microbatch_token_ids,
                attention_mask=microbatch_attention_mask,
                train_mask=(
                    microbatch_train_mask
                    * microbatch_attention_mask[:, 1:]
                ),
                sampler_logprobs=microbatch_sampler_logprobs,
                advantages=microbatch_advantages,
            )

            def loss_fn(current_params: Params):
                hidden_states = self.model.hidden_states(
                    current_params,
                    batch.token_ids,
                    attention_mask=batch.attention_mask,
                )
                learner_logprobs = _linear_logprobs(
                    hidden_states,
                    self.model.output_weights(current_params),
                    batch.token_ids,
                )
                loss, metrics, normalizer = self.algorithm.loss(
                    learner_logprobs,
                    batch,
                )
                metric_normalizer = batch.train_mask.sum()
                return (
                    loss * normalizer,
                    (metrics, normalizer, metric_normalizer),
                )

            (loss_sum, auxiliary), gradients = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            metrics, normalizer, metric_normalizer = auxiliary
            gradient_sum = jax.tree.map(
                jnp.add,
                gradient_sum,
                gradients,
            )
            return gradient_sum, (
                loss_sum,
                metrics,
                normalizer,
                metric_normalizer,
            )

        gradient_sum, outputs = jax.lax.scan(
            accumulate,
            jax.tree.map(jnp.zeros_like, params),
            (
                token_ids,
                attention_mask,
                train_mask,
                sampler_logprobs,
                advantages,
            ),
        )
        loss_sums, microbatch_metrics, normalizers, metric_normalizers = outputs
        normalizer = normalizers.sum()
        gradients = jax.tree.map(
            lambda gradient: gradient / normalizer,
            gradient_sum,
        )
        metric_normalizer = jnp.maximum(metric_normalizers.sum(), 1.0)
        metrics = jax.tree.map(
            lambda values: (
                values * metric_normalizers
            ).sum() / metric_normalizer,
            microbatch_metrics,
        )
        loss = loss_sums.sum() / normalizer
        updates, optimizer_state = self.optimizer.update(
            gradients,
            optimizer_state,
            params,
        )
        params = optax.apply_updates(params, updates)
        return params, optimizer_state, step + 1, {
            **metrics,
            "grad_norm": optax.global_norm(gradients),
            "loss": loss,
        }

    @staticmethod
    def _validate_batch(batch: TrainingBatch) -> None:
        assert batch.token_ids.ndim == 2
        assert batch.attention_mask.shape == batch.token_ids.shape
        assert batch.token_ids.shape[1] >= 2
        expected_shape = (
            batch.token_ids.shape[0],
            batch.token_ids.shape[1] - 1,
        )
        for value in (
            batch.train_mask,
            batch.sampler_logprobs,
            batch.advantages,
        ):
            assert value.shape == expected_shape


def _next_token_logprobs(logits: jax.Array, token_ids: jax.Array) -> jax.Array:
    """Return ``log p(token[t + 1] | token[:t + 1])`` for every sequence token."""
    next_token_logits = logits[:, :-1, :]
    next_token_ids = token_ids[:, 1:, None]
    logprobs = jax.nn.log_softmax(next_token_logits.astype(jnp.float32), axis=-1)
    return jnp.take_along_axis(logprobs, next_token_ids, axis=-1)[..., 0]

def _linear_logprobs(
    hidden_states: jax.Array,
    output_weights: jax.Array,
    token_ids: jax.Array,
    chunk_size: int = VOCAB_CHUNK_SIZE,
) -> jax.Array:
    """Compute next-token logprobs without materializing full vocabulary logits."""
    hidden_states = hidden_states[:, :-1]
    target_ids = token_ids[:, 1:]
    vocab_size = output_weights.shape[1]
    padding = -vocab_size % chunk_size
    output_weights = jnp.pad(output_weights, ((0, 0), (0, padding)))
    weight_chunks = output_weights.reshape(
        output_weights.shape[0],
        -1,
        chunk_size,
    ).transpose(1, 0, 2)
    chunk_starts = jnp.arange(weight_chunks.shape[0]) * chunk_size
    initial = (
        jnp.full(target_ids.shape, -jnp.inf, dtype=jnp.float32),
        jnp.full(target_ids.shape, -jnp.inf, dtype=jnp.float32),
    )

    def visit_chunk(carry, chunk):
        log_normalizer, target_logits = carry
        start, weights = chunk
        logits = einsum(
            hidden_states,
            weights,
            "b s d, d v -> b s v",
        ).astype(jnp.float32)
        logits = jnp.where(
            start + jnp.arange(chunk_size) < vocab_size,
            logits,
            -jnp.inf,
        )
        log_normalizer = jnp.logaddexp(
            log_normalizer,
            jax.nn.logsumexp(logits, axis=-1),
        )
        local_ids = jnp.clip(target_ids - start, 0, chunk_size - 1)
        chunk_targets = jnp.take_along_axis(
            logits,
            local_ids[..., None],
            axis=-1,
        )[..., 0]
        in_chunk = (target_ids >= start) & (target_ids < start + chunk_size)
        target_logits = jnp.where(in_chunk, chunk_targets, target_logits)
        return (log_normalizer, target_logits), None

    (log_normalizer, target_logits), _ = jax.lax.scan(
        jax.checkpoint(visit_chunk),
        initial,
        (chunk_starts, weight_chunks),
    )
    return target_logits - log_normalizer
