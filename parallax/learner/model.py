"""Learner model class."""

from __future__ import annotations

from typing import TypeAlias

import jax
import jax.numpy as jnp
import jaxtyping
from einops import einsum
from jax.sharding import PartitionSpec
from jax.sharding import PartitionSpec as P

from parallax.learner.configs import LearnerModelConfig

Params: TypeAlias = jaxtyping.PyTree[jax.Array]
PartitionSpecs: TypeAlias = jaxtyping.PyTree[PartitionSpec]


def _rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    x_float = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    normalized = x_float * jax.lax.rsqrt(variance + eps)
    return normalized.astype(x.dtype) * weight.astype(x.dtype)


def _add_rms_norm(
    x: jax.Array,
    residual: jax.Array,
    weight: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
    dtype = x.dtype
    x = x.astype(jnp.float32) + residual.astype(jnp.float32)
    residual = x.astype(dtype)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normalized = x * jax.lax.rsqrt(variance + eps)
    return normalized.astype(dtype) * weight.astype(dtype), residual


def _apply_rope(
    x: jax.Array,
    position_ids: jax.Array,
    rope_theta: int,
) -> jax.Array:
    """Apply rotary position embeddings to ``[batch, sequence, heads, dim]``."""
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE requires an even attention head dimension.")

    frequencies = 1.0 / (
        rope_theta
        ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    angles = position_ids[..., None].astype(jnp.float32) * frequencies
    cos = jnp.cos(angles)[..., None, :]
    sin = jnp.sin(angles)[..., None, :]
    first_half, second_half = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    rotated = jnp.concatenate(
        (
            first_half * cos - second_half * sin,
            second_half * cos + first_half * sin,
        ),
        axis=-1,
    )
    return rotated.astype(x.dtype)


def _eager_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_mask: jax.Array | None,
) -> jax.Array:
    repeats = q.shape[2] // k.shape[2]
    k = jnp.repeat(k, repeats, axis=2)
    v = jnp.repeat(v, repeats, axis=2)
    scores = einsum(q, k, "b q h d, b k h d -> b h q k") * (
        q.shape[-1] ** -0.5
    )
    sequence_length = q.shape[1]
    mask = jnp.tril(
        jnp.ones((sequence_length, sequence_length), dtype=bool)
    )[None, None, :, :]
    if attention_mask is not None:
        mask = mask & attention_mask[:, None, None, :].astype(bool)
    scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    probabilities = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(
        q.dtype
    )
    return einsum(probabilities, v, "b h q k, b k h d -> b q h d")


def _flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_mask: jax.Array | None,
) -> jax.Array:
    assert q.dtype in (jnp.bfloat16, jnp.float16)
    sequence_lengths = (
        None
        if attention_mask is None
        else jnp.sum(attention_mask, axis=-1, dtype=jnp.int32)
    )
    return jax.nn.dot_product_attention(
        q,
        k,
        v,
        scale=q.shape[-1] ** -0.5,
        is_causal=True,
        query_seq_lengths=sequence_lengths,
        key_value_seq_lengths=sequence_lengths,
        implementation="cudnn",
    )


class JaxAttention:
    """Grouped-query causal self-attention."""

    def __init__(self, config: LearnerModelConfig) -> None:
        self.config = config

    def setup(self, key: jax.Array) -> Params:
        """Create this attention block's parameter subtree."""
        cfg = self.config
        q_key, k_key, v_key, o_key = jax.random.split(key, 4)
        scale = cfg.embed_dim**-0.5

        def normal(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
            return jax.random.normal(key, shape, cfg.param_dtype) * scale

        return {
            "q": normal(q_key, (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
            "k": normal(k_key, (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
            "v": normal(v_key, (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
            "o": normal(o_key, (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
            "q_norm": jnp.ones((cfg.head_dim,), cfg.param_dtype),
            "k_norm": jnp.ones((cfg.head_dim,), cfg.param_dtype),
        }

    def partition_specs(self) -> PartitionSpecs:
        return {
            "q": P("fsdp", "tp", None),
            "k": P("fsdp", "tp", None),
            "v": P("fsdp", "tp", None),
            "o": P("tp", None, "fsdp"),
            "q_norm": P(),
            "k_norm": P(),
        }

    def __call__(
        self,
        params: Params,
        x: jax.Array,
        position_ids: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Apply causal attention to ``x`` with shape ``[batch, sequence, dim]``."""
        cfg = self.config
        q = einsum(
            x,
            params["q"].astype(cfg.dtype),
            "b s d, d h k -> b s h k",
        )
        k = einsum(
            x,
            params["k"].astype(cfg.dtype),
            "b s d, d h k -> b s h k",
        )
        v = einsum(
            x,
            params["v"].astype(cfg.dtype),
            "b s d, d h k -> b s h k",
        )
        q = jax.lax.with_sharding_constraint(
            q,
            P("fsdp", None, "tp", None),
        )
        k = jax.lax.with_sharding_constraint(
            k,
            P("fsdp", None, "tp", None),
        )
        v = jax.lax.with_sharding_constraint(
            v,
            P("fsdp", None, "tp", None),
        )
        q = _apply_rope(_rms_norm(q, params["q_norm"], cfg.norm_eps), position_ids, cfg.rope_theta)
        k = _apply_rope(_rms_norm(k, params["k_norm"], cfg.norm_eps), position_ids, cfg.rope_theta)

        assert cfg.num_heads % cfg.num_kv_heads == 0
        attended = (
            _flash_attention(q, k, v, attention_mask)
            if cfg.use_flash_attention
            else _eager_attention(q, k, v, attention_mask)
        )
        output = einsum(
            attended,
            params["o"].astype(cfg.dtype),
            "b s h d, h d o -> b s o",
        )
        return jax.lax.with_sharding_constraint(
            output,
            P("fsdp", None, None),
        )


class DenseFFN:
    """Dense SwiGLU feed-forward network."""

    def __init__(self, config: LearnerModelConfig) -> None:
        self.config = config

    def setup(self, key: jax.Array) -> Params:
        """Create this MLP's parameter subtree."""
        cfg = self.config
        gate_key, up_key, down_key = jax.random.split(key, 3)
        scale = cfg.embed_dim**-0.5

        def normal(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
            return jax.random.normal(key, shape, cfg.param_dtype) * scale

        return {
            "gate_proj": normal(gate_key, (cfg.embed_dim, cfg.hidden_dim)),
            "up_proj": normal(up_key, (cfg.embed_dim, cfg.hidden_dim)),
            "down_proj": normal(down_key, (cfg.hidden_dim, cfg.embed_dim)),
        }

    def partition_specs(self) -> PartitionSpecs:
        return {
            "gate_proj": P("fsdp", "tp"),
            "up_proj": P("fsdp", "tp"),
            "down_proj": P("tp", "fsdp"),
        }

    def __call__(self, params: Params, x: jax.Array) -> jax.Array:
        dtype = self.config.dtype
        x_up = einsum(
            x,
            params["up_proj"].astype(dtype),
            "b s d, d f -> b s f",
        )
        x_gate = einsum(
            x,
            params["gate_proj"].astype(dtype),
            "b s d, d f -> b s f",
        )
        x_up = jax.lax.with_sharding_constraint(
            x_up,
            P("fsdp", None, "tp"),
        )
        x_gate = jax.lax.with_sharding_constraint(
            x_gate,
            P("fsdp", None, "tp"),
        )
        x_act = jax.nn.silu(x_gate) * x_up
        output = einsum(
            x_act,
            params["down_proj"].astype(dtype),
            "b s f, f d -> b s d",
        )
        return jax.lax.with_sharding_constraint(
            output,
            P("fsdp", None, None),
        )


class DecoderBlock:
    """A pre-norm attention-and-MLP decoder block."""

    def __init__(self, config: LearnerModelConfig) -> None:
        self.config = config
        self.attention = JaxAttention(config)
        self.ffn = DenseFFN(config)

    def setup(self, key: jax.Array) -> Params:
        attention_key, ffn_key = jax.random.split(key)
        return {
            "attention_norm": jnp.ones(
                (self.config.embed_dim,),
                self.config.param_dtype,
            ),
            "attention": self.attention.setup(attention_key),
            "ffn_norm": jnp.ones(
                (self.config.embed_dim,),
                self.config.param_dtype,
            ),
            "ffn": self.ffn.setup(ffn_key),
        }

    def partition_specs(self) -> PartitionSpecs:
        return {
            "attention_norm": P(),
            "attention": self.attention.partition_specs(),
            "ffn_norm": P(),
            "ffn": self.ffn.partition_specs(),
        }

    def __call__(
        self,
        params: Params,
        x: jax.Array,
        residual: jax.Array,
        position_ids: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        x, residual = _add_rms_norm(
            x,
            residual,
            params["attention_norm"],
            self.config.norm_eps,
        )
        x = self.attention(
            params["attention"],
            x,
            position_ids,
            attention_mask,
        )
        x, residual = _add_rms_norm(
            x,
            residual,
            params["ffn_norm"],
            self.config.norm_eps,
        )
        x = self.ffn(
            params["ffn"],
            x,
        )
        return x, residual


class LearnerModel:
    """Qwen3 model with a scan over stacked decoder-block parameters."""

    def __init__(self, config: LearnerModelConfig) -> None:
        if config.num_experts is not None:
            raise NotImplementedError("MoE models are not implemented.")
        self.config = config
        self.block = DecoderBlock(config)

    def setup(self, key: jax.Array) -> Params:
        """Create the complete parameter pytree."""
        cfg = self.config
        embedding_key, output_key, layer_key = jax.random.split(key, 3)
        scale = cfg.embed_dim**-0.5
        token_embeddings = (
            jax.random.normal(
                embedding_key,
                (cfg.vocab_size, cfg.embed_dim),
                cfg.param_dtype,
            )
            * scale
        )
        layer_keys = jax.random.split(layer_key, cfg.num_layers)
        params = {
            "token_embeddings": token_embeddings,
            "layers": jax.vmap(self.block.setup)(layer_keys),
            "final_norm": jnp.ones((cfg.embed_dim,), cfg.param_dtype),
        }
        if not cfg.tie_word_embeddings:
            params["output"] = (
                jax.random.normal(
                    output_key,
                    (cfg.embed_dim, cfg.vocab_size),
                    cfg.param_dtype,
                )
                * scale
            )
        return params

    def partition_specs(self) -> PartitionSpecs:
        """Return a sharding tree with the same structure as the parameters."""
        layer_specs = jax.tree.map(
            lambda spec: P(None, *spec),
            self.block.partition_specs(),
        )
        specs = {
            "token_embeddings": P("tp", "fsdp"),
            "layers": layer_specs,
            "final_norm": P(),
        }
        if not self.config.tie_word_embeddings:
            specs["output"] = P("fsdp", "tp")
        return specs

    def __call__(
        self,
        params: Params,
        token_ids: jax.Array,
        position_ids: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Return next-token logits with shape ``[batch, sequence, vocab]``."""
        x = self.hidden_states(
            params,
            token_ids,
            position_ids,
            attention_mask,
        )
        # TODO: Consider performing the final unembedding in FP32.
        logits = einsum(
            x,
            self.output_weights(params),
            "b s d, d v -> b s v",
        )
        return jax.lax.with_sharding_constraint(
            logits,
            P("fsdp", None, "tp"),
        )

    def hidden_states(
        self,
        params: Params,
        token_ids: jax.Array,
        position_ids: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Return final hidden states without materializing vocabulary logits."""
        cfg = self.config
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(token_ids.shape[1]),
                token_ids.shape,
            )
        x = params["token_embeddings"][token_ids].astype(cfg.dtype)
        x = jax.lax.with_sharding_constraint(
            x,
            P("fsdp", None, None),
        )

        @jax.checkpoint
        def apply_block(
            carry: tuple[jax.Array, jax.Array],
            block_params: Params,
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            hidden_states, residual = carry
            return (
                self.block(
                    block_params,
                    hidden_states,
                    residual,
                    position_ids,
                    attention_mask,
                ),
                None,
            )

        (x, residual), _ = jax.lax.scan(
            apply_block,
            (x, jnp.zeros_like(x)),
            params["layers"],
        )
        x, _ = _add_rms_norm(
            x,
            residual,
            params["final_norm"],
            cfg.norm_eps,
        )
        return x

    def output_weights(self, params: Params) -> jax.Array:
        """Return the unembedding matrix with shape ``[hidden, vocab]``."""
        return (
            params["token_embeddings"].T
            if self.config.tie_word_embeddings
            else params["output"]
        ).astype(self.config.dtype)
