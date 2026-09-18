from __future__ import annotations

from dataclasses import dataclass
import jax.numpy as jnp
from transformers import AutoConfig


@dataclass(slots=True)
class LearnerModelConfig:
    num_layers: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    rope_theta: int
    norm_eps: float
    vocab_size: int
    embed_dim: int
    tie_word_embeddings: bool = False
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    use_flash_attention: bool = True

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        param_dtype: str,
    ) -> LearnerModelConfig:
        config = AutoConfig.from_pretrained(model)
        dtypes = {
            "bfloat16": jnp.bfloat16,
            "float16": jnp.float16,
            "float32": jnp.float32,
        }
        dtype_name = str(
            getattr(config, "dtype", None)
            or getattr(config, "torch_dtype", None)
            or "float32"
        ).removeprefix("torch.")
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            # Transformers 5 moved rope_theta into rope_parameters.
            rope_parameters = config.rope_parameters
            assert rope_parameters is not None
            rope_theta = rope_parameters["rope_theta"]
        return cls(
            num_layers=config.num_hidden_layers,
            hidden_dim=config.intermediate_size,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            norm_eps=config.rms_norm_eps,
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            num_experts=getattr(config, "num_experts", None),
            num_experts_per_tok=getattr(config, "num_experts_per_tok", None),
            dtype=dtypes[dtype_name],
            param_dtype=dtypes[param_dtype],
        )