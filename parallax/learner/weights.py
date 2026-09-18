"""Load Hugging Face Qwen3 weights into the learner parameter tree."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from parallax.learner.configs import LearnerModelConfig
from parallax.learner.model import LearnerModel
from parallax.learner.model import Params


@dataclass(frozen=True, slots=True)
class WeightTransferSpec:
    """Logical inference weight and its source/destination sharding."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    source_shard_dim: int | None
    destination_shard_dim: int | None

    def manifest_entry(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "source_shard_dim": self.source_shard_dim,
        }


def load_hf_weights(
    model_path: str | Path,
    config: LearnerModelConfig,
):
    weights = {}
    files = sorted(Path(model_path).glob("*.safetensors"))
    assert files
    for file in files:
        with safe_open(file, framework="numpy") as tensors:
            for name in tensors.keys():
                assert name not in weights
                weights[name] = tensors.get_tensor(name)
    return convert_from_hf_weights(weights, config)


def convert_from_hf_weights(
    hf_weights: Mapping[str, Any],
    config: LearnerModelConfig,
):
    """Convert dense Qwen3 weights to the learner's stacked parameter tree."""
    assert config.num_experts is None

    def array(name: str) -> np.ndarray:
        return np.asarray(hf_weights[name], dtype=config.param_dtype)

    def linear(name: str) -> np.ndarray:
        return array(name).T

    layer_ids = {
        int(name.split(".")[2])
        for name in hf_weights
        if name.startswith("model.layers.")
    }
    assert layer_ids == set(range(config.num_layers))
    layers = []

    for layer_id in range(config.num_layers):
        prefix = f"model.layers.{layer_id}"
        attn = f"{prefix}.self_attn"
        mlp = f"{prefix}.mlp"
        layers.append(
            {
                "attention_norm": array(f"{prefix}.input_layernorm.weight"),
                "attention": {
                    "q": linear(f"{attn}.q_proj.weight").reshape(
                        config.embed_dim,
                        config.num_heads,
                        config.head_dim,
                    ),
                    "k": linear(f"{attn}.k_proj.weight").reshape(
                        config.embed_dim,
                        config.num_kv_heads,
                        config.head_dim,
                    ),
                    "v": linear(f"{attn}.v_proj.weight").reshape(
                        config.embed_dim,
                        config.num_kv_heads,
                        config.head_dim,
                    ),
                    "o": linear(f"{attn}.o_proj.weight").reshape(
                        config.num_heads,
                        config.head_dim,
                        config.embed_dim,
                    ),
                    "q_norm": array(f"{attn}.q_norm.weight"),
                    "k_norm": array(f"{attn}.k_norm.weight"),
                },
                "ffn_norm": array(f"{prefix}.post_attention_layernorm.weight"),
                "ffn": {
                    "gate_proj": linear(f"{mlp}.gate_proj.weight"),
                    "up_proj": linear(f"{mlp}.up_proj.weight"),
                    "down_proj": linear(f"{mlp}.down_proj.weight"),
                },
            }
        )

    params = {
        "token_embeddings": array("model.embed_tokens.weight"),
        "layers": jax.tree.map(lambda *weights: np.stack(weights), *layers),
        "final_norm": array("model.norm.weight"),
    }
    if not config.tie_word_embeddings:
        params["output"] = linear("lm_head.weight")

    expected = jax.eval_shape(LearnerModel(config).setup, jax.random.key(0))
    assert jax.tree.structure(params) == jax.tree.structure(expected)
    assert all(
        weight.shape == expected_weight.shape
        for weight, expected_weight in zip(
            jax.tree.leaves(params),
            jax.tree.leaves(expected),
        )
    )
    return params


def save_hf_weights(
    params: Params,
    config: LearnerModelConfig,
    path: Path,
) -> None:
    """Write one tensor per file to bound host staging memory."""
    assert not path.exists()
    path.mkdir(parents=True)

    for index, (name, weight) in enumerate(_hf_weights(params, config)):
        host_weight = np.ascontiguousarray(
            jax.device_get(weight.astype(config.dtype))
        )
        save_file(
            {name: host_weight},
            path / f"{index:05d}.safetensors",
        )


def weight_transfer_specs(
    config: LearnerModelConfig,
    source_axis: str | None,
) -> list[WeightTransferSpec]:
    """Describe per-weight M2N transfers in Hugging Face layout."""
    if source_axis not in {None, "fsdp", "tp"}:
        raise ValueError(f"Unsupported learner source axis: {source_axis}")

    dtype = str(np.dtype(config.dtype))
    # TODO: This is kinda unsafe and should be replaced with a per-weight spec.
    def spec(
        name: str,
        shape: tuple[int, ...],
        destination_shard_dim: int | None,
    ) -> WeightTransferSpec:
        if destination_shard_dim is None or source_axis is None:
            source_shard_dim = None
        elif source_axis == "tp":
            source_shard_dim = destination_shard_dim
        else:
            source_shard_dim = 1 - destination_shard_dim
        return WeightTransferSpec(
            name=name,
            shape=shape,
            dtype=dtype,
            source_shard_dim=source_shard_dim,
            destination_shard_dim=destination_shard_dim,
        )

    embed_dim = config.embed_dim
    attention_dim = config.num_heads * config.head_dim
    kv_dim = config.num_kv_heads * config.head_dim
    hidden_dim = config.hidden_dim
    specs = [
        spec(
            "model.embed_tokens.weight",
            (config.vocab_size, embed_dim),
            0,
        )
    ]
    for layer_id in range(config.num_layers):
        prefix = f"model.layers.{layer_id}"
        attn = f"{prefix}.self_attn"
        mlp = f"{prefix}.mlp"
        specs.extend(
            [
                spec(f"{prefix}.input_layernorm.weight", (embed_dim,), None),
                spec(f"{attn}.q_proj.weight", (attention_dim, embed_dim), 0),
                spec(f"{attn}.k_proj.weight", (kv_dim, embed_dim), 0),
                spec(f"{attn}.v_proj.weight", (kv_dim, embed_dim), 0),
                spec(f"{attn}.o_proj.weight", (embed_dim, attention_dim), 1),
                spec(f"{attn}.q_norm.weight", (config.head_dim,), None),
                spec(f"{attn}.k_norm.weight", (config.head_dim,), None),
                spec(
                    f"{prefix}.post_attention_layernorm.weight",
                    (embed_dim,),
                    None,
                ),
                spec(f"{mlp}.gate_proj.weight", (hidden_dim, embed_dim), 0),
                spec(f"{mlp}.up_proj.weight", (hidden_dim, embed_dim), 0),
                spec(f"{mlp}.down_proj.weight", (embed_dim, hidden_dim), 1),
            ]
        )
    specs.append(spec("model.norm.weight", (embed_dim,), None))
    if not config.tie_word_embeddings:
        specs.append(spec("lm_head.weight", (config.vocab_size, embed_dim), 0))
    return specs


def weight_transfer_values(
    params: Params,
    config: LearnerModelConfig,
    source_axis: str | None,
):
    """Yield M2N specs and GPU-resident values in inference layout."""
    specs = weight_transfer_specs(config, source_axis)
    weights = _hf_weights(params, config)
    for spec, (name, weight) in zip(specs, weights, strict=True):
        if name != spec.name or weight.shape != spec.shape:
            raise RuntimeError(f"Transfer layout mismatch for {spec.name}: got {name} with shape {weight.shape}")
        yield spec, weight.astype(config.dtype) # cast to the sampler dtype.


def _hf_weights(params: Params, config: LearnerModelConfig):
    """Yield learner parameters in Hugging Face names and layouts."""
    yield "model.embed_tokens.weight", params["token_embeddings"]

    for layer_id in range(config.num_layers):
        prefix = f"model.layers.{layer_id}"
        layer = jax.tree.map(lambda weight, i=layer_id: weight[i], params["layers"])
        attention = layer["attention"]
        ffn = layer["ffn"]

        yield f"{prefix}.input_layernorm.weight", layer["attention_norm"]
        yield f"{prefix}.self_attn.q_proj.weight", attention["q"].reshape(
            config.embed_dim,
            -1,
        ).T
        yield f"{prefix}.self_attn.k_proj.weight", attention["k"].reshape(
            config.embed_dim,
            -1,
        ).T
        yield f"{prefix}.self_attn.v_proj.weight", attention["v"].reshape(
            config.embed_dim,
            -1,
        ).T
        yield f"{prefix}.self_attn.o_proj.weight", attention["o"].reshape(
            -1,
            config.embed_dim,
        ).T
        yield f"{prefix}.self_attn.q_norm.weight", attention["q_norm"]
        yield f"{prefix}.self_attn.k_norm.weight", attention["k_norm"]
        yield f"{prefix}.post_attention_layernorm.weight", layer["ffn_norm"]
        yield f"{prefix}.mlp.gate_proj.weight", ffn["gate_proj"].T
        yield f"{prefix}.mlp.up_proj.weight", ffn["up_proj"].T
        yield f"{prefix}.mlp.down_proj.weight", ffn["down_proj"].T

    yield "model.norm.weight", params["final_norm"]
    if not config.tie_word_embeddings:
        yield "lm_head.weight", params["output"].T
