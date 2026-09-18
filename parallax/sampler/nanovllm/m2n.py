"""NCCL M2N receiver for live nano-vLLM weights."""

from __future__ import annotations

from dataclasses import dataclass

import nccl.core as nccl
import torch
from nccl import m2n

from nanovllm.config import Config
from nanovllm.models.qwen3 import Qwen3ForCausalLM


@dataclass(frozen=True, slots=True)
class WeightTarget:
    name: str
    tensor: torch.Tensor
    destination_shard_dim: int | None


@dataclass(frozen=True, slots=True)
class PreparedWeight:
    target: WeightTarget
    source_shard_dim: int | None
    source_local_shape: tuple[int, ...]


def _weight_targets(model: Qwen3ForCausalLM) -> list[WeightTarget]:
    """Return writable parameter views in learner transfer order."""
    targets = [
        WeightTarget(
            "model.embed_tokens.weight",
            model.model.embed_tokens.weight.data,
            0,
        )
    ]
    for layer_id, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{layer_id}"
        attention = layer.self_attn
        qkv = attention.qkv_proj.weight.data
        q_end = attention.q_size
        k_end = q_end + attention.kv_size
        ffn = layer.mlp
        gate_up = ffn.gate_up_proj.weight.data
        hidden_size = gate_up.shape[0] // 2
        targets.extend(
            [
                WeightTarget(
                    f"{prefix}.input_layernorm.weight",
                    layer.input_layernorm.weight.data,
                    None,
                ),
                WeightTarget(
                    f"{prefix}.self_attn.q_proj.weight",
                    qkv[:q_end],
                    0,
                ),
                WeightTarget(
                    f"{prefix}.self_attn.k_proj.weight",
                    qkv[q_end:k_end],
                    0,
                ),
                WeightTarget(
                    f"{prefix}.self_attn.v_proj.weight",
                    qkv[k_end:],
                    0,
                ),
                WeightTarget(
                    f"{prefix}.self_attn.o_proj.weight",
                    attention.o_proj.weight.data,
                    1,
                ),
                WeightTarget(
                    f"{prefix}.self_attn.q_norm.weight",
                    attention.q_norm.weight.data,
                    None,
                ),
                WeightTarget(
                    f"{prefix}.self_attn.k_norm.weight",
                    attention.k_norm.weight.data,
                    None,
                ),
                WeightTarget(
                    f"{prefix}.post_attention_layernorm.weight",
                    layer.post_attention_layernorm.weight.data,
                    None,
                ),
                WeightTarget(
                    f"{prefix}.mlp.gate_proj.weight",
                    gate_up[:hidden_size],
                    0,
                ),
                WeightTarget(
                    f"{prefix}.mlp.up_proj.weight",
                    gate_up[hidden_size:],
                    0,
                ),
                WeightTarget(
                    f"{prefix}.mlp.down_proj.weight",
                    ffn.down_proj.weight.data,
                    1,
                ),
            ]
        )
    targets.append(
        WeightTarget("model.norm.weight", model.model.norm.weight.data, None)
    )
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        targets.append(WeightTarget("lm_head.weight", model.lm_head.weight.data, 0))
    return targets


def _local_shape(
    global_shape: tuple[int, ...],
    shard_dim: int | None,
    shard_count: int,
) -> tuple[int, ...]:
    shape = list(global_shape)
    if shard_dim is not None:
        if shape[shard_dim] % shard_count:
            raise ValueError(
                f"Weight shape {global_shape} is not divisible by "
                f"{shard_count} along dimension {shard_dim}"
            )
        shape[shard_dim] //= shard_count
    return tuple(shape)


def _manifest_source_shard_dim(
    entry: dict[str, object],
    tensor_rank: int,
) -> int | None:
    source_shard_dim = entry.get("source_shard_dim")
    if source_shard_dim is None:
        return None
    if (
        isinstance(source_shard_dim, bool)
        or not isinstance(source_shard_dim, int)
        or not 0 <= source_shard_dim < tensor_rank
    ):
        raise ValueError(
            f"Invalid source shard dimension: {source_shard_dim!r}"
        )
    return source_shard_dim


class M2NReceiver:
    """Receive JAX learner shards directly into live PyTorch parameters."""

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        config: Config,
        rank: int,
    ) -> None:
        self.model = model
        self.config = config
        self.rank = rank
        self.communicator = None
        self.handle = None
        self.stream = None
        self.prepared_weights: list[PreparedWeight] | None = None
        self.source_mesh = m2n.Mesh([config.learner_num_gpus], start_rank=0)
        self.destination_mesh = m2n.Mesh(
            [config.num_replicas, config.tensor_parallel_size],
            start_rank=config.learner_num_gpus,
        )

    def connect(self, unique_id_bytes: bytes) -> None:
        if self.communicator is not None:
            raise RuntimeError("M2N receiver is already connected")
        unique_id = nccl.UniqueId.from_bytes(unique_id_bytes)
        destination_rank = (
            self.config.learner_num_gpus
            + self.config.replica_id * self.config.tensor_parallel_size
            + self.rank
        )
        world_size = (
            self.config.learner_num_gpus
            + self.config.num_replicas * self.config.tensor_parallel_size
        )
        self.communicator = nccl.Communicator.init(
            world_size,
            destination_rank,
            unique_id,
        )
        self.handle = m2n.init()
        self.stream = torch.cuda.Stream()

    def prepare(self, manifest: list[dict[str, object]]) -> None:
        if self.communicator is None:
            raise RuntimeError("M2N receiver is not connected")
        if self.prepared_weights is not None:
            raise RuntimeError("An M2N update is already prepared")
        targets = _weight_targets(self.model)
        if len(manifest) != len(targets):
            raise ValueError(
                f"Expected {len(targets)} weights, received {len(manifest)}"
            )
        prepared_weights = []
        for entry, target in zip(manifest, targets):
            try:
                shape = tuple(int(dim) for dim in entry["shape"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Invalid M2N weight manifest entry") from error
            source_shard_dim = _manifest_source_shard_dim(entry, len(shape))
            expected_local_shape = _local_shape(
                shape,
                target.destination_shard_dim,
                self.config.tensor_parallel_size,
            )
            expected_entry = {
                "name": target.name,
                "shape": list(shape),
                "dtype": str(target.tensor.dtype).removeprefix("torch."),
                "source_shard_dim": source_shard_dim,
            }
            if entry != expected_entry:
                raise ValueError(f"Invalid manifest entry for {target.name}: {entry!r}")
            if not target.tensor.is_contiguous():
                raise RuntimeError(f"M2N target {target.name} is not contiguous")
            if target.tensor.shape != expected_local_shape:
                raise ValueError(
                    f"M2N target {target.name} has local shape "
                    f"{tuple(target.tensor.shape)}, expected {expected_local_shape}"
                )
            prepared_weights.append(
                PreparedWeight(
                    target,
                    source_shard_dim,
                    _local_shape(
                        shape,
                        source_shard_dim,
                        self.config.learner_num_gpus,
                    ),
                )
            )
        torch.cuda.synchronize()
        self.prepared_weights = prepared_weights

    def receive(self) -> None:
        if self.communicator is None or self.handle is None or self.stream is None:
            raise RuntimeError("M2N receiver is not connected")
        if self.prepared_weights is None:
            raise RuntimeError("No M2N update has been prepared")
        prepared_weights = self.prepared_weights
        self.prepared_weights = None
        for prepared in prepared_weights:
            target = prepared.target
            source_shard_dim = prepared.source_shard_dim
            if source_shard_dim is None:
                source_placements = [m2n.Replicate()]
            else:
                source_placements = [m2n.Shard(source_shard_dim)]
            if target.destination_shard_dim is None:
                destination_placements = [m2n.Replicate(), m2n.Replicate()]
            else:
                destination_placements = [
                    m2n.Replicate(),
                    m2n.Shard(target.destination_shard_dim),
                ]
            m2n.reshard(
                None,
                target.tensor,
                self.communicator,
                stream=self.stream,
                src_mesh=self.source_mesh,
                src_placements=source_placements,
                src_local_shape=prepared.source_local_shape,
                src_dtype=target.tensor.dtype,
                dst_mesh=self.destination_mesh,
                dst_placements=destination_placements,
                handle=self.handle,
            ) # Directly use the python m2n wrapper as each rank is in a different process.
        self.stream.synchronize()

    def close(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()
        if self.handle is not None:
            self.handle.destroy()
            self.handle = None
        if self.communicator is not None:
            self.communicator.finalize()
            self.communicator.destroy()
            self.communicator = None
