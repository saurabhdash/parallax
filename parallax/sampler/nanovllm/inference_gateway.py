import atexit
import hashlib
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

import torch
import torch._dynamo
import torch.multiprocessing as mp

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams

RequestId = tuple[int, int]


@dataclass(slots=True)
class _Replica:
    id: int
    connection: Connection
    process: mp.Process
    active_requests: int = 0


class InferenceGateway:
    """Run one tensor-parallel engine in an isolated process per replica."""

    @staticmethod
    def _visible_devices() -> list[int]:
        return list(range(torch.cuda.device_count()))

    @staticmethod
    def _partition_devices(
        devices: list[int], num_replicas: int, tensor_parallel_size: int
    ) -> list[list[int]]:
        assert len(devices) >= num_replicas * tensor_parallel_size
        return [
            devices[i * tensor_parallel_size:(i + 1) * tensor_parallel_size]
            for i in range(num_replicas)
        ]

    @staticmethod
    def _replica_main(
        connection: Connection,
        model: str,
        tensor_parallel_size: int,
        replica_id: int,
        device_ids: list[int],
        config_kwargs: dict[str, Any],
    ):
        torch._dynamo.config.recompile_limit = 32
        engine = LLMEngine(
            model,
            tensor_parallel_size=tensor_parallel_size,
            replica_id=replica_id,
            num_replicas=config_kwargs.pop("num_replicas"),
            device_ids=device_ids,
            **config_kwargs,
        )
        connection.send(None)
        while True:
            method_name, args = connection.recv()
            if method_name == "exit":
                engine.exit()
                connection.send(None)
                return
            connection.send(getattr(engine, method_name)(*args))

    def __init__(
        self,
        model: str,
        *,
        num_replicas: int,
        tensor_parallel_size: int = 1,
        **kwargs,
    ):
        self.tensor_parallel_size = tensor_parallel_size
        self.learner_num_gpus = kwargs["learner_num_gpus"]
        device_groups = self._partition_devices(
            self._visible_devices(),
            num_replicas,
            tensor_parallel_size,
        )

        self.replicas: list[_Replica] = []
        ctx = mp.get_context("spawn")
        for replica_id in range(num_replicas):
            parent_connection, child_connection = ctx.Pipe()
            process = ctx.Process(
                target=self._replica_main,
                args=(
                    child_connection,
                    model,
                    tensor_parallel_size,
                    replica_id,
                    device_groups[replica_id],
                    {**kwargs, "num_replicas": num_replicas},
                ),
            )
            process.start()
            child_connection.close()
            self.replicas.append(_Replica(replica_id, parent_connection, process))
        for replica in self.replicas:
            replica.connection.recv() # wait for the replica to be ready
        atexit.register(self.exit)

    def exit(self):
        for replica in self.replicas:
            if replica.process.is_alive():
                replica.connection.send(("exit", ()))
        for replica in self.replicas:
            if replica.process.is_alive():
                replica.connection.recv()
                replica.process.join()

    def initialize_weight_transfer(self, unique_id: bytes) -> None:
        for replica in self.replicas:
            replica.connection.send(("initialize_weight_transfer", (unique_id,)))
        for replica in self.replicas:
            replica.connection.recv()

    def prepare_weight_update(
        self,
        manifest: list[dict[str, object]],
    ) -> None:
        for replica in self.replicas:
            replica.connection.send(("prepare_weight_update", (manifest,)))
        for replica in self.replicas:
            replica.connection.recv()

    def update_weights(self) -> None:
        for replica in self.replicas:
            replica.connection.send(("update_weights", ()))
        for replica in self.replicas:
            replica.connection.recv()

    def load_checkpoint(self, path: str) -> None:
        for replica in self.replicas:
            replica.connection.send(("load_checkpoint", (path,)))
        for replica in self.replicas:
            replica.connection.recv()

    def close_weight_transfer(self) -> None:
        for replica in self.replicas:
            replica.connection.send(("close_weight_transfer", ()))
        for replica in self.replicas:
            replica.connection.recv()

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        session_id: str | None = None,
    ) -> RequestId:
        replica = self._pick_replica(session_id)
        replica.connection.send(("add_request", (prompt, sampling_params)))
        sequence_id = replica.connection.recv()
        replica.active_requests += 1
        return replica.id, sequence_id

    def step(
        self,
    ) -> tuple[list[tuple[RequestId, list[int], list[float]]], int, int]:
        active_replicas = [replica for replica in self.replicas if replica.active_requests]
        for replica in active_replicas:
            replica.connection.send(("step", ()))

        outputs = []
        prefill_tokens = 0
        decode_tokens = 0
        for replica in active_replicas:
            completed, replica_prefill_tokens, replica_decode_tokens = (
                replica.connection.recv()
            )
            prefill_tokens += replica_prefill_tokens
            decode_tokens += replica_decode_tokens
            replica.active_requests -= len(completed)
            outputs.extend(
                ((replica.id, sequence_id), token_ids, logprobs)
                for sequence_id, token_ids, logprobs in completed
            )
        return outputs, prefill_tokens, decode_tokens

    def is_finished(self):
        return not any(replica.active_requests for replica in self.replicas)

    def _pick_replica(self, session_id: str | None) -> _Replica:
        if session_id is None:
            return min(self.replicas, key=lambda replica: replica.active_requests)
        digest = hashlib.blake2b(session_id.encode(), digest_size=8).digest()
        replica_id = int.from_bytes(digest, "little") % len(self.replicas)
        return self.replicas[replica_id]
