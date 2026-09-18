"""NCCL M2N sender for JAX learner weights."""

from __future__ import annotations

import base64
import ctypes
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import jax
import nccl.core as nccl
from cuda.core import Device, system
from nccl import m2n

from parallax.learner.learner import Learner, WeightSnapshot
from parallax.learner.weights import (
    WeightTransferSpec,
    weight_transfer_specs,
    weight_transfer_values,
)


def _load_native_m2n():
    library_path = os.environ.get("NCCL_M2N_LIBRARY")
    if library_path is None:
        library_path = str(
            Path(m2n.__file__).parent / "lib/cu12/libnccl_m2n.so"
        )
    library = ctypes.CDLL(library_path)
    library.ncclReshard.argtypes = [ctypes.c_void_p] * 5 # Number of arguments and their types.
    library.ncclReshard.restype = ctypes.c_int # Return type.
    library.ncclM2nGetLastError.argtypes = []
    library.ncclM2nGetLastError.restype = ctypes.c_char_p
    return library


@contextmanager
def _cuda_device(rank: int) -> Iterator[Device]:
    previous_device = Device()
    device = Device(rank)
    device.set_current()
    try:
        yield device
    finally:
        if device.device_id != previous_device.device_id:
            previous_device.set_current()


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


def _source_axis(learner: Learner) -> str | None:
    fsdp_size = learner.config.fsdp_size
    tensor_parallel_size = learner.config.tensor_parallel_size
    if fsdp_size > 1 and tensor_parallel_size > 1:
        raise ValueError(
            "NCCL M2N supports one sharded learner mesh axis; "
            "configure either learner FSDP or learner tensor parallelism"
        )
    if fsdp_size > 1:
        return "fsdp"
    if tensor_parallel_size > 1:
        return "tp"
    return None


class M2NSender:
    """Own the learner ranks in a cross-mesh NCCL communicator."""

    def __init__(self, learner: Learner, bootstrap: dict[str, object]) -> None:
        self.learner = learner
        self.source_axis = _source_axis(learner)
        self.source_count = learner.config.num_gpus
        try:
            unique_id_bytes = base64.b64decode(
                str(bootstrap["unique_id"]),
                validate=True,
            )
            destination_replicas = int(bootstrap["num_replicas"])
            destination_tp_size = int(bootstrap["tensor_parallel_size"])
            bootstrap_source_count = int(bootstrap["learner_num_gpus"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid M2N bootstrap response") from error
        if bootstrap_source_count != self.source_count:
            raise ValueError(
                f"Sampler expects {bootstrap_source_count} learner GPUs, "
                f"but learner owns {self.source_count}"
            )
        device_count = system.get_num_devices()
        if device_count < self.source_count:
            raise RuntimeError(
                f"CUDA sees only {device_count} GPUs, "
                f"but the learner uses {self.source_count}"
            )

        self.destination_replicas = destination_replicas
        self.destination_tp_size = destination_tp_size
        self.source_mesh = m2n.Mesh([self.source_count], start_rank=0)
        self.destination_mesh = m2n.Mesh(
            [destination_replicas, destination_tp_size],
            start_rank=self.source_count,
        )
        unique_id = nccl.UniqueId.from_bytes(unique_id_bytes)
        world_size = self.source_count + destination_replicas * destination_tp_size
        self.communicators = []
        with nccl.group():
            # Each learner rank initializes a communicator with the rest of the world.
            for rank in range(self.source_count):
                with _cuda_device(rank):
                    self.communicators.append(
                        nccl.Communicator.init(
                            world_size,
                            rank,
                            unique_id,
                        )
                    )
        self.streams = []
        for rank in range(self.source_count):
            with _cuda_device(rank) as device:
                self.streams.append(device.create_stream()) # Create a stream for each rank to overlap computation and communication.
        self.handle = m2n.init()
        # IMPORTANT:
        # The python M2N wrapper assumes that each rank has it's own process (true for pytorch)
        # But JAX uses a single process for all ranks. So we need to load the native NCCL M2N library.
        # And make all the ranks enter reshard simultaneously using a threadpool executor.
        self.native_m2n = _load_native_m2n()
        self.executor = ThreadPoolExecutor(max_workers=self.source_count)
        self.devices = tuple(learner.mesh.devices.flat)

    def manifest(self) -> list[dict[str, object]]:
        return [
            spec.manifest_entry()
            for spec in weight_transfer_specs(
                self.learner.model_config,
                self.source_axis,
            )
        ]

    def send(self, snapshot: WeightSnapshot) -> None:
        # TODO: This can be made faster by streaming a subset of weights at a time at the cost of higher peak memory.
        for spec, weight in weight_transfer_values(
            snapshot.params,
            snapshot.config,
            self.source_axis,
        ):
            self._send_weight(spec, weight)

    def _send_weight(
        self,
        spec: WeightTransferSpec,
        weight: jax.Array,
    ) -> None:
        weight.block_until_ready() # Finish producing the weight buffer.
        shards = {shard.device: shard.data for shard in weight.addressable_shards}

        source_buffers = []
        for device in self.devices:
            buffer = shards[device]

            # Sanity check the weight shape and dtype before sending.
            expected_shape = _local_shape(
                spec.shape,
                spec.source_shard_dim,
                self.source_count,
            )
            if tuple(buffer.shape) != expected_shape:
                raise RuntimeError(
                    f"{spec.name} has local shape {tuple(buffer.shape)}, "
                    f"expected {expected_shape}"
                )
            if str(buffer.dtype) != spec.dtype:
                raise RuntimeError(
                    f"{spec.name} has dtype {buffer.dtype}, expected {spec.dtype}"
                )
            source_buffers.append(buffer)

        destination_shape = _local_shape(
            spec.shape,
            spec.destination_shard_dim,
            self.destination_tp_size,
        )
        if spec.source_shard_dim is None:
            source_placements = [m2n.Replicate()]
        else:
            source_placements = [m2n.Shard(spec.source_shard_dim)]
        if spec.destination_shard_dim is None:
            destination_placements = [m2n.Replicate(), m2n.Replicate()]
        else:
            destination_placements = [
                m2n.Replicate(),
                m2n.Shard(spec.destination_shard_dim),
            ]
        source_descriptors = [
            m2n.DistTensor(
                buffer.unsafe_buffer_pointer(),
                local_shape=buffer.shape,
                dtype=buffer.dtype,
                mesh=self.source_mesh,
                placements=source_placements,
            ).as_binding()
            for buffer in source_buffers
        ]
        destination_descriptors = [
            m2n.DistTensor(
                None,
                local_shape=destination_shape,
                dtype=spec.dtype,
                mesh=self.destination_mesh,
                placements=destination_placements,
            ).as_binding()
            for _ in source_buffers
        ]
        list(
            self.executor.map(
                self._native_reshard,
                range(self.source_count),
                source_descriptors,
                destination_descriptors,
            )
        ) # We need to use threadpool executor so that all the ranks enter reshard simultaneously.
        # Keep temporary JAX cast buffers alive until NCCL finishes reading them.
        for stream in self.streams:
            stream.sync()

    def _native_reshard(
        self,
        rank: int,
        source,
        destination,
    ) -> None:
        with _cuda_device(rank):
            result = self.native_m2n.ncclReshard(
                self.handle.ptr,
                self.communicators[rank].ptr,
                source.struct.ptr,
                destination.struct.ptr,
                int(self.streams[rank].handle),
            )
        if result != 0:
            detail = self.native_m2n.ncclM2nGetLastError().decode()
            raise RuntimeError(
                f"NCCL M2N reshard failed with result {result}: {detail}"
            )

    def close(self) -> None:
        self.executor.shutdown()
        for stream in self.streams:
            stream.sync()
        self.handle.destroy()
        with nccl.group():
            for communicator in self.communicators:
                communicator.finalize()
        for communicator in self.communicators:
            communicator.destroy()
        for stream in self.streams:
            stream.close()
