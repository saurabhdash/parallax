"""Learner-to-sampler weight transport."""

from __future__ import annotations

import asyncio
from typing import Protocol

from parallax.learner.learner import Learner
from parallax.learner.m2n import M2NSender
from parallax.sampler.client import SamplerClient


class WeightTransport(Protocol):
    async def transfer(
        self,
        learner: Learner,
        sampler: SamplerClient,
    ) -> None: ...


class M2NTransport:
    """Reshard JAX weights directly into sampler GPUs with NCCL M2N."""

    def __init__(self) -> None:
        self.sender: M2NSender | None = None

    async def _connect(
        self,
        learner: Learner,
        sampler: SamplerClient,
    ) -> None:
        bootstrap = await sampler.m2n_bootstrap() # Get the bootstrap data from the sampler for the sampler mesh.
        receiver = asyncio.create_task(sampler.connect_m2n()) # Build the receiver.
        sender = await asyncio.to_thread(M2NSender, learner, bootstrap)
        await receiver
        self.sender = sender

    async def transfer(
        self,
        learner: Learner,
        sampler: SamplerClient,
    ) -> None:
        if self.sender is None:
            await self._connect(learner, sampler)
        sender = self.sender
        if sender is None:
            raise RuntimeError("M2N sender failed to connect")

        async with learner.weights() as snapshot:
            await sampler.prepare_weight_update(
                snapshot.version,
                sender.manifest(),
            )
            await asyncio.gather(
                asyncio.to_thread(sender.send, snapshot),
                sampler.update_weights(snapshot.version),
            )

    async def close(self, sampler: SamplerClient) -> None:
        if self.sender is not None:
            await asyncio.gather(
                asyncio.to_thread(self.sender.close),
                sampler.close_m2n(),
            )
            self.sender = None
