"""Client for the sampler server."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx

from parallax.sampler.sampling_params import SamplingParams


class SamplerClient:
    def __init__(self, url: str) -> None:
        self.client = httpx.AsyncClient(
            base_url=url,
            timeout=600,
            limits=httpx.Limits(
                max_connections=512,
                max_keepalive_connections=0,
            ),
        )
        self.stats_client = httpx.AsyncClient(
            base_url=url,
            timeout=300,
            limits=httpx.Limits(max_connections=1),
        )
        self.inflight_requests = 0

    async def __aenter__(self) -> SamplerClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        self.inflight_requests += 1
        try:
            response = await self.client.post(
                "/generate",
                json={
                    "prompt": prompt,
                    "temperature": sampling_params.temperature,
                    "max_tokens": sampling_params.max_tokens,
                    "ignore_eos": sampling_params.ignore_eos,
                    "session_id": session_id,
                },
            )
            response.raise_for_status()
            return response.json()
        finally:
            self.inflight_requests -= 1

    async def update_weights(self, version: int) -> None:
        response = await self.client.post(
            "/update_weights",
            json={"version": version},
        )
        response.raise_for_status()

    async def load_checkpoint(self, path: Path) -> None:
        response = await self.client.post(
            "/load_checkpoint",
            json={"path": str(path.resolve())},
        )
        response.raise_for_status()

    async def m2n_bootstrap(self) -> dict[str, object]:
        response = await self.client.get("/m2n/bootstrap")
        response.raise_for_status()
        return response.json()

    async def connect_m2n(self) -> None:
        response = await self.client.post("/m2n/connect")
        response.raise_for_status()

    async def close_m2n(self) -> None:
        response = await self.client.post("/m2n/close")
        response.raise_for_status()

    async def prepare_weight_update(
        self,
        version: int,
        manifest: list[dict[str, object]],
    ) -> None:
        response = await self.client.post(
            "/prepare_weight_update",
            json={"version": version, "manifest": manifest},
        )
        response.raise_for_status()

    async def health(self) -> dict[str, Any]:
        response = await self.client.get("/health")
        response.raise_for_status()
        return response.json()

    async def stats(self) -> dict[str, int]:
        response = await self.stats_client.get("/stats")
        response.raise_for_status()
        return response.json()

    async def wait_until_ready(self) -> None:
        while True:
            if (await self.health())["ready"]:
                return
            await asyncio.sleep(0.1)

    async def close(self) -> None:
        await asyncio.gather(
            self.client.aclose(),
            self.stats_client.aclose(),
        )
