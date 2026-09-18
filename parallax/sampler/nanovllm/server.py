import asyncio
import base64
import os
from pathlib import Path

from fastapi import FastAPI
from huggingface_hub import snapshot_download
import nccl.core as nccl
from transformers import AutoTokenizer
import uvicorn

from nanovllm.inference_gateway import InferenceGateway
from nanovllm.sampling_params import SamplingParams


class InferenceServer:

    def __init__(self):
        self.gateway = None
        self.tokenizer = None
        self.weight_version = 0
        self.m2n_unique_id = None
        self.m2n_connected = False
        self.prepared_version = None
        self.prepared_manifest = None
        self.enable_mixed_rollouts = False
        self.queued_requests = 0
        self.decode_tokens = 0
        self.update_lock = asyncio.Lock()
        self.sampling_enabled = asyncio.Event()
        self.sampler_paused = asyncio.Event()
        self.sampling_enabled.set()
        self.queue = asyncio.Queue()
        self.pending = {}
        self.app = FastAPI()

        @self.app.post("/initialize")
        async def initialize(request: dict):
            await self.initialize(
                request["model"],
                request["num_replicas"],
                request.get("tensor_parallel_size", 1),
                request.get("max_model_len", 4096),
                request.get("enable_mixed_rollouts", False),
                request.get("learner_num_gpus", 1),
                request.get("enforce_eager", False),
            )
            return {"ok": True}

        @self.app.post("/generate")
        async def generate(request: dict):
            assert self.gateway is not None
            token_ids, logprobs = await self.submit(
                request["prompt"],
                SamplingParams(
                    request.get("temperature", 1.0),
                    request.get("max_tokens", 64),
                    request.get("ignore_eos", False),
                ),
                request.get("session_id"),
            )
            return {
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids,
                "logprobs": logprobs,
            }

        @self.app.get("/health")
        async def health():
            return {
                "ready": self.gateway is not None,
                "weight_version": self.weight_version,
            }

        @self.app.get("/stats")
        async def stats():
            return self.stats()

        @self.app.get("/m2n/bootstrap")
        async def m2n_bootstrap():
            return self.m2n_bootstrap()

        @self.app.post("/m2n/connect")
        async def connect_m2n():
            await self.connect_m2n()
            return {"ok": True}

        @self.app.post("/m2n/close")
        async def close_m2n():
            await self.close_m2n()
            return {"ok": True}

        @self.app.post("/prepare_weight_update")
        async def prepare_weight_update(request: dict):
            await self.prepare_weight_update(
                request["version"],
                request["manifest"],
            )
            return {"ok": True}

        @self.app.post("/update_weights")
        async def update_weights(request: dict):
            await self.update_weights(request["version"])
            return {"ok": True, "weight_version": self.weight_version}

        @self.app.post("/load_checkpoint")
        async def load_checkpoint(request: dict):
            await self.load_checkpoint(request["path"])
            return {"ok": True}

        @self.app.post("/teardown")
        async def teardown():
            await self.teardown()
            return {"ok": True}

        @self.app.on_event("shutdown")
        async def shutdown():
            await self.teardown()

    async def submit(self, prompt, sampling_params, session_id):
        future = asyncio.get_running_loop().create_future()
        self.queued_requests += 1
        self.queue.put_nowait((prompt, sampling_params, session_id, future))
        return await future

    def stats(self) -> dict[str, int]:
        return {
            "sampler_queued_requests": self.queued_requests,
            "sampler_active_requests": len(self.pending),
            "sampler_decode_tokens_total": self.decode_tokens,
        }

    async def scheduler(self):
        while True:
            # An update pauses at the next scheduler boundary.  A step already
            # in flight may complete in mixed-rollout mode; otherwise drain
            # active sequences before accepting the update.
            if not self.sampling_enabled.is_set() and (
                self.enable_mixed_rollouts or self.gateway.is_finished()
            ):
                self.sampler_paused.set()
                await self.sampling_enabled.wait()
                self.sampler_paused.clear()
                continue

            while self.sampling_enabled.is_set() and (
                self.gateway.is_finished() or not self.queue.empty()
            ):
                if self.gateway.is_finished():
                    request = await self.queue.get()
                else:
                    request = self.queue.get_nowait()
                if request is None:
                    continue
                if not self.sampling_enabled.is_set():
                    self.queue.put_nowait(request)
                    break
                self.queued_requests -= 1
                request_id = self.gateway.add_request(*request[:3])
                self.pending[request_id] = request[3]

            if self.gateway.is_finished():
                continue
            completed, _, decode_tokens = await asyncio.to_thread(
                self.gateway.step
            )
            self.decode_tokens += decode_tokens
            for request_id, token_ids, logprobs in completed:
                self.pending.pop(request_id).set_result((token_ids, logprobs))

    async def initialize(
        self,
        model,
        num_replicas,
        tensor_parallel_size=1,
        max_model_len=4096,
        enable_mixed_rollouts=False,
        learner_num_gpus=1,
        enforce_eager=False,
    ):
        assert self.gateway is None
        self.enable_mixed_rollouts = enable_mixed_rollouts
        local_model_path = Path(model).expanduser()
        model_path = (
            str(local_model_path.resolve())
            if local_model_path.is_dir()
            else await asyncio.to_thread(snapshot_download, repo_id=model)
        )
        self.m2n_unique_id = bytes(nccl.get_unique_id().as_bytes)
        self.gateway = await asyncio.to_thread(
            InferenceGateway,
            model_path,
            num_replicas=num_replicas,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            learner_num_gpus=learner_num_gpus,
            enforce_eager=enforce_eager,
        )
        self.tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained,
            model_path,
            use_fast=True,
        )
        self.scheduler_task = asyncio.create_task(self.scheduler())

    def m2n_bootstrap(self) -> dict[str, object]:
        if self.gateway is None or self.m2n_unique_id is None:
            raise RuntimeError("Sampler is not initialized")
        return {
            "unique_id": base64.b64encode(self.m2n_unique_id).decode(),
            "learner_num_gpus": self.gateway.learner_num_gpus,
            "num_replicas": len(self.gateway.replicas),
            "tensor_parallel_size": self.gateway.tensor_parallel_size,
        }

    async def connect_m2n(self) -> None:
        if self.gateway is None or self.m2n_unique_id is None:
            raise RuntimeError("Sampler is not initialized")
        if self.m2n_connected:
            raise RuntimeError("M2N is already connected")
        async with self.update_lock:
            await self.pause_sampling()
            try:
                await asyncio.to_thread(
                    self.gateway.initialize_weight_transfer,
                    self.m2n_unique_id,
                )
            finally:
                self.sampling_enabled.set()
        self.m2n_connected = True

    async def close_m2n(self) -> None:
        if self.gateway is None or not self.m2n_connected:
            return
        async with self.update_lock:
            await self.pause_sampling()
            try:
                await asyncio.to_thread(self.gateway.close_weight_transfer)
            finally:
                self.sampling_enabled.set()
        self.m2n_connected = False

    async def pause_sampling(self) -> None:
        self.sampler_paused.clear()
        self.sampling_enabled.clear()
        self.queue.put_nowait(None)
        await self.sampler_paused.wait()

    async def prepare_weight_update(
        self,
        version: int,
        manifest: list[dict[str, object]],
    ) -> None:
        if self.gateway is None or not self.m2n_connected:
            raise RuntimeError("M2N is not connected")
        await self.update_lock.acquire()
        try:
            if version <= self.weight_version:
                raise ValueError(
                    f"Weight version {version} is not newer than "
                    f"{self.weight_version}"
                )
            await self.pause_sampling()
            await asyncio.to_thread(
                self.gateway.prepare_weight_update,
                manifest,
            )
            self.prepared_version = version
            self.prepared_manifest = manifest
        except BaseException:
            self.sampling_enabled.set()
            self.update_lock.release()
            raise

    async def update_weights(self, version: int) -> None:
        if self.gateway is None:
            raise RuntimeError("Sampler is not initialized")
        if version != self.prepared_version or self.prepared_manifest is None:
            if self.prepared_manifest is not None:
                self.prepared_version = None
                self.prepared_manifest = None
                self.sampling_enabled.set()
                self.update_lock.release()
            raise ValueError(f"Weight version {version} was not prepared")
        try:
            await asyncio.to_thread(self.gateway.update_weights)
        except BaseException:
            self.prepared_version = None
            self.prepared_manifest = None
            self.update_lock.release()
            # A failed collective may have partially updated live parameters.
            # Keep sampling paused; NCCL M2N failures are fail-stop.
            raise
        self.weight_version = version
        self.prepared_version = None
        self.prepared_manifest = None
        self.sampling_enabled.set()
        self.update_lock.release()

    async def load_checkpoint(self, path: str) -> None:
        if self.gateway is None:
            raise RuntimeError("Sampler is not initialized")
        if self.m2n_connected:
            raise RuntimeError("Cannot load a checkpoint while M2N is connected")
        checkpoint = Path(path).resolve()
        if not checkpoint.is_dir():
            raise ValueError(f"Checkpoint does not exist: {checkpoint}")

        async with self.update_lock:
            await self.pause_sampling()
            try:
                await asyncio.to_thread(
                    self.gateway.load_checkpoint,
                    str(checkpoint),
                )
            finally:
                self.sampling_enabled.set()

    async def teardown(self):
        if self.gateway is None:
            return
        assert self.gateway.is_finished()
        self.scheduler_task.cancel()
        await asyncio.to_thread(self.gateway.exit)
        self.gateway = None
        self.tokenizer = None
        self.m2n_unique_id = None
        self.m2n_connected = False
        self.prepared_version = None
        self.prepared_manifest = None


def main() -> None:
    server = InferenceServer()
    port = int(os.environ.get("PARALLAX_SAMPLER_PORT", "8000"))
    uvicorn.run(server.app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
