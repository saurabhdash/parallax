"""Self-contained live dashboard for pipeline performance."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import time

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from pynvml import nvmlDeviceGetCount
from pynvml import nvmlDeviceGetHandleByIndex
from pynvml import nvmlDeviceGetMemoryInfo
from pynvml import nvmlDeviceGetUtilizationRates
from pynvml import nvmlInit
import uvicorn

from parallax.runtime import Snapshot

DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8001
SAMPLE_INTERVAL = 0.1
HISTORY_SIZE = int(60 * 60 / SAMPLE_INTERVAL)

SnapshotFn = Callable[[], Awaitable[Snapshot]]


class GpuMonitor:
    """Read utilization directly from NVIDIA's management library."""

    def __init__(self) -> None:
        nvmlInit()
        self.handles = [
            nvmlDeviceGetHandleByIndex(index)
            for index in range(nvmlDeviceGetCount())
        ]

    def utilization(self) -> list[int]:
        return [
            nvmlDeviceGetUtilizationRates(handle).gpu
            for handle in self.handles
        ]

    def memory_used(self) -> list[int]:
        return [
            nvmlDeviceGetMemoryInfo(handle).used // 2**20
            for handle in self.handles
        ]


class _DashboardServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


class Dashboard:
    """Collect, persist, and serve live runtime snapshots."""

    def __init__(
        self,
        snapshot: SnapshotFn,
        run_directory: Path,
        gpu_monitor: GpuMonitor,
    ) -> None:
        self.snapshot = snapshot
        self.run_directory = run_directory
        self.gpu_monitor = gpu_monitor
        self.history: deque[Snapshot] = deque(maxlen=HISTORY_SIZE)
        self.stopped = asyncio.Event()
        self.sample_ready = asyncio.Condition()
        self.sample_id = 0
        self.app = FastAPI()
        html = Path(__file__).with_name("dashboard.html").read_text(
            encoding="utf-8"
        )

        @self.app.get("/")
        async def index() -> HTMLResponse:
            return HTMLResponse(html)

        @self.app.get("/events")
        async def events() -> StreamingResponse:
            return StreamingResponse(
                self._events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        self.server = _DashboardServer(
            uvicorn.Config(
                self.app,
                host=DASHBOARD_HOST,
                port=DASHBOARD_PORT,
                access_log=False,
                log_level="warning",
            )
        )

    async def run(self) -> None:
        try:
            print(
                f"Dashboard running at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"
            )
            async with asyncio.TaskGroup() as task_group:
                task_group.create_task(self._collect())
                task_group.create_task(self.server.serve())
        except* (Exception, SystemExit) as errors:
            print(f"Dashboard stopped: {errors.exceptions[0]}")

    async def stop(self) -> None:
        self.stopped.set()
        async with self.sample_ready:
            self.sample_ready.notify_all()
        self.server.should_exit = True

    async def _collect(self) -> None:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        path = self.run_directory / "runtime.jsonl"
        previous_time = None
        previous_tokens = None
        with path.open("a", encoding="utf-8", buffering=1) as output:
            while not self.stopped.is_set():
                now = time.time()
                snapshot = await self.snapshot()
                decode_tokens = snapshot["sampler_decode_tokens_total"]
                tokens_per_second = (
                    0.0
                    if previous_time is None or previous_tokens is None
                    else (decode_tokens - previous_tokens)
                    / (now - previous_time)
                )
                snapshot.update(
                    {
                        "sample": self.sample_id,
                        "timestamp": now,
                        "phase_elapsed_seconds": (
                            now - snapshot["phase_started_at"]
                        ),
                        "sampler_tokens_per_second": tokens_per_second,
                        "gpu_utilization": self.gpu_monitor.utilization(),
                        "gpu_memory_used": self.gpu_monitor.memory_used(),
                    }
                )
                self.history.append(snapshot)
                output.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
                self.sample_id += 1
                async with self.sample_ready:
                    self.sample_ready.notify_all()
                previous_time = now
                previous_tokens = decode_tokens
                if not self.stopped.is_set():
                    await asyncio.sleep(SAMPLE_INTERVAL)

    async def _events(self) -> AsyncIterator[str]:
        history = tuple(self.history)
        last_sample = history[-1]["sample"] if history else -1
        initial = {
            "history": history,
            "history_size": HISTORY_SIZE,
            "sample_interval": SAMPLE_INTERVAL,
        }
        yield f"data:{json.dumps(initial, separators=(',', ':'))}\n\n"
        while not self.stopped.is_set():
            async with self.sample_ready:
                await self.sample_ready.wait_for(
                    lambda: (
                        self.stopped.is_set()
                        or (
                            bool(self.history)
                            and self.history[-1]["sample"] > last_sample
                        )
                    )
                )
            for snapshot in tuple(self.history):
                if snapshot["sample"] > last_sample:
                    last_sample = snapshot["sample"]
                    yield f"data:{json.dumps(snapshot, separators=(',', ':'))}\n\n"
