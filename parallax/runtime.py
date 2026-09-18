"""Learner runtime state exposed to telemetry consumers."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
import math
import time
from typing import Any

Snapshot = dict[str, Any]


@dataclass(slots=True)
class RuntimeStats:
    """Mutable runtime state owned by the learner event loop."""

    phase: str = "initializing"
    phase_started_at: float = field(default_factory=time.time)
    wallclock_started_at: float = field(default_factory=time.perf_counter)
    wallclock_seconds: float | None = None
    setup_seconds: float | None = None
    durations: dict[str, float] = field(default_factory=dict)
    last_step_at: float | None = None
    inter_step_seconds: float | None = None
    inter_step_count: int = 0
    inter_step_mean_seconds: float = 0.0
    inter_step_m2: float = 0.0

    def record_setup_complete(self) -> None:
        self.setup_seconds = time.perf_counter() - self.wallclock_started_at

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self.phase_started_at = time.time()

    @contextmanager
    def timer(self, phase: str) -> Iterator[None]:
        self.set_phase(phase)
        started_at = self.phase_started_at
        try:
            yield
        finally:
            self.durations[f"{phase}_seconds"] = time.time() - started_at

    def record_step(self) -> None:
        now = time.perf_counter()
        self.wallclock_seconds = now - self.wallclock_started_at
        if self.last_step_at is not None:
            self.inter_step_seconds = now - self.last_step_at
            self.inter_step_count += 1
            delta = self.inter_step_seconds - self.inter_step_mean_seconds
            self.inter_step_mean_seconds += delta / self.inter_step_count
            self.inter_step_m2 += delta * (
                self.inter_step_seconds - self.inter_step_mean_seconds
            )
        self.last_step_at = now

    def inter_step_metrics(self) -> dict[str, float]:
        if self.inter_step_seconds is None:
            return {}
        return {
            "inter_step_seconds": self.inter_step_seconds,
            "inter_step_mean_seconds": self.inter_step_mean_seconds,
            "inter_step_std_seconds": math.sqrt(
                self.inter_step_m2 / self.inter_step_count
            ),
        }

    def step_metrics(self) -> dict[str, float]:
        metrics = self.inter_step_metrics()
        if self.wallclock_seconds is not None:
            metrics["wallclock_seconds"] = self.wallclock_seconds
        if self.setup_seconds is not None:
            metrics["setup_seconds"] = self.setup_seconds
        return metrics

    def snapshot(self) -> Snapshot:
        return {
            "phase": self.phase,
            "phase_started_at": self.phase_started_at,
            **self.durations,
            **self.step_metrics(),
        }
