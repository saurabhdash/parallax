"""Sampling configuration shared with the sampler client."""

from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        assert self.temperature > 1e-10
