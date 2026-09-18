"""Inference checkpoints used by speedrun evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import time

from transformers import AutoConfig
from transformers import PreTrainedTokenizerBase

from parallax.learner.learner import Learner
from parallax.learner.learner import WeightSnapshot
from parallax.learner.weights import save_hf_weights


class CheckpointWriter:
    """Write self-contained Hugging Face checkpoints atomically."""

    def __init__(
        self,
        directory: Path,
        model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        started_at: float,
    ) -> None:
        self.directory = directory
        self.model_name = model_name
        self.started_at = started_at
        self.model_files = directory / ".model"
        self.model_files.mkdir(parents=True)
        AutoConfig.from_pretrained(model_name).save_pretrained(self.model_files)
        tokenizer.save_pretrained(self.model_files)

    def elapsed_seconds(self) -> float:
        """Return setup-and-training time when an update completes."""
        return time.monotonic() - self.started_at

    async def save(
        self,
        learner: Learner,
        elapsed_seconds: float,
    ) -> None:
        async with learner.weights() as snapshot:
            await asyncio.to_thread(
                self._write,
                snapshot,
                elapsed_seconds,
            )

    def _write(
        self,
        snapshot: WeightSnapshot,
        elapsed_seconds: float,
    ) -> None:
        name = f"step-{snapshot.version:08d}"
        path = self.directory / name
        temporary_path = self.directory / f".{name}.tmp"
        save_hf_weights(snapshot.params, snapshot.config, temporary_path)
        shutil.copytree(self.model_files, temporary_path, dirs_exist_ok=True)
        metadata = {
            "step": snapshot.version,
            "elapsed_seconds": elapsed_seconds,
            "model": self.model_name,
        }
        (temporary_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.rename(path)
