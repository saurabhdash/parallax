"""Learner and orchestrator process."""

import argparse
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
import time

import wandb
from transformers import AutoTokenizer

from parallax.checkpoint import CheckpointWriter
from parallax.dashboard import Dashboard
from parallax.dashboard import GpuMonitor
from parallax.envs.multiplication import EVAL_SEED
from parallax.envs.multiplication import MultiplicationDataset
from parallax.learner.algorithms import Algorithm
from parallax.learner.learner import Learner
from parallax.metrics import WandbMetricsLogger
from parallax.orchestrator import AsyncRLOrchestrator
from parallax.sampler.client import SamplerClient
from parallax.transport import M2NTransport
from parallax.utils.config import Config


@asynccontextmanager
async def managed_transport(
    sampler: SamplerClient,
) -> AsyncIterator[M2NTransport]:
    transport = M2NTransport()
    try:
        yield transport
    finally:
        await transport.close(sampler)


@asynccontextmanager
async def running_dashboard(
    dashboard: Dashboard,
) -> AsyncIterator[None]:
    async with asyncio.TaskGroup() as task_group:
        task_group.create_task(dashboard.run())
        try:
            yield
        finally:
            await dashboard.stop()


async def train(
    config_path: str | Path,
    *,
    speedrun: bool = False,
    run_directory: Path | None = None,
    started_at: float | None = None,
) -> None:
    config = Config.from_toml(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    train_envs = MultiplicationDataset(tokenizer, digits=5, num_examples=10_000)
    eval_envs = (
        ()
        if speedrun
        else MultiplicationDataset(
            tokenizer,
            digits=5,
            num_examples=config.async_rl.num_eval_examples,
            seed=EVAL_SEED,
        )
    )
    algorithm = Algorithm.from_config(config.algorithm)
    learner = Learner(config.model.name, algorithm, config.learner)
    run = (
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            mode=config.wandb.mode,
            settings=(
                wandb.Settings(base_url=config.wandb.base_url)
                if config.wandb.base_url is not None
                else None
            ),
            config=asdict(config),
        )
        if config.wandb is not None
        else None
    )
    try:
        if run_directory is None:
            run_id = run.id if run is not None else str(int(time.time()))
            run_directory = Path("runs") / run_id
        checkpoint_writer = None
        if speedrun:
            if started_at is None:
                raise ValueError("Speedrun start time is required")
            checkpoint_writer = CheckpointWriter(
                run_directory / "checkpoints",
                config.model.name,
                tokenizer,
                started_at,
            )
        async with SamplerClient("http://127.0.0.1:8000") as sampler:
            async with managed_transport(sampler) as transport:
                orchestrator = AsyncRLOrchestrator(
                    train_envs,
                    sampler,
                    algorithm,
                    learner,
                    transport,
                    config.async_rl,
                    metrics_logger=WandbMetricsLogger(run) if run is not None else None,
                    eval_envs=eval_envs,
                    max_steps=config.speedrun.steps if speedrun else None,
                    checkpoint_writer=checkpoint_writer,
                )
                dashboard = Dashboard(
                    orchestrator.runtime_snapshot,
                    run_directory,
                    GpuMonitor(),
                )
                async with running_dashboard(dashboard):
                    await orchestrator.run()
    finally:
        if run is not None:
            run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--speedrun", action="store_true")
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--started-at", type=float)
    args = parser.parse_args()
    asyncio.run(
        train(
            args.config,
            speedrun=args.speedrun,
            run_directory=args.run_directory,
            started_at=args.started_at,
        )
    )


if __name__ == "__main__":
    main()
