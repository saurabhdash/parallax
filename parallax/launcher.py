"""Launch sampling and learning on separate GPUs."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import httpx

from parallax.speedrun import candidate_checkpoints
from parallax.speedrun import evaluate_speedrun
from parallax.utils.config import Config

HOST = "127.0.0.1"
PORT = 8000
ROOT = Path(__file__).resolve().parents[1]
LEARNER_PYTHON = ROOT / "parallax/learner/.venv/bin/python"
SAMPLER_PYTHON = ROOT / "parallax/sampler/.venv/bin/python"
LEARNER_MEMORY_FRACTION = "0.95"


def _visible_devices() -> list[str]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True,
    )
    return output.splitlines()


def _process_env(devices: list[str]) -> dict[str, str]:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(devices)}
    env.setdefault("NCCL_RESHARD_COPY_ALGORITHM", "DIRECT")
    return env


def _wait_for_sampler(process: subprocess.Popen[bytes]) -> None:
    deadline = time.time() + 30
    while process.poll() is None and time.time() < deadline:
        with socket.socket() as connection:
            if connection.connect_ex((HOST, PORT)) == 0:
                return
        time.sleep(0.1)
    assert process.poll() is None, "Sampler server exited during startup"
    raise TimeoutError("Sampler server did not start")


def _signal_group(
    process: subprocess.Popen[bytes],
    signal_number: signal.Signals,
) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        pass


def _stop(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        process.wait()


def _initialize_sampler(
    config: Config,
    model: str,
    *,
    num_replicas: int,
    enable_mixed_rollouts: bool,
    learner_num_gpus: int,
) -> None:
    response = httpx.post(
        f"http://{HOST}:{PORT}/initialize",
        json={
            "model": model,
            "num_replicas": num_replicas,
            "tensor_parallel_size": config.sampler.tensor_parallel_size,
            "max_model_len": config.sampler.max_model_len,
            "enable_mixed_rollouts": enable_mixed_rollouts,
            "learner_num_gpus": learner_num_gpus,
        },
        timeout=600,
    )
    response.raise_for_status()


def _run_speedrun_eval(
    config: Config,
    run_directory: Path,
    devices: list[str],
) -> None:
    checkpoint_directory = run_directory / "checkpoints"
    candidates = candidate_checkpoints(
        checkpoint_directory,
        config.speedrun.eval_last_n,
    )
    num_replicas = len(devices) // config.sampler.tensor_parallel_size
    sampler = subprocess.Popen(
        [SAMPLER_PYTHON, "-m", "nanovllm.server"],
        cwd=ROOT / "parallax/sampler",
        env=_process_env(devices),
        start_new_session=True,
    )
    try:
        _wait_for_sampler(sampler)
        _initialize_sampler(
            config,
            str(candidates[0]),
            num_replicas=num_replicas,
            enable_mixed_rollouts=False,
            learner_num_gpus=1,
        )
        asyncio.run(
            evaluate_speedrun(
                config,
                candidates,
                run_directory / "speedrun-result.json",
            )
        )
    finally:
        _stop(sampler)


def launch(config_path: Path, *, speedrun: bool = False) -> None:
    """Run the sampler and learner until either exits."""
    started_at = time.monotonic()
    config = Config.from_toml(config_path)
    run_directory = (
        ROOT
        / "runs"
        / f"speedrun-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        if speedrun
        else None
    )
    devices = _visible_devices()
    learner_count = config.learner.num_gpus
    sampler_count = (
        config.sampler.num_replicas
        * config.sampler.tensor_parallel_size
    )
    if config.learner.fsdp_size > 1 and config.learner.tensor_parallel_size > 1:
        raise ValueError("NCCL M2N supports one sharded learner mesh axis")
    assert learner_count + sampler_count <= len(devices)
    assert LEARNER_PYTHON.is_file()
    assert SAMPLER_PYTHON.is_file()

    learner_devices = devices[:learner_count]
    sampler_devices = devices[
        learner_count : learner_count + sampler_count
    ]
    sampler = subprocess.Popen(
        [SAMPLER_PYTHON, "-m", "nanovllm.server"],
        cwd=ROOT / "parallax/sampler",
        env=_process_env(sampler_devices),
        start_new_session=True,
    )

    learner = None
    try:
        _wait_for_sampler(sampler)
        learner_env = _process_env(learner_devices)
        learner_env.setdefault(
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            LEARNER_MEMORY_FRACTION,
        )
        learner_command = [
            str(LEARNER_PYTHON),
            "-m",
            "parallax.train",
            "--config",
            str(config_path),
        ]
        if run_directory is not None:
            learner_command.extend(
                [
                    "--speedrun",
                    "--run-directory",
                    str(run_directory),
                    "--started-at",
                    str(started_at),
                ]
            )
        learner = subprocess.Popen(
            learner_command,
            cwd=ROOT,
            env=learner_env,
            start_new_session=True,
        )
        _initialize_sampler(
            config,
            config.model.name,
            num_replicas=config.sampler.num_replicas,
            enable_mixed_rollouts=config.async_rl.enable_mixed_rollouts,
            learner_num_gpus=config.learner.num_gpus,
        )

        while sampler.poll() is None and learner.poll() is None:
            time.sleep(0.2)

        assert sampler.poll() is None, "Sampler server exited"
        assert learner.returncode == 0
    finally:
        try:
            if learner is not None:
                _stop(learner)
        finally:
            _stop(sampler)
    if run_directory is not None:
        _run_speedrun_eval(
            config,
            run_directory,
            devices,
        )
