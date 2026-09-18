"""Launch sampling and learning on separate GPUs."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import httpx

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
    _signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    _signal_group(process, signal.SIGKILL)
    try:
        process.wait()
    except ChildProcessError:
        pass


def launch(config_path: Path) -> None:
    """Run the sampler and learner until either exits."""
    config = Config.from_toml(config_path)
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
        learner = subprocess.Popen(
            [
                LEARNER_PYTHON,
                "-m",
                "parallax.train",
                "--config",
                config_path,
            ],
            cwd=ROOT,
            env=learner_env,
            start_new_session=True,
        )
        response = httpx.post(
            f"http://{HOST}:{PORT}/initialize",
            json={
                "model": config.model.name,
                "num_replicas": config.sampler.num_replicas,
                "tensor_parallel_size": config.sampler.tensor_parallel_size,
                "max_model_len": config.sampler.max_model_len,
                "enable_mixed_rollouts": config.async_rl.enable_mixed_rollouts,
                "learner_num_gpus": config.learner.num_gpus,
            },
            timeout=600,
        )
        response.raise_for_status()

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
