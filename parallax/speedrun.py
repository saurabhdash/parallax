"""Post-training checkpoint evaluation for speedruns."""

from __future__ import annotations

from functools import partial
import json
from pathlib import Path

from transformers import AutoTokenizer

from parallax.envs.multiplication import EVAL_SEED
from parallax.envs.multiplication import MultiplicationDataset
from parallax.evaluation import evaluate_envs
from parallax.evaluation import reward_metrics
from parallax.sampler.client import SamplerClient
from parallax.utils.config import Config


def checkpoint_metadata(path: Path) -> dict[str, object]:
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def candidate_checkpoints(directory: Path, last_n: int) -> list[Path]:
    checkpoints = sorted(
        path
        for path in directory.glob("step-*")
        if path.is_dir()
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {directory}")
    return checkpoints[-last_n:]


async def evaluate_speedrun(
    config: Config,
    candidates: list[Path],
    output_path: Path,
) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(candidates[0])
    eval_envs = MultiplicationDataset(
        tokenizer,
        digits=5,
        num_examples=config.async_rl.num_eval_examples,
        seed=EVAL_SEED,
    )
    results = []
    selected = None

    async with SamplerClient("http://127.0.0.1:8000") as sampler:
        await sampler.wait_until_ready()
        sample_fn = partial(
            sampler.generate,
            sampling_params=config.async_rl.sampling,
        )
        for index, checkpoint in enumerate(candidates):
            if index:
                await sampler.load_checkpoint(checkpoint)
            groups = await evaluate_envs(
                eval_envs,
                sample_fn,
                config.async_rl.num_eval_generations,
                config.async_rl.num_eval_workers,
            )
            metadata = checkpoint_metadata(checkpoint)
            metrics = reward_metrics("eval", groups)
            pass_at_1 = metrics["eval/pass@1"]
            result = {
                "checkpoint": str(checkpoint),
                "step": int(metadata["step"]),
                "pass_at_1": pass_at_1,
                "time_to_checkpoint_seconds": float(
                    metadata["elapsed_seconds"]
                ),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            if pass_at_1 >= config.speedrun.target_pass_at_1:
                selected = result
                break

    summary = {
        "selected": selected,
        "target_pass_at_1": config.speedrun.target_pass_at_1,
        "evaluated": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if selected is None:
        print(
            f"\033[1;31mTarget pass@1 "
            f"{config.speedrun.target_pass_at_1:.0%} not reached.\033[0m"
        )
    else:
        print(
            f"\033[1;32mTime to target: "
            f"{selected['time_to_checkpoint_seconds']:.1f}s "
            f"(step {selected['step']}, pass@1 "
            f"{selected['pass_at_1']:.3f}).\033[0m"
        )
    return summary
