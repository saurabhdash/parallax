"""Multiplication environment."""

import random
import re
from typing import Iterator

from transformers import PreTrainedTokenizerBase

from parallax.envs.base import Env
from parallax.envs.base import EnvDataset
from parallax.envs.base import Rollout
from parallax.envs.base import RolloutGroup
from parallax.envs.base import SamplerFn

ANSWER_PATTERN = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")
EVAL_SEED = 1
NUM_EVAL_EXAMPLES = 1024


def parse_answer(text: str) -> int | None:
    match = ANSWER_PATTERN.search(text)
    return int(match.group(1)) if match else None


class MultiplicationEnv(Env):
    def __init__(
        self,
        left: int,
        right: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.answer = left * right
        self.prompt_token_ids = list(
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": f"{left}x{right}=? Answer in \\boxed{{}}",
                    }
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        )

    async def rollout(
        self,
        sample_fn: SamplerFn,
    ) -> Rollout:
        output = await sample_fn(self.prompt_token_ids)
        completion_token_ids = output["token_ids"]
        prompt_targets = len(self.prompt_token_ids) - 1
        return Rollout(
            text=output["text"],
            token_ids=self.prompt_token_ids + completion_token_ids,
            train_mask=[False] * prompt_targets
            + [True] * len(completion_token_ids),
            sampler_logprobs=[0.0] * prompt_targets + output["logprobs"],
        )

    def score(self, rollouts: list[Rollout]) -> RolloutGroup:
        rewards = [
            float(parse_answer(rollout.text) == self.answer)
            for rollout in rollouts
        ]
        return RolloutGroup(rollouts, rewards, is_valid=len(set(rewards)) > 1)


class MultiplicationDataset(EnvDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_examples: int,
        digits: int = 5,
        seed: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.digits = digits
        self.seed = seed

    def __iter__(self) -> Iterator[Env]:
        held_out_problem_ids = (
            set(self._problems(EVAL_SEED, NUM_EVAL_EXAMPLES))
            if self.seed == 0
            else set()
        )
        for left, right in self._problems(
            self.seed,
            self.num_examples,
            held_out_problem_ids,
        ):
            yield MultiplicationEnv(
                left,
                right,
                self.tokenizer,
            )

    def _problems(
        self,
        seed: int,
        num_examples: int,
        excluded_problem_ids: set[tuple[int, int]] | None = None,
    ) -> Iterator[tuple[int, int]]:
        random_generator = random.Random(seed)
        low = 10 ** (self.digits - 1)
        high = 10**self.digits - 1
        problem_ids = set(excluded_problem_ids or ())
        target_num_problem_ids = len(problem_ids) + num_examples
        while len(problem_ids) < target_num_problem_ids:
            problem = tuple(sorted((
                random_generator.randint(low, high),
                random_generator.randint(low, high),
            )))
            if problem in problem_ids:
                continue
            problem_ids.add(problem)
            yield problem
