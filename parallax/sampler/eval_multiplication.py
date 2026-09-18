import argparse
import json
from math import comb
import random
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

import uvicorn
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from nanovllm.server import InferenceServer

ANSWER_RE = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")
URL = "http://127.0.0.1:8000"


def request(path, body):
    request = Request(
        f"{URL}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def wait_for_server():
    while True:
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=1):
                return
        except OSError:
            time.sleep(1)


def generate(prompt, session_id):
    return request("/generate", {
        "prompt": prompt,
        "temperature": 1.0,
        "max_tokens": 4096,
        "session_id": session_id,
    })["text"]


def parse_answer(text):
    match = ANSWER_RE.search(text)
    return int(match.group(1)) if match else None


def pass_at_k(correct, n, k):
    return 1 - comb(n - correct, k) / comb(n, k) if n - correct >= k else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digits", type=int, default=5)
    parser.add_argument("--questions", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--output", default="multiplication_eval.txt")
    args = parser.parse_args()

    rng = random.Random(0)
    low = 10 ** (args.digits - 1)
    high = 10 ** args.digits - 1
    problems = [(rng.randint(low, high), rng.randint(low, high)) for _ in range(args.questions)]
    model = snapshot_download(repo_id="Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model)
    inference_server = InferenceServer()
    server = uvicorn.Server(uvicorn.Config(inference_server.app, host="127.0.0.1", port=8000))
    thread = threading.Thread(target=server.run)
    thread.start()
    wait_for_server()
    request("/initialize", {
        "model": model,
        "num_replicas": 8,
        "tensor_parallel_size": 1,
        "max_model_len": 8192,
    })

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{a}x{b}=? Answer in \\boxed{{}}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for a, b in problems
    ]
    requests = [
        (prompt, f"problem-{i}")
        for i, prompt in enumerate(prompts)
        for _ in range(args.rollouts)
    ]
    with ThreadPoolExecutor(max_workers=min(64, len(requests))) as executor:
        outputs = list(executor.map(lambda request: generate(*request), requests))

    with open(args.output, "w") as f:
        f.write("question\treference\textracted\treward\n")
        for i, (a, b) in enumerate(problems):
            for j in range(args.rollouts):
                extracted = parse_answer(outputs[i * args.rollouts + j])
                reward = int(extracted == a * b)
                f.write(f"{a}x{b}\t{a * b}\t{extracted}\t{reward}\n")

    correct = [
        [parse_answer(outputs[i * args.rollouts + j]) == a * b for j in range(args.rollouts)]
        for i, (a, b) in enumerate(problems)
    ]
    pass_at_1 = sum(
        pass_at_k(sum(result), args.rollouts, 1) for result in correct
    ) / len(correct)
    pass_at_8 = sum(
        pass_at_k(sum(result), args.rollouts, args.rollouts) for result in correct
    ) / len(correct)
    print(f"pass@1: {pass_at_1:.3f}")
    print(f"pass@{args.rollouts}: {pass_at_8:.3f}")
    request("/teardown", {})
    server.should_exit = True
    thread.join()


if __name__ == "__main__":
    main()
