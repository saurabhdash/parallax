import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

import uvicorn
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from nanovllm.server import InferenceServer

HOST = "127.0.0.1"
PORT = 8000
NUM_REPLICAS = 2
TENSOR_PARALLEL_SIZE = 1


def wait_for_server():
    while True:
        try:
            with socket.create_connection((HOST, PORT), timeout=1):
                return
        except OSError:
            time.sleep(1)


def request(path, body):
    request = Request(
        f"http://{HOST}:{PORT}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def generate(i, prompt):
    return request("/generate", {
        "prompt": prompt,
        "temperature": 0.6,
        "max_tokens": 5,
        "session_id": f"session-{i}",
    })


def main():
    model = snapshot_download(repo_id="Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model)
    inference_server = InferenceServer()
    server = uvicorn.Server(uvicorn.Config(inference_server.app, host=HOST, port=PORT))
    thread = threading.Thread(target=server.run)
    thread.start()
    wait_for_server()
    request("/initialize", {
        "model": model,
        "num_replicas": NUM_REPLICAS,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
    })

    questions = [f"Give me an interesting fact about the number {i}." for i in range(4)]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for question in questions
    ]
    with ThreadPoolExecutor(max_workers=8) as executor:
        outputs = executor.map(generate, range(len(prompts)), prompts)
        for question, output in zip(questions, outputs):
            print(f"{question}\n{output['text']}\n")
            print(f"Logprobs: {output['logprobs']}\n")

    request("/teardown", {})
    server.should_exit = True
    thread.join()


if __name__ == "__main__":
    main()
