import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    replica_id: int = 0
    num_replicas: int = 1
    learner_num_gpus: int = 1
    device_ids: list[int] | None = None
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.device_ids is None:
            self.device_ids = list(range(self.tensor_parallel_size))
        assert len(self.device_ids) == self.tensor_parallel_size
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

    @property
    def distributed_init_method(self) -> str:
        port = int(os.environ.get("PARALLAX_DISTRIBUTED_PORT_BASE", "2333"))
        return f"tcp://127.0.0.1:{port + self.replica_id}"

    @property
    def shared_memory_name(self) -> str:
        return f"nanovllm-{self.replica_id}"
