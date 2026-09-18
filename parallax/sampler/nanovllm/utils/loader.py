import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight, non_blocking=True)


def load_model(model: nn.Module, path: str, pin_memory: bool = False):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    assert files
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)
                if pin_memory:
                    # Bound extra host memory to one pinned tensor at a time.
                    loaded_weight = loaded_weight.pin_memory()
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, loaded_weight, shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
