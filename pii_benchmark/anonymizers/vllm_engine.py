"""Process-wide cache of vLLM engines, so anonymizers can share one.

A vLLM ``LLM`` reserves ``gpu_memory_utilization`` of the *whole* device up
front and holds it for the engine's lifetime. That fraction is of total
memory, not of what is free, so two engines built at the default 0.6 can
never coexist (0.6 + 0.6 > 1.0) however large the card is -- the second one
dies with "Free memory on device cuda:0 ... is less than desired GPU memory
utilization". A run that mixes llama_basic with llama_rescriber asks for
exactly that, and both want the same checkpoint anyway, so they take one
engine from here instead of each constructing their own.
"""

import os
from typing import Dict, Tuple

import torch
from vllm import LLM

# Llama 3.1 advertises a 131072 context, which makes vLLM reserve ~16 GiB of
# KV cache and fail to start on a 32GB V100. Measured worst case across the
# 300 benchmark is ~2200 prompt + 2048 output tokens.
DEFAULT_MAX_MODEL_LEN = 8192

# 0.90 leaves ~0 margin for any other job on a shared GPU. 8B weights only
# need ~16GB and max_model_len caps the KV cache, so 0.6 tolerates real-world
# contention while staying well above what this model actually needs.
# On a 24GB card 0.6 is 14.4GB, below the bf16 weights, so a dedicated pod
# (scripts/runpod/launch.sh) raises it through VLLM_GPU_MEMORY_UTILIZATION.
DEFAULT_GPU_MEMORY_UTILIZATION = float(
    os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.6")
)

_ENGINES: Dict[Tuple[str, str, int, int], LLM] = {}


def engine_dtype() -> str:
    """fp16 on pre-Ampere, bf16 otherwise.

    The checkpoint is bf16, but SM70 (V100) has no bf16 tensor cores, so fp16
    is the only option there that is both fast and fits.
    """
    major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    return "float16" if major < 8 else "bfloat16"


def get_engine(
    model_version: str,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
) -> LLM:
    """Return a vLLM engine for this checkpoint, building it only once.

    Callers that differ only in prompting (llama_basic vs llama_rescriber)
    get the same engine back; anything that changes what the engine *is* --
    a different checkpoint, parallelism or context length -- keys a separate
    one. gpu_memory_utilization is deliberately not part of the key: it sizes
    the memory pool rather than changing the model, so the first caller's
    value applies and later callers share that pool.
    """
    if tensor_parallel_size is None:
        tensor_parallel_size = max(1, torch.cuda.device_count())

    dtype = engine_dtype()
    key = (model_version, dtype, tensor_parallel_size, max_model_len)

    if key not in _ENGINES:
        _ENGINES[key] = LLM(
            model=f"meta-llama/Llama-{model_version}",
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    return _ENGINES[key]
