"""Pick the LLM inference backend from the RAT_BACKEND environment variable.

mlx    Apple Silicon, mlx-lm (pii_benchmark.mlx_engine).
vllm   Linux with CUDA, vLLM (pii_benchmark.anonymizers.vllm_engine).
runpod Code runs on a RunPod GPU pod, which is Linux with CUDA, so it uses the
       vLLM engine. scripts/runpod/launch.sh sets it on the pod.

Default: mlx on macOS, vllm elsewhere.
"""

import os
import sys

BACKENDS = ("mlx", "vllm", "runpod")


def get_backend() -> str:
    backend = os.environ.get("RAT_BACKEND") or (
        "mlx" if sys.platform == "darwin" else "vllm"
    )
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown RAT_BACKEND {backend!r}. Valid options: {', '.join(BACKENDS)}"
        )
    return backend


def load_engine():
    """Return (SamplingParams, DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_MODEL_LEN, get_engine) for the selected backend."""
    if get_backend() == "mlx":
        from pii_benchmark.mlx_engine import (
            DEFAULT_GPU_MEMORY_UTILIZATION,
            DEFAULT_MAX_MODEL_LEN,
            SamplingParams,
            get_engine,
        )
    else:
        from vllm import SamplingParams

        from pii_benchmark.anonymizers.vllm_engine import (
            DEFAULT_GPU_MEMORY_UTILIZATION,
            DEFAULT_MAX_MODEL_LEN,
            get_engine,
        )
    return (
        SamplingParams,
        DEFAULT_GPU_MEMORY_UTILIZATION,
        DEFAULT_MAX_MODEL_LEN,
        get_engine,
    )
