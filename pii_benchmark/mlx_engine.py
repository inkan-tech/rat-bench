"""Apple Silicon stand-in for ``vllm_engine``, built on mlx-lm.

It mirrors the slice of the vLLM API this repo uses (``get_engine``,
``SamplingParams`` and ``LLM.chat``), so the Llama anonymizers run unchanged.
Prompts are generated one after the other: mlx-lm has no continuous batching
here, so ``chat`` only keeps vLLM's call shape.
"""

from dataclasses import dataclass
from typing import Dict, List

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

# Kept equal to vllm_engine so callers can import either module.
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_GPU_MEMORY_UTILIZATION = 0.6

MLX_REPOS = {
    "3.1-8B-Instruct": "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
}

_ENGINES: Dict[str, "MlxEngine"] = {}


@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 2048


@dataclass
class _Completion:
    text: str


@dataclass
class _RequestOutput:
    outputs: List[_Completion]


class MlxEngine:
    def __init__(self, repo: str):
        self.model, self.tokenizer = load(repo)

    def chat(
        self, conversations: List[List[Dict[str, str]]], sampling_params: SamplingParams
    ) -> List[_RequestOutput]:
        sampler = make_sampler(temp=sampling_params.temperature)
        results = []
        for chat in conversations:
            prompt = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )
            text = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=sampling_params.max_tokens,
                sampler=sampler,
            )
            results.append(_RequestOutput([_Completion(text)]))
        return results


def get_engine(
    model_version: str,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
) -> MlxEngine:
    """Return an MLX engine for this checkpoint, loading it only once.

    tensor_parallel_size, gpu_memory_utilization and max_model_len exist only
    for signature parity with vllm_engine.get_engine and are ignored.
    """
    if model_version not in MLX_REPOS:
        raise ValueError(
            f"No MLX checkpoint for {model_version!r}. Known: {sorted(MLX_REPOS)}"
        )
    if model_version not in _ENGINES:
        _ENGINES[model_version] = MlxEngine(MLX_REPOS[model_version])
    return _ENGINES[model_version]
