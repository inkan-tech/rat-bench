"""Apple Silicon stand-in for ``vllm_engine``, built on mlx-lm.

It mirrors the slice of the vLLM API this repo uses (``get_engine``,
``SamplingParams`` and ``LLM.chat``), so the Llama anonymizers run unchanged.
Several conversations passed to ``chat`` are generated as one batch.

Only one process may hold the model at a time: it takes ~9GB of unified
memory, and the machine must stay usable while a run is going.
"""

import fcntl
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Dict, List

from mlx_lm import batch_generate, generate, load
from mlx_lm.sample_utils import make_sampler

# Kept equal to vllm_engine so callers can import either module.
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_GPU_MEMORY_UTILIZATION = 0.6

# Each sequence in a batch adds its own KV cache (~1GB at 8192 tokens for an
# 8B model), so the batch stays small enough to keep the machine usable.
BATCH_SIZE = int(os.environ.get("MLX_BATCH_SIZE", "4"))

MLX_REPOS = {
    "3.1-8B-Instruct": "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
}

LOCK_PATH = os.path.join(tempfile.gettempdir(), "rat-bench-mlx.lock")

_ENGINES: Dict[str, "MlxEngine"] = {}
_lock_file = None


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


def _acquire_process_lock() -> None:
    """Refuse to load a model while another process holds one."""
    global _lock_file
    if _lock_file is not None:
        return
    f = open(LOCK_PATH, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise RuntimeError(
            f"Another process already holds an MLX model ({LOCK_PATH}). "
            "Only one model at a time: wait for it to finish."
        )
    # Held until the process exits.
    _lock_file = f


class MlxEngine:
    def __init__(self, repo: str):
        self.model, self.tokenizer = load(repo)
        # attack.py calls infer from a thread pool; MLX generation is not
        # thread-safe, so requests are serialized here.
        self._lock = threading.Lock()
        # Single requests waiting for submit() to batch them.
        self._queue: List[list] = []
        self._queue_lock = threading.Lock()

    def submit(
        self, chat: List[Dict[str, str]], sampling_params: SamplingParams
    ) -> _RequestOutput:
        """Generate one conversation, batched with the others waiting.

        Callers that each send one conversation from their own thread (the
        attack thread pool) are grouped into one chat() call: whoever gets the
        engine runs every queued request with the same sampling parameters.
        """
        item = [chat, sampling_params, threading.Event(), None, None]
        with self._queue_lock:
            self._queue.append(item)
        while not item[2].is_set():
            if not self._lock.acquire(blocking=False):
                item[2].wait(0.05)
                continue
            try:
                with self._queue_lock:
                    params = self._queue[0][1] if self._queue else None
                    batch = [it for it in self._queue if it[1] == params]
                    self._queue = [it for it in self._queue if it[1] != params]
                if batch:
                    try:
                        outputs = self._chat([it[0] for it in batch], params)
                        for it, output in zip(batch, outputs):
                            it[3] = output
                    except Exception as e:
                        for it in batch:
                            it[4] = e
                    for it in batch:
                        it[2].set()
            finally:
                self._lock.release()
        if item[4] is not None:
            raise item[4]
        return item[3]

    def chat(
        self, conversations: List[List[Dict[str, str]]], sampling_params: SamplingParams
    ) -> List[_RequestOutput]:
        with self._lock:
            return self._chat(conversations, sampling_params)

    def _chat(
        self, conversations: List[List[Dict[str, str]]], sampling_params: SamplingParams
    ) -> List[_RequestOutput]:
        sampler = make_sampler(temp=sampling_params.temperature)
        if len(conversations) == 1:
            prompt = self.tokenizer.apply_chat_template(
                conversations[0], add_generation_prompt=True, tokenize=False
            )
            texts = [
                generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=sampling_params.max_tokens,
                    sampler=sampler,
                )
            ]
        else:
            prompts = [
                self.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
                for chat in conversations
            ]
            texts = batch_generate(
                self.model,
                self.tokenizer,
                prompts,
                max_tokens=sampling_params.max_tokens,
                sampler=sampler,
                completion_batch_size=BATCH_SIZE,
                prefill_batch_size=BATCH_SIZE,
            ).texts
        return [_RequestOutput([_Completion(t)]) for t in texts]


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
        _acquire_process_lock()
        _ENGINES[model_version] = MlxEngine(MLX_REPOS[model_version])
    return _ENGINES[model_version]
