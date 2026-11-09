import json
from typing import Dict, List

from pii_benchmark.anonymizers.anonymizer import Anonymizer
from pii_benchmark.anonymizers.vllm_engine import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_MODEL_LEN,
    SamplingParams,
    get_engine,
)
from pii_benchmark.prompts import get_anonymization_prompt

# Rescriber emits only a JSON entity list; worst case measured on the 300
# benchmark is ~1600 tokens.
MAX_OUTPUT_TOKENS = 2048

MAX_MODEL_LEN = DEFAULT_MAX_MODEL_LEN


class LlamaRescriberAnonymizer(Anonymizer):
    """Rescriber anonymizer backed by vLLM so profiles can be batched.

    vLLM's continuous batching only pays off when many prompts are submitted
    together, so prefer anonymize_batch over calling anonymize in a loop.
    """

    def __init__(
        self,
        prompt_type: str,
        attributes: List[str],
        model_version: str = "3.1-8B-Instruct",
        scenario: str = "medical",
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        max_model_len: int = MAX_MODEL_LEN,
    ):
        super().__init__()
        self.prompt_type = prompt_type
        self.attributes = attributes
        self.model_version = model_version
        self.scenario = scenario

        # Shared with LlamaAnonymizer when both want this checkpoint;
        # see vllm_engine.get_engine.
        self.model = get_engine(
            model_version=model_version,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

    def _build_chat(self, text: str, scenario: str) -> List[Dict[str, str]]:
        prompt = get_anonymization_prompt(
            method="rescriber",
            text=text,
            instruct_template=True,
            scenario=scenario or self.scenario,
        )
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]

    def _redact(self, text: str, model_output: str) -> str:
        redacted_text = text
        for e in self.parse_results(model_output):
            entity_text = e.get("text")
            if entity_text:
                redacted_text = redacted_text.replace(
                    entity_text, "*" * len(entity_text)
                )
        return redacted_text

    def anonymize(self, text: str, scenario: str = "") -> str:
        return self.anonymize_batch([text], scenario)[0]

    def anonymize_batch(
        self, texts: List[str], scenario: str = "", scenarios: List[str] | None = None
    ) -> List[str]:
        """Anonymize many texts in a single batched vLLM call.

        scenarios, when given, supplies a per-text scenario and must align
        with texts; otherwise scenario applies to all of them.
        """
        if not texts:
            return []
        if scenarios is not None and len(scenarios) != len(texts):
            raise ValueError(
                f"scenarios has {len(scenarios)} items but texts has {len(texts)}"
            )

        conversations = [
            self._build_chat(text, scenarios[i] if scenarios is not None else scenario)
            for i, text in enumerate(texts)
        ]

        outputs = self.model.chat(conversations, self.sampling_params)

        # vLLM returns results in submission order.
        return [
            self._redact(text, output.outputs[0].text)
            for text, output in zip(texts, outputs)
        ]
    def parse_results(self, output) -> str:
        lines = output.splitlines()

        entities = []

        for l in lines:
            line = l.strip().strip("]").strip("[").strip(",")
            if len(line)==0 or line[0] != "{":
                continue
            if "entity_type" in line and "text" in line:
                try:
                    entity = json.loads(line)
                    entities.append(entity)
                except json.JSONDecodeError:
                    pass
        if entities==[]:
            i = 0
            while i<len(lines)-1:
                line = lines[i]
                if line=="{":
                    i += 1
                elif "entity_type" in line:
                    if "text" in lines[i+1]:
                        entities.append({
                            "entity_type": line.split(":")[-1].strip(",").strip("\"").strip(),
                            "text": lines[i+1].split(":")[-1].strip().strip("\"")
                        })
                    i += 2
                elif line=="}":
                    i += 1
                else:
                    i += 1


        return entities
