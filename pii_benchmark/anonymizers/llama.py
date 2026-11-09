import json
from typing import Dict, List

import torch
from tqdm import tqdm

from pii_benchmark.anonymizers.anonymizer import Anonymizer
from pii_benchmark.anonymizers.vllm_engine import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_MODEL_LEN,
    SamplingParams,
    get_engine,
)
from pii_benchmark.prompts import get_anonymization_prompt

# Rescriber emits only a JSON entity list; the other prompt types return a
# full rewritten text, so they get more headroom.
MAX_OUTPUT_TOKENS_RESCRIBER = 2048
MAX_OUTPUT_TOKENS = 4096

MAX_MODEL_LEN = DEFAULT_MAX_MODEL_LEN


def _pipeline_dtype():
    """fp16 on pre-Ampere (bf16 is emulated there), bf16 otherwise."""
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.float16 if major < 8 else torch.bfloat16


class LlamaAnonymizer(Anonymizer):
    """vLLM-backed anonymizer; prefer anonymize_batch over calling anonymize
    in a loop, since vLLM's continuous batching only pays off when many
    prompts are submitted together (see LlamaRescriberAnonymizer).
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

        # Shared with LlamaRescriberAnonymizer when both want this
        # checkpoint; see vllm_engine.get_engine.
        self.model = get_engine(
            model_version=model_version,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

    def anonymize(
        self, text: str, scenario: str = "", attributes: List[str] | None = None, prompt_type: str | None = None,
    ) -> str:
        if prompt_type == None:
            pt = self.prompt_type
        elif prompt_type == "rescriber":
            return self.anonymize_rescriber(text, scenario)
        elif prompt_type=="clio":
            return self.anonymize_clio(text)
        else:
            pt = prompt_type
        if attributes == None:
            atts = self.attributes
        else:
            atts = attributes
        chat = self._build_chat(pt, text, atts)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)
        outputs = self.model.chat([chat], sampling_params)
        return outputs[0].outputs[0].text

    def _build_chat(self, prompt_type: str, text: str, attributes: List[str] | None) -> List[Dict[str, str]]:
        prompt = get_anonymization_prompt(
            prompt_type, text, attributes, instruct_template=True
        )
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]

    def anonymize_batch(
        self,
        texts: List[str],
        scenario: str = "",
        scenarios: List[str] | None = None,
        attributes: List[str] | None = None,
        prompt_type: str | None = None,
    ) -> List[str]:
        """Anonymize many texts in a single batched vLLM call.

        scenarios, when given, supplies a per-text scenario and must align
        with texts; scenario is unused here (the underlying prompt types
        this batches don't vary by scenario) but kept for interface
        parity with LlamaRescriberAnonymizer.anonymize_batch.
        """
        if not texts:
            return []
        if scenarios is not None and len(scenarios) != len(texts):
            raise ValueError(
                f"scenarios has {len(scenarios)} items but texts has {len(texts)}"
            )

        pt = self.prompt_type if prompt_type is None else prompt_type
        atts = self.attributes if attributes is None else attributes

        conversations = [self._build_chat(pt, text, atts) for text in texts]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)
        outputs = self.model.chat(conversations, sampling_params)

        # vLLM returns results in submission order.
        return [output.outputs[0].text for output in outputs]

    def anonymize_rescriber(self, text: str, scenario: str = "") -> str:
        redacted_text = text
        entities = []

        prompt = get_anonymization_prompt(
            method="rescriber",
            text=text,
            instruct_template=True,
            scenario=scenario or self.scenario,
        )

        chat = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]

        # Rescriber emits only a JSON entity list; observed worst case is
        # ~1600 tokens, so more would just let runaway generations burn time.
        sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS_RESCRIBER)
        outputs = self.model.chat([chat], sampling_params)
        resp = outputs[0].outputs[0].text
        entities = self.parse_results(resp)

        for e in entities:
            entity_text = e["text"]
            redacted_text = redacted_text.replace(entity_text, "*" * len(entity_text))
            # while entity_text in redacted_text:
            #     start = redacted_text.find(entity_text)
            #     end = start + len(entity_text)
            #     redacted_text = (
            #         redacted_text[:start] + ("*" * len(entity_text)) + redacted_text[end:]
            #     )

        return redacted_text
    
    def parse_results(self, output) -> List[Dict[str, str]]:
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
            while i<len(lines):
                line = lines[i]
                if line=="{":
                    i += 1
                elif "entity_type" in line:
                    if len(lines)> i+1 and "text" in lines[i+1]:
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
    
    def anonymize_clio(
            self, text:str
    ):
        prompt1 = get_anonymization_prompt(method="clio", text=text, scenario=self.scenario)
        prompt1 = prompt1 + "\n{text}</conversation>"

        chat = [
            {"role": "system", "content": prompt1},
            {"role": "user", "content": text},
        ]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)
        outputs = self.model.chat([chat], sampling_params)
        anon_text = outputs[0].outputs[0].text

        # chat.append({
        #             "role": "assistant", "content": response1
        #             }
        # )
        # chat.append({
        #     "role": "user", "content": prompt2
        # })

        # response = self.model(chat, max_new_tokens=4096)
        # anon_text = ""
        # for r in response[0]["generated_text"]:
        #     if r["role"] == "assistant":
        #         anon_text = r["content"]
    
        return anon_text
