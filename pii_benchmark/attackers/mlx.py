from typing import List

from pii_benchmark.mlx_engine import SamplingParams, get_engine
from pii_benchmark.prompts import get_staab_prompt_llama
from pii_benchmark.utils import parse_output


class MlxAttacker:
    def __init__(self, model_version: str = "3.1-8B-Instruct") -> None:
        self.model_version = model_version
        self.model = get_engine(model_version)

    def infer(
        self, text: str, attributes: List[str] = None, scenario: str = "reddit"
    ):
        system_prompt, prompt = get_staab_prompt_llama(
            attributes=attributes, text=text, scenario=scenario
        )

        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        outputs = self.model.chat(
            [chat], SamplingParams(temperature=0.0, max_tokens=2048)
        )
        return parse_output(outputs[0].outputs[0].text)
