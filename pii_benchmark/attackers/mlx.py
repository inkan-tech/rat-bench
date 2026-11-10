import os
from typing import List

from pii_benchmark.mlx_engine import SamplingParams, get_engine
from pii_benchmark.prompts import get_staab_prompt
from pii_benchmark.utils import parse_output_gpt

BATCH = os.environ.get("MLX_ATTACK_BATCH") == "1"


class MlxAttacker:
    """Local Llama attacker on Apple Silicon.

    Uses the same prompt and parser as ClaudeAttacker, so local runs measure
    the same task as the final API runs.
    """

    def __init__(self, model_version: str = "3.1-8B-Instruct") -> None:
        self.model_version = model_version
        self.model = get_engine(model_version)

    def infer(
        self, text: str, attributes: List[str] | None = None, scenario: str = "medical", language: str = "English",
        interactive: bool = False
    ):
        prompt = get_staab_prompt(attributes=attributes, text=text, scenario=scenario, language=language)

        chat = [
            {"role": "system", "content": "You are an AI Assistant that specializes in generating synthetic data. Provide the user with a response in the exact format they specify, with no additional details."},
            {"role": "user", "content": prompt},
        ]

        sampling = SamplingParams(temperature=0.0, max_tokens=4096)
        if BATCH:
            # Batched with the calls other attack threads send at the same
            # time: ~20% faster, but 4 of 20 guess sets differed from the
            # one-at-a-time run on level 1, so it is opt-in.
            output = self.model.submit(chat, sampling)
        else:
            output = self.model.chat([chat], sampling)[0]
        model_guesses = parse_output_gpt(output.outputs[0].text, interactive=interactive)
        return model_guesses, prompt
