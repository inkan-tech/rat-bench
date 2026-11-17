from typing import List
import anthropic
from pii_benchmark.anonymizers.anonymizer import Anonymizer
from pii_benchmark.prompts import get_system_prompt

from pii_benchmark.credentials import anthropic_api_key


class ClioAnonymizer(Anonymizer):
    def __init__(
        self,
        model_version: str = "claude-opus-5-5",
        attributes: List[str] = None,
    ):
        super().__init__()
        self.model_version = model_version
        self.attributes = attributes

    def anonymize(self, text: str, scenario: str = "") -> str:
        base_prompt = get_system_prompt(self.base_prompt)

        client = anthropic.Anthropic(api_key=anthropic_api_key)
        message = client.messages.create(
            model=self.model_version,
            max_tokens=1000,
            system=base_prompt,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": text}]}
            ],
        )

        return "".join(b.text for b in message.content if b.type == "text")
