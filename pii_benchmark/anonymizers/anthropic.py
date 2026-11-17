from typing import List
import anthropic
from pii_benchmark.anonymizers.anonymizer import Anonymizer
from pii_benchmark.anonymizers.gpt_anon import parse_results_rescriber
from pii_benchmark.prompts import get_anonymization_prompt

from pii_benchmark.credentials import anthropic_api_key


class AnthropicAnonymizer(Anonymizer):
    def __init__(
        self,
        model_version: str = "claude-haiku-5-5",
        attributes: List[str]|None = None,
        language: str = "English",
        prompt_type: str = "anthropic"
    ):
        super().__init__()
        self.model_version = model_version
        self.attributes = attributes
        self.language = language
        self.prompt_type = prompt_type
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)

    def anonymize(self, text: str, scenario: str = "medical") -> str:
        base_prompt = get_anonymization_prompt(self.prompt_type, text, self.attributes, scenario=scenario, language=self.language)

        message = self.client.messages.create(
            model=self.model_version,
            max_tokens=3000,
            system=base_prompt,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": text}]}
            ],
        )

        # Thinking is on by default: read text blocks by type, not by position
        out = "".join(b.text for b in message.content if b.type == "text")

        if self.prompt_type=="rescriber":
            redacted_text = text
            entities = parse_results_rescriber(out)
            # print("entities:" + str(entities))
            if len(entities) > 0 and "results" in entities[0]:
                entities = entities[0]["results"]
            for e in entities:
                entity_text = e["text"]
                redacted_text = redacted_text.replace(entity_text, "*" * len(entity_text))
            anon_text = redacted_text
        else:
            anon_text = out

        return anon_text