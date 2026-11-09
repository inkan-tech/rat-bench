from typing import List
from pii_benchmark.anonymizers.anonymizer import Anonymizer
from pii_benchmark.anonymizers.gpt_anon import parse_results_rescriber
from pii_benchmark.claude_code_cli import run_claude_code
from pii_benchmark.prompts import get_anonymization_prompt
from pii_benchmark.utils import retry_with_backoff


class ClaudeCodeAnonymizer(Anonymizer):
    """Same prompts as AnthropicAnonymizer, but calls the Claude Code CLI (`claude -p`) instead of the API."""

    def __init__(
        self,
        model_version: str = "haiku",
        attributes: List[str]|None = None,
        language: str = "English",
        prompt_type: str = "anthropic"
    ):
        super().__init__()
        self.model_version = model_version
        self.attributes = attributes
        self.language = language
        self.prompt_type = prompt_type

    def anonymize(self, text: str, scenario: str = "medical") -> str:
        base_prompt = get_anonymization_prompt(self.prompt_type, text, self.attributes, scenario=scenario, language=self.language)

        out = retry_with_backoff(run_claude_code, prompt=text, system_prompt=base_prompt, model=self.model_version)

        if self.prompt_type=="rescriber":
            redacted_text = text
            entities = parse_results_rescriber(out)
            if len(entities) > 0 and "results" in entities[0]:
                entities = entities[0]["results"]
            for e in entities:
                entity_text = e["text"]
                redacted_text = redacted_text.replace(entity_text, "*" * len(entity_text))
            anon_text = redacted_text
        else:
            anon_text = out

        return anon_text
