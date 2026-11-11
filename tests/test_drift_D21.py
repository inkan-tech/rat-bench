"""D21: LlamaAnonymizer.anonymize_rescriber uses the medical Rescriber prompt
for every profile, as the paper-era code (b8cface) did. Scenario prompts stay
available behind scenario_prompts=True. Offline: fake engine, no model."""
from pathlib import Path
from types import SimpleNamespace

import pii_benchmark
from pii_benchmark.anonymizers.llama import LlamaAnonymizer

PROMPTS = Path(pii_benchmark.__file__).parent / "anonymizers" / "anon_prompts"


class FakeEngine:
    def __init__(self):
        self.chats = []

    def chat(self, conversations, sampling_params):
        self.chats.extend(conversations)
        return [SimpleNamespace(outputs=[SimpleNamespace(text="[]")]) for _ in conversations]


def _anonymizer(**attrs):
    a = object.__new__(LlamaAnonymizer)
    a.prompt_type = "rescriber"
    a.attributes = None
    a.scenario = "medical"
    a.model = FakeEngine()
    for k, v in attrs.items():
        setattr(a, k, v)
    return a


def test_rescriber_uses_medical_prompt_for_every_scenario():
    medical = (PROMPTS / "rescriber_medical.txt").read_text()
    a = _anonymizer(scenario_prompts=False)
    for scenario in ("Chatbot conversation", "Meeting transcript", "Medical consultation", ""):
        a.anonymize_rescriber("Hello, I am John.", scenario)
    assert len(a.model.chats) == 4
    assert all(chat[0]["content"] == medical for chat in a.model.chats)


def test_scenario_prompts_option_keeps_routing():
    chatbot = (PROMPTS / "rescriber_chatbot.txt").read_text()
    a = _anonymizer(scenario_prompts=True)
    a.anonymize_rescriber("Hello, I am John.", "Chatbot conversation")
    assert a.model.chats[0][0]["content"] == chatbot
