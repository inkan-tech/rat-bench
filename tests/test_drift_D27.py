"""D27: LlamaRescriberAnonymizer (method llama_rescriber) uses the medical
Rescriber prompt for every profile, as the paper-era llama_rescriber did.
Scenario prompts stay available behind scenario_prompts=True.
Offline: fake engine, no model."""
from pathlib import Path
from types import SimpleNamespace

import pii_benchmark
from pii_benchmark.anonymizers.llamarescriber import LlamaRescriberAnonymizer

PROMPTS = Path(pii_benchmark.__file__).parent / "anonymizers" / "anon_prompts"


class FakeEngine:
    def __init__(self):
        self.chats = []

    def chat(self, conversations, sampling_params):
        self.chats.extend(conversations)
        return [SimpleNamespace(outputs=[SimpleNamespace(text="[]")]) for _ in conversations]


def _anonymizer(**attrs):
    a = object.__new__(LlamaRescriberAnonymizer)
    a.prompt_type = "rescriber"
    a.attributes = None
    a.scenario = "medical"
    a.model = FakeEngine()
    a.sampling_params = None
    for k, v in attrs.items():
        setattr(a, k, v)
    return a


def test_batch_uses_medical_prompt_for_every_scenario():
    medical = (PROMPTS / "rescriber_medical.txt").read_text()
    a = _anonymizer(scenario_prompts=False)
    scenarios = ["Chatbot conversation", "Meeting transcript", "Medical consultation"]
    out = a.anonymize_batch(["a", "b", "c"], scenarios=scenarios)
    assert out == ["a", "b", "c"]
    assert len(a.model.chats) == 3
    assert all(chat[0]["content"] == medical for chat in a.model.chats)


def test_single_uses_medical_prompt_with_chatbot_scenario():
    medical = (PROMPTS / "rescriber_medical.txt").read_text()
    a = _anonymizer(scenario="chatbot", scenario_prompts=False)
    a.anonymize("Hello, I am John.", "Chatbot conversation")
    assert a.model.chats[0][0]["content"] == medical


def test_scenario_prompts_option_keeps_routing():
    meeting = (PROMPTS / "rescriber_meeting.txt").read_text()
    a = _anonymizer(scenario_prompts=True)
    a.anonymize_batch(["a"], scenarios=["Meeting transcript"])
    assert a.model.chats[0][0]["content"] == meeting
