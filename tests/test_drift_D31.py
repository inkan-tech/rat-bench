"""D31: the English Presidio analyzer is built with the default NLP engine
(AnalyzerEngine(), i.e. en_core_web_lg), as the paper-era code effectively
did, and not with an en_core_web_sm engine. Offline: fake engines, no spaCy
model is loaded."""
from pii_benchmark.anonymizers import presidio


class FakeRegistry:
    def __init__(self):
        self.added = []

    def add_recognizer(self, r):
        self.added.append(r)


class FakeAnalyzer:
    built = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.registry = FakeRegistry()
        FakeAnalyzer.built.append(self)


def test_english_analyzer_uses_default_engine(monkeypatch):
    loaded = []
    FakeAnalyzer.built = []
    monkeypatch.setattr(presidio, "AnalyzerEngine", FakeAnalyzer)
    monkeypatch.setattr(presidio, "AnonymizerEngine", lambda *a, **k: object())
    monkeypatch.setattr(presidio, "LoadedSpacyNlpEngine", lambda *a, **k: ("sm-engine", a, k))
    monkeypatch.setattr(presidio.spacy, "load", lambda name, *a, **k: loaded.append(name) or name)

    anon = presidio.PresidioAnonymizer(language="en")

    assert anon.analyzer is FakeAnalyzer.built[-1]
    assert anon.analyzer.args == ()
    assert anon.analyzer.kwargs.get("nlp_engine") is None
    assert anon.language == "en"
    assert presidio.titles_recognizer in anon.analyzer.registry.added
    assert presidio.sex_recognizer in anon.analyzer.registry.added


def test_presidio_default_nlp_model_is_large():
    # AnalyzerEngine() without nlp_engine builds NlpEngineProvider's default
    # configuration; the paper Presidio numbers came from en_core_web_lg.
    import inspect
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    assert "en_core_web_lg" in inspect.getsource(NlpEngineProvider)
