"""D34: Scrubadub keeps the paper-era DateOfBirthDetector for English locales
(fix in b0d4550); other locales keep it off. Offline: the module-level
nltk.download is stubbed, and the test only inspects the detector set."""
import importlib

import nltk


def _module(monkeypatch):
    monkeypatch.setattr(nltk, "download", lambda *a, **k: True)
    return importlib.import_module("pii_benchmark.anonymizers.scrubadub")


def test_english_has_date_of_birth_detector(monkeypatch):
    mod = _module(monkeypatch)
    a = mod.ScrubadubAnonymizer(locale="en_US")
    assert "date_of_birth" in a.scrubber._detectors
    assert "text_blob_name" in a.scrubber._detectors


def test_spanish_keeps_date_of_birth_detector_off(monkeypatch):
    mod = _module(monkeypatch)
    a = mod.ScrubadubAnonymizer(locale="es_MX")
    assert "date_of_birth" not in a.scrubber._detectors
