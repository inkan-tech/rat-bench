from typing import List

import scrubadub

from pii_benchmark.anonymizers.anonymizer import Anonymizer
import nltk

nltk.download("punkt_tab")


class ScrubadubAnonymizer(Anonymizer):
    def __init__(self, locale: str = "en_US", attributes: List[str] = None): # es_MX
        super().__init__()
        self.locale = locale
        self.scrubber = scrubadub.Scrubber(locale=self.locale)

        # English only: the paper-era code (b8cface) used it, and without it
        # level-1 re-identification rose from ~0.44 to 0.66 with the same
        # attacker. It was disabled for all locales when es_MX was added.
        if self.locale.startswith("en"):
            self.scrubber.add_detector(scrubadub.detectors.DateOfBirthDetector)
        self.scrubber.add_detector(scrubadub.detectors.TextBlobNameDetector)

    def anonymize(self, text: str, scenario: str = "") -> str:
        anonymized_text = self.scrubber.clean(text)
        return anonymized_text
