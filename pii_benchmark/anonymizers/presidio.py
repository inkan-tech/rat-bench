from typing import List
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.predefined_recognizers import EmailRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer.nlp_engine import NlpEngineProvider, SpacyNlpEngine

from pii_benchmark.anonymizers.anonymizer import Anonymizer
import spacy

titles_list = [
    "Sir",
    "Ma'am",
    "Madam",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Miss",
    "Dr.",
    "Professor",
]

sex_indicator_list = ["male", "female", "man", "woman"]
titles_recognizer = PatternRecognizer(
    supported_entity="TITLE", deny_list=titles_list
)
sex_recognizer = PatternRecognizer(
    supported_entity="SEX", deny_list=sex_indicator_list
)

class LoadedSpacyNlpEngine(SpacyNlpEngine):
    def __init__(self, loaded_spacy_model):
        super().__init__()
        self.nlp = {"en": loaded_spacy_model}

class PresidioAnonymizer(Anonymizer):
    def __init__(self, language: str = "en", attributes: List[str]|None=None, scenario: str|None=None) -> None:
        super().__init__()

        if language == "en":
            # Paper-era (b8cface) built an en_core_web_sm engine, then
            # replaced it with AnalyzerEngine(), whose default NLP engine is
            # en_core_web_lg. The paper numbers come from that (drift D31).
            self.language = language
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()

            self.analyzer.registry.add_recognizer(titles_recognizer)
            self.analyzer.registry.add_recognizer(sex_recognizer)

        elif language == "es":
            provider = NlpEngineProvider(conf_file="/data/natasa/rat-bench/pii_benchmark/anonymizers/languages_config.yml")
            nlp_engine_with_spanish = provider.create_engine()

            # email_recognizer_es = EmailRecognizer(supported_language="es", context=["correo", "electrónico"])
            # sex_indicator_list_es = ["hombre", "mujer", "masculino", "femenino"]
            # titles_list_es = [
            #     "Señor",
            #     "Señora",
            #     "Señorita",
            #     "Dr.",
            #     "Profesor",
            # ]
            # titles_recognizer_es = PatternRecognizer(
            #     supported_entity="TITLE", deny_list=titles_list_es, supported_language="es"
            # )
            # sex_recognizer_es = PatternRecognizer(
            #     supported_entity="SEX", deny_list=sex_indicator_list_es, supported_language="es"
                
            #     )
            
            registry = RecognizerRegistry(supported_languages=["es"])

            # Add recognizers to registry
            # registry.add_recognizer(titles_recognizer_es)
            # registry.add_recognizer(sex_recognizer_es)
            # registry.add_recognizer(email_recognizer_es)
                        
            self.language = language
            self.analyzer = AnalyzerEngine(
                registry=registry,
                supported_languages=["es"],
                nlp_engine=nlp_engine_with_spanish)
            self.anonymizer = AnonymizerEngine()
        
        elif language == "sr":
            provider = NlpEngineProvider(conf_file="/data/natasa/rat-bench/pii_benchmark/anonymizers/languages_config.yml")
            nlp_engine_with_croatian = provider.create_engine()

            # email_recognizer_sr = EmailRecognizer(supported_language="cr", context=["email", "adresa", "elektronska pošta"])
            # sex_indicator_list_sr = ["muškarac", "žena", "muški", "ženski"]
            # titles_list_sr = [
            #     "Gospodin",
            #     "Gospođa",
            #     "Gospođica",
            #     "Dr.",
            #     "Profesor",
            # ]
            # titles_recognizer_sr = PatternRecognizer(
            #     supported_entity="TITLE", deny_list=titles_list_sr, supported_language="cr"
            # )
            # sex_recognizer_sr = PatternRecognizer(
            #     supported_entity="SEX", deny_list=sex_indicator_list_sr, supported_language="cr"
                
            #     )
            
            registry = RecognizerRegistry(supported_languages=["cr"])

            # Add recognizers to registry
            # registry.add_recognizer(titles_recognizer_sr)
            # registry.add_recognizer(sex_recognizer_sr)
            # registry.add_recognizer(email_recognizer_sr)
                        
            self.language = "sr"
            self.analyzer = AnalyzerEngine(
                registry=registry,
                supported_languages=["cr"],
                nlp_engine=nlp_engine_with_croatian)
            self.anonymizer = AnonymizerEngine()
        elif language == "nl":
            provider = NlpEngineProvider(conf_file="/data/natasa/rat-bench/pii_benchmark/anonymizers/languages_config.yml")
            nlp_engine_with_dutch = provider.create_engine()

            # email_recognizer_nl = EmailRecognizer(supported_language="nl", context=["email", "adres", "elektronische post"])
            # sex_indicator_list_nl = ["man", "vrouw", "mannelijk", "vrouwelijk"]
            # titles_list_nl = [
            #     "Dhr.",
            #     "Mevr.",
            #     "Ms.",
            #     "Dr.",
            #     "Prof.",
            # ]
            # titles_recognizer_nl = PatternRecognizer(
            #     supported_entity="TITLE", deny_list=titles_list_nl, supported_language="nl"
            # )
            # sex_recognizer_nl = PatternRecognizer(
            #     supported_entity="SEX", deny_list=sex_indicator_list_nl, supported_language="nl"
                
            #     )
            
            registry = RecognizerRegistry(supported_languages=["nl"])

            # Add recognizers to registry
            # registry.add_recognizer(titles_recognizer_nl)
            # registry.add_recognizer(sex_recognizer_nl)
            # registry.add_recognizer(email_recognizer_nl)
                        
            self.language = "nl"
            self.analyzer = AnalyzerEngine(
                registry=registry,
                supported_languages=["nl"],
                nlp_engine=nlp_engine_with_dutch)
            self.anonymizer = AnonymizerEngine()


    def anonymize(self, text: str, scenario: str|None=None) -> str:
        # Analyze
        if self.language=="en":
            analyzer_results = self.analyzer.analyze(
                    text=text, language=self.language, entities=[
                        "CREDIT_CARD", "DATE_TIME", "EMAIL_ADDRESS", "NRP", "LOCATION",
                        "PERSON", "PHONE_NUMBER", "US_SSN"
                    ]
                )
        elif self.language=="es":
            analyzer_results = self.analyzer.analyze(
                    text=text, language=self.language, entities=[
                        "CREDIT_CARD", "DATE_TIME", "EMAIL_ADDRESS", "NRP", "LOCATION",
                        "PERSON", "PHONE_NUMBER"
                    ]
                )   
        elif self.language=="sr":
            analyzer_results = self.analyzer.analyze(
                    text=text, language="cr")
        elif self.language=="nl":
            analyzer_results = self.analyzer.analyze(
                    text=text, language="nl", entities=[
                        "CREDIT_CARD", "DATE_TIME", "EMAIL_ADDRESS", "NRP", "LOCATION",
                        "PERSON", "PHONE_NUMBER"
                    ]
                )
        # Anonymize
        anonymized_results = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators={
                "DEFAULT": OperatorConfig(
                    "mask",
                    {
                        "type": "mask",
                        "masking_char": "*",
                        "chars_to_mask": 50,
                        "from_end": True,
                    },
                )
            },
        )
        return anonymized_results.text
