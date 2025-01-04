import langdetect
from langchain.schema import runnable
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer

from backend.src.common import BaseObject, Config
from backend.src.common.constants import NLP_CONFIG


class BotAnonymizer(BaseObject):
    def __init__(self, config: Config = None):
        super(BotAnonymizer, self).__init__()
        self.config = config if config is not None else Config()
        self._anonymizer = PresidioReversibleAnonymizer(languages_config=NLP_CONFIG)

    @property
    def anonymizer(self):
        return self._anonymizer

    @property
    def supported_lang(self):
        return ['vi', 'en']

    def _detect_lang(self, input_dict: dict) -> dict:
        language = langdetect.detect(input_dict["input"])
        if language not in self.supported_lang:
            self.logger.warning(
                "Language %s is not supported by the anonymizer. Supported languages are %s",
            )
            language = None
        return {"language": language, **input_dict}

    def anonymize_func(self, input_dict: dict) -> dict:
        if input_dict["language"] is None:
            return {
                "input": input_dict["input"],
                "history": input_dict["history"],
            }
        return {
            "input": self.anonymizer.anonymize(input_dict["input"], input_dict["language"]),
            "history": self.anonymizer.anonymize(input_dict["history"], input_dict["language"]),
        }

    def get_runnable_anonymizer(self):
        return runnable.RunnableLambda(self._detect_lang) | runnable.RunnableLambda(self.anonymize_func)
