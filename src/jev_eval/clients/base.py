from abc import ABC, abstractmethod

from jev_eval.prompts.templates import TaskPrompt
from jev_eval.types import ClassificationResult


class ClassifierClient(ABC):
    @abstractmethod
    def classify(self, message: str, criteria: dict[str, str], prompt: TaskPrompt) -> ClassificationResult:
        raise NotImplementedError
