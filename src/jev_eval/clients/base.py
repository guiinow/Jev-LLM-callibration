from abc import ABC, abstractmethod

from jev_eval.types import ClassificationResult


class ClassifierClient(ABC):
    @abstractmethod
    def classify(self, message: str, criteria: dict[str, str], language: str) -> ClassificationResult:
        raise NotImplementedError

