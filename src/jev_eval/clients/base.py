from abc import ABC, abstractmethod
from typing import Any

from jev_eval.types import ClassificationResult


class InvalidResponse(ValueError):
    """A deterministic, unusable answer (bad JSON, unknown intent, truncated context).

    Subclasses ValueError so the runner does not retry it, and carries what the server
    returned so the raw output is still saved for auditing.
    """

    def __init__(self, message: str, raw_response: Any = None, served_model: str | None = None,
                 usage: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.served_model = served_model
        self.usage = usage


class ClassifierClient(ABC):
    model_id: str = ""
    prompt_version: str = ""

    @abstractmethod
    def classify(self, message: str, criteria: dict[str, str], language: str) -> ClassificationResult:
        raise NotImplementedError
