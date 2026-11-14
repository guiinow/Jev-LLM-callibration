from abc import ABC, abstractmethod
from typing import Any

from jev_eval.prompts.templates import TaskPrompt
from jev_eval.types import ClassificationResult


class InvalidResponse(ValueError):
    """The server answered, but the answer is unusable. Keeps the raw output for auditing.

    A ValueError, so it is not retried (deterministic at temperature 0)."""

    def __init__(self, message: str, raw_response: Any = None, served_model: str | None = None,
                 usage: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.served_model = served_model
        self.usage = usage


class ClassifierClient(ABC):
    model_id = ""

    @abstractmethod
    def classify(self, message: str, criteria: dict[str, str], prompt: TaskPrompt) -> ClassificationResult:
        raise NotImplementedError
