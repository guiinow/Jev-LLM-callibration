import math
import os
import re
from typing import Any

from jev_eval.clients.base import ClassifierClient
from jev_eval.prompts.templates import TaskPrompt
from jev_eval.types import ClassificationResult


def _mock_choice(message: str, labels: list[str]) -> tuple[str, float, dict[str, Any]]:
    tokens = set(re.findall(r"[a-z0-9_]+", message.lower()))
    scores = [sum(token in tokens for token in label.split("_")) for label in labels]
    best = max(range(len(labels)), key=lambda index: (scores[index], -index))
    logits = [float(score) for score in scores]
    exps = [math.exp(value) for value in logits]
    probabilities = [value / sum(exps) for value in exps]
    return labels[best], probabilities[best], {"choice": labels[best], "probabilities": probabilities}


class JevClient(ClassifierClient):
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("TYPESAFE_DEFAULT_MODEL", "jev-latest")
        self.model_id = self.model
        api_key = os.getenv("TYPESAFE_API_KEY")
        if not api_key:
            raise RuntimeError("TYPESAFE_API_KEY is required for JevClient; use --mock for local smoke runs.")
        from typesafe_sdk import TypeSafeClient
        self.client = TypeSafeClient(model=self.model)

    def classify(self, message: str, criteria: dict[str, str], prompt: TaskPrompt) -> ClassificationResult:
        from typesafe_sdk import Choice
        question_id = "intent"
        response = self.client.system_one(
            state=message,
            questions={question_id: Choice(instructions=prompt.instructions, criteria=criteria)},
        )
        answer = response.answers[question_id]
        return ClassificationResult(answer.choice, float(answer.confidence), answer, self.model, prompt.version)


class MockJevClient(ClassifierClient):
    model_id = "mock-jev"

    def classify(self, message: str, criteria: dict[str, str], prompt: TaskPrompt) -> ClassificationResult:
        labels = list(criteria)
        label, confidence, raw = _mock_choice(message, labels)
        return ClassificationResult(label, confidence, raw, self.model_id, prompt.version)
