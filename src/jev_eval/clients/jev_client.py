import math
import os
import re
from typing import Any

from jev_eval.clients.base import ClassifierClient, InvalidResponse
from jev_eval.types import ClassificationResult

JEV_PROMPT_VERSION = "jev-native"


def chosen_probability(probabilities: Any, choice: str) -> float:
    """Proposal 4.2: Jev's confidence is its probability for the chosen option."""
    if hasattr(probabilities, "items"):
        probabilities = dict(probabilities.items())
    if not isinstance(probabilities, dict) or choice not in probabilities:
        raise ValueError(f"Jev probabilities have no entry for the chosen option {choice!r}")
    value = float(probabilities[choice])
    if not 0 <= value <= 1:
        raise ValueError(f"Jev probability for {choice!r} is outside [0, 1]: {value}")
    return value


def _mock_choice(message: str, labels: list[str]) -> tuple[str, float, dict[str, Any]]:
    tokens = set(re.findall(r"[a-z0-9_]+", message.lower()))
    scores = [sum(token in tokens for token in label.split("_")) for label in labels]
    best = max(range(len(labels)), key=lambda index: (scores[index], -index))
    logits = [float(score) for score in scores]
    exps = [math.exp(value) for value in logits]
    probabilities = {label: value / sum(exps) for label, value in zip(labels, exps)}
    return labels[best], probabilities[labels[best]], {"choice": labels[best], "probabilities": probabilities}


class JevClient(ClassifierClient):
    prompt_version = JEV_PROMPT_VERSION

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("TYPESAFE_DEFAULT_MODEL", "jev-latest")
        self.model_id = self.model
        api_key = os.getenv("TYPESAFE_API_KEY")
        if not api_key:
            raise RuntimeError("TYPESAFE_API_KEY is required for JevClient; use --mock for local smoke runs.")
        from typesafe_sdk import TypeSafeClient
        self.client = TypeSafeClient(model=self.model)

    def classify(self, message: str, criteria: dict[str, str], language: str) -> ClassificationResult:
        from typesafe_sdk import Choice
        question_id = "intent"
        response = self.client.system_one(
            state=message,
            questions={question_id: Choice(instructions="Choose the banking intent.", criteria=criteria)},
        )
        answer = response.answers[question_id]
        served = getattr(response, "model", None)
        try:
            confidence = chosen_probability(answer.probabilities, answer.choice)
        except ValueError as error:
            raise InvalidResponse(str(error), answer, served) from error
        return ClassificationResult(answer.choice, confidence, answer, self.model, JEV_PROMPT_VERSION,
                                    confidence_sdk=float(answer.confidence), served_model=served)


class MockJevClient(ClassifierClient):
    model_id = "mock-jev"
    prompt_version = JEV_PROMPT_VERSION

    def classify(self, message: str, criteria: dict[str, str], language: str) -> ClassificationResult:
        labels = list(criteria)
        label, confidence, raw = _mock_choice(message, labels)
        return ClassificationResult(label, confidence, raw, self.model_id, JEV_PROMPT_VERSION,
                                    confidence_sdk=confidence, served_model=self.model_id)
