from dataclasses import dataclass
from typing import Any


@dataclass
class PredictionRecord:
    example_id: str
    dataset: str
    condition: str
    system: str
    true_label_name: str
    predicted_label_name: str
    confidence: float
    correct: bool
    raw_response: Any
    model_id: str
    prompt_version: str
    error: str | None


@dataclass
class ClassificationResult:
    predicted_label: str
    confidence: float
    raw_response: Any
    model_id: str
    prompt_version: str = ""
