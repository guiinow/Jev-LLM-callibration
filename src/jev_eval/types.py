from dataclasses import dataclass
from typing import Any


@dataclass
class SampleExample:
    example_id: str
    split: str
    label_id: int
    label_name: str
    text_en: str
    text_pt_br: str
    translation_status: str | None = None


@dataclass
class IntentDefinition:
    label_id: int
    label_name: str
    description_en: str
    description_pt_br: str


@dataclass
class PredictionRecord:
    example_id: str
    language: str
    system: str
    true_label_name: str
    predicted_label_name: str
    confidence: float
    correct: bool
    raw_response: Any
    model_id: str
    prompt_version: str
    error: str | None
    confidence_sdk: float | None = None  # Jev's own `confidence` statistic, kept as a check (not used for ECE)
    served_model: str | None = None  # model/version the API reports it actually used
    usage: dict[str, Any] | None = None  # token counts reported by the LLM server
    input_hash: str | None = None  # fingerprint of (message, options, model, prompt); resume key
    latency_s: float | None = None  # wall time of the call, retries included


@dataclass
class ClassificationResult:
    predicted_label: str
    confidence: float
    raw_response: Any
    model_id: str
    prompt_version: str = ""
    confidence_sdk: float | None = None
    served_model: str | None = None
    usage: dict[str, Any] | None = None
