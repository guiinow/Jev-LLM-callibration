import json
import os
import re
from typing import Any

from jev_eval.clients.base import ClassifierClient
from jev_eval.prompts.templates import TaskPrompt
from jev_eval.types import ClassificationResult


def parse_llm_json(text: str, labels: set[str]) -> tuple[str, float]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("LLM response does not contain a JSON object")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise ValueError("LLM response contains invalid JSON") from error
    intent = payload.get("intent")
    confidence = payload.get("confidence")
    if intent not in labels:
        raise ValueError(f"Invalid intent returned by LLM: {intent!r}")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        raise ValueError("LLM confidence must be a number between 0 and 1")
    return intent, float(confidence)


class LlmClient(ClassifierClient):
    """OpenAI-compatible chat client (OpenAI API or a local server such as Ollama's /v1)."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", api_key: str | None = None) -> None:
        from openai import OpenAI
        api_key = api_key or os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is required for LlmClient; use --mock for local smoke runs.")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4.1-mini")
        self.model_id = self.model
        self.client = OpenAI(api_key=api_key, base_url=base_url or os.getenv("OPENAI_BASE_URL"))

    def classify(self, message: str, criteria: dict[str, str], prompt: TaskPrompt) -> ClassificationResult:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt.render_llm(message, criteria)}],
        )
        text = response.choices[0].message.content or ""
        intent, confidence = parse_llm_json(text, set(criteria))
        return ClassificationResult(intent, confidence, text, self.model, prompt.version)


class MockLlmClient(ClassifierClient):
    model_id = "mock-llm"

    def classify(self, message: str, criteria: dict[str, str], prompt: TaskPrompt) -> ClassificationResult:
        from jev_eval.clients.jev_client import _mock_choice
        labels = list(criteria)
        label, base, raw = _mock_choice(message, labels)
        confidence = min(0.99, base + 0.25)
        raw["confidence"] = confidence
        return ClassificationResult(label, confidence, raw, self.model_id, prompt.version)

