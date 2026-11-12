import json
import os
import re

from jev_eval.clients.base import ClassifierClient, InvalidResponse
from jev_eval.prompts.llm_prompt import PROMPT_VERSION, build_prompt
from jev_eval.types import ClassificationResult

# Ollama silently truncates prompts longer than num_ctx; treat a request that used >= this
# share of the window as truncated instead of trusting its answer.
CONTEXT_GUARD = 0.9


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

    prompt_version = PROMPT_VERSION

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", api_key: str | None = None,
                 num_ctx: int | None = None) -> None:
        from openai import OpenAI
        api_key = api_key or os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is required for LlmClient; use --mock for local smoke runs.")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4.1-mini")
        self.model_id = self.model
        self.num_ctx = num_ctx
        self.client = OpenAI(api_key=api_key, base_url=base_url or os.getenv("OPENAI_BASE_URL"))

    def classify(self, message: str, criteria: dict[str, str], language: str) -> ClassificationResult:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": build_prompt(message, criteria, language)}],
        )
        text = response.choices[0].message.content or ""
        usage = None
        if response.usage is not None:
            usage = {"prompt_tokens": response.usage.prompt_tokens,
                     "completion_tokens": response.usage.completion_tokens}
        served = getattr(response, "model", None)
        if self.num_ctx and usage and (usage["prompt_tokens"] or 0) + (usage["completion_tokens"] or 0) \
                >= CONTEXT_GUARD * self.num_ctx:
            raise InvalidResponse(f"ContextOverflow: used {usage} of num_ctx={self.num_ctx}", text, served, usage)
        try:
            intent, confidence = parse_llm_json(text, set(criteria))
        except ValueError as error:
            raise InvalidResponse(str(error), text, served, usage) from error
        return ClassificationResult(intent, confidence, text, self.model, PROMPT_VERSION,
                                    served_model=served, usage=usage)


class MockLlmClient(ClassifierClient):
    model_id = "mock-llm"
    prompt_version = PROMPT_VERSION

    def classify(self, message: str, criteria: dict[str, str], language: str) -> ClassificationResult:
        from jev_eval.clients.jev_client import _mock_choice
        labels = list(criteria)
        label, base, raw = _mock_choice(message, labels)
        confidence = min(0.99, base + 0.25)
        raw["confidence"] = confidence
        return ClassificationResult(label, confidence, raw, self.model_id, PROMPT_VERSION)
