import json
import os

from jev_eval.clients.base import ClassifierClient, InvalidResponse
from jev_eval.prompts.templates import TOPK, TaskPrompt
from jev_eval.types import ClassificationResult

# Ollama silently truncates prompts longer than num_ctx; treat a request that used >= this
# share of the window as truncated instead of trusting its answer.
CONTEXT_GUARD = 0.9


def first_json_object(text: str) -> dict:
    """The first complete JSON object in the answer; later blocks and notes are ignored.

    (The model sometimes appends a note and a second "corrected" JSON block; joining the
    first `{` to the last `}` would make the whole answer invalid.)"""
    start = text.find("{")
    if start < 0:
        raise ValueError("LLM response does not contain a JSON object")
    try:
        payload, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as error:
        raise ValueError("LLM response contains invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON is not an object")
    return payload


def parse_llm_json(text: str, labels: set[str]) -> tuple[str, float]:
    payload = first_json_object(text)
    intent = payload.get("intent")
    confidence = payload.get("confidence")
    if intent not in labels:
        raise ValueError(f"Invalid intent returned by LLM: {intent!r}")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        raise ValueError("LLM confidence must be a number between 0 and 1")
    return intent, float(confidence)


def parse_topk_json(text: str, labels: set[str]) -> tuple[str, float]:
    """Top-k answer -> (guess with the highest probability, its probability); first wins ties.

    As in Tian et al., only the top guess is used, so only it must be a valid intent; lower
    guesses with invented intents are ignored (ablation tables count them). Every guess still
    needs an intent string and a probability in [0, 1]; the list may be shorter than k.
    Probabilities are not required to sum to 1 (Tian et al. do not either).
    """
    guesses = first_json_object(text).get("guesses")
    if not isinstance(guesses, list) or not 1 <= len(guesses) <= TOPK:
        raise ValueError(f"LLM 'guesses' must be a list of 1 to {TOPK} objects")
    parsed = []
    for guess in guesses:
        intent = guess.get("intent") if isinstance(guess, dict) else None
        probability = guess.get("probability") if isinstance(guess, dict) else None
        if not isinstance(intent, str):
            raise ValueError("LLM guess without an intent string")
        if not isinstance(probability, (int, float)) or isinstance(probability, bool) or not 0 <= probability <= 1:
            raise ValueError("LLM probability must be a number between 0 and 1")
        parsed.append((intent, float(probability)))
    intent, probability = max(parsed, key=lambda guess: guess[1])
    if intent not in labels:
        raise ValueError(f"Invalid intent returned by LLM: {intent!r}")
    return intent, probability


def invalid_lower_guesses(text: str, labels: set[str]) -> int:
    """How many guesses of an accepted top-k answer name an intent outside the options."""
    try:
        guesses = first_json_object(text).get("guesses") or []
    except ValueError:
        return 0
    return sum(1 for g in guesses if isinstance(g, dict) and g.get("intent") not in labels)


PARSERS = {"single": parse_llm_json, "topk": parse_topk_json}


class LlmClient(ClassifierClient):
    """OpenAI-compatible chat client (OpenAI API or a local server such as Ollama's /v1)."""

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

    def classify(self, message: str, criteria: dict[str, str], prompt: TaskPrompt) -> ClassificationResult:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt.render_llm(message, criteria)}],
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
            intent, confidence = PARSERS[prompt.answer](text, set(criteria))
        except ValueError as error:
            raise InvalidResponse(str(error), text, served, usage) from error
        return ClassificationResult(intent, confidence, text, self.model, prompt.version,
                                    served_model=served, usage=usage)


class MockLlmClient(ClassifierClient):
    model_id = "mock-llm"

    def classify(self, message: str, criteria: dict[str, str], prompt: TaskPrompt) -> ClassificationResult:
        from jev_eval.clients.jev_client import _mock_choice
        labels = list(criteria)
        label, base, raw = _mock_choice(message, labels)
        confidence = min(0.99, base + 0.25)
        raw["confidence"] = confidence
        return ClassificationResult(label, confidence, raw, self.model_id, prompt.version)
