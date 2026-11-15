"""Open-weight LLMs (Gemma, Qwen) read in two ways from the SAME loaded model:

- `OptionProbClient` ("<model>_prob"): the answer is the code of an option; the model's
  next-token distribution restricted to the option codes and renormalized is one probability
  per option, and the confidence is the probability of the chosen option (as in the Jev
  benchmark, Deußer et al. 2026, 3.6). No text is generated.
- `HfVerbalClient` ("<model>_verb"): the model writes the intent and a 0-1 confidence as JSON,
  with the same main prompt the other LLM gets.

Both sit on a backend with the three methods below, so tests use a fake and the real one
(`hf_backend.HfBackend`) loads weights only when first called.
"""
import math
from typing import Any, Protocol

from jev_eval.clients.base import ClassifierClient, InvalidResponse
from jev_eval.clients.jev_client import _mock_choice
from jev_eval.clients.llm_client import PARSERS
from jev_eval.clients.option_codes import assign_codes
from jev_eval.prompts.templates import TaskPrompt
from jev_eval.types import ClassificationResult

MAX_NEW_TOKENS = 96


class LanguageBackend(Protocol):
    model_id: str

    def single_token_id(self, text: str) -> int | None:
        """The token id if `text` is exactly one token of the vocabulary, else None."""

    def next_token_logprobs(self, user_text: str, token_ids: list[int]) -> tuple[list[float], int]:
        """Log-probabilities (over the FULL vocabulary) of `token_ids` as the first answer token,
        and the prompt length in tokens."""

    def generate(self, user_text: str, max_new_tokens: int) -> tuple[str, dict[str, Any]]:
        """Greedy answer text and token usage."""


class OptionProbClient(ClassifierClient):
    def __init__(self, backend: LanguageBackend) -> None:
        self.backend = backend
        self.model_id = backend.model_id
        self._codes: dict[tuple[str, ...], dict[str, str]] = {}

    def _codes_for(self, labels: list[str]) -> dict[str, str]:
        key = tuple(labels)
        if key not in self._codes:
            self._codes[key] = assign_codes(labels, lambda code: self.backend.single_token_id(code) is not None)
        return self._codes[key]

    def classify(self, message: str, criteria: dict[str, str], prompt: TaskPrompt) -> ClassificationResult:
        labels = list(criteria)
        codes = self._codes_for(labels)
        coded_options = {codes[label]: criteria[label] for label in labels}
        token_ids = [self.backend.single_token_id(codes[label]) for label in labels]
        logprobs, prompt_tokens = self.backend.next_token_logprobs(
            prompt.render_llm(message, coded_options), token_ids)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 0}
        if not all(math.isfinite(value) for value in logprobs):
            raise InvalidResponse("Non-finite option log-probabilities", None, self.model_id, usage)
        shift = max(logprobs)
        weights = [math.exp(value - shift) for value in logprobs]
        total = sum(weights)
        probabilities = {label: weight / total for label, weight in zip(labels, weights)}
        chosen = max(labels, key=lambda label: probabilities[label])  # first label wins ties
        raw = {"probabilities": probabilities, "valid_mass": math.exp(shift) * total,
               "codes": {label: codes[label] for label in labels}}
        return ClassificationResult(chosen, probabilities[chosen], raw, self.model_id, prompt.version,
                                    served_model=self.model_id, usage=usage)


class HfVerbalClient(ClassifierClient):
    def __init__(self, backend: LanguageBackend, max_new_tokens: int = MAX_NEW_TOKENS) -> None:
        self.backend = backend
        self.model_id = backend.model_id
        self.max_new_tokens = max_new_tokens

    def classify(self, message: str, criteria: dict[str, str], prompt: TaskPrompt) -> ClassificationResult:
        text, usage = self.backend.generate(prompt.render_llm(message, criteria), self.max_new_tokens)
        try:
            intent, confidence = PARSERS[prompt.answer](text, set(criteria))
        except ValueError as error:
            raise InvalidResponse(str(error), text, self.model_id, usage) from error
        return ClassificationResult(intent, confidence, text, self.model_id, prompt.version,
                                    served_model=self.model_id, usage=usage)


class MockOptionProbClient(ClassifierClient):
    def __init__(self, name: str) -> None:
        self.model_id = f"mock-{name}-prob"

    def classify(self, message: str, criteria: dict[str, str], prompt: TaskPrompt) -> ClassificationResult:
        label, confidence, raw = _mock_choice(message, list(criteria))
        raw["valid_mass"] = 1.0
        return ClassificationResult(label, confidence, raw, self.model_id, prompt.version)


class MockVerbalClient(ClassifierClient):
    def __init__(self, name: str) -> None:
        self.model_id = f"mock-{name}-verb"

    def classify(self, message: str, criteria: dict[str, str], prompt: TaskPrompt) -> ClassificationResult:
        label, base, raw = _mock_choice(message, list(criteria))
        confidence = min(0.99, base + 0.3)
        raw["confidence"] = confidence
        return ClassificationResult(label, confidence, raw, self.model_id, prompt.version)
