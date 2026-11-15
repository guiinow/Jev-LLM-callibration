import math

import pytest

from jev_eval.clients.base import InvalidResponse
from jev_eval.clients.open_models import HfVerbalClient, MockOptionProbClient, OptionProbClient
from jev_eval.conditions import CONDITIONS
from jev_eval.prompts.templates import build_task_prompt


class FakeBackend:
    """Stands in for a loaded model: fixed token ids for codes, canned logits and answers."""
    model_id = "fake/model@abc123"

    def __init__(self, logprob_by_code=None, answer="", multi_token=()):
        self.logprob_by_code = logprob_by_code or {}
        self.answer, self.multi_token = answer, set(multi_token)
        self.seen_prompts = []

    def single_token_id(self, text):
        return None if text in self.multi_token else 1000 + ord(text[0]) * 40 + (ord(text[1]) if len(text) > 1 else 0)

    def next_token_logprobs(self, user_text, token_ids):
        self.seen_prompts.append(user_text)
        by_id = {self.single_token_id(code): value for code, value in self.logprob_by_code.items()}
        return [by_id.get(token, -30.0) for token in token_ids], 321

    def generate(self, user_text, max_new_tokens):
        self.seen_prompts.append(user_text)
        return self.answer, {"prompt_tokens": 50, "completion_tokens": 12}


CRITERIA = {"card_arrival": "card arrival", "card_linking": "card linking", "age_limit": "age limit"}
PROMPT = build_task_prompt("banking77", CONDITIONS["C1"], "codes")


def test_probabilities_are_renormalized_over_the_option_codes():
    backend = FakeBackend({"A": math.log(0.6), "B": math.log(0.2), "C": math.log(0.1)})  # 0.1 mass elsewhere
    result = OptionProbClient(backend).classify("my card", CRITERIA, PROMPT)
    probabilities = result.raw_response["probabilities"]
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["card_arrival"] == pytest.approx(0.6 / 0.9)
    assert result.predicted_label == "card_arrival" and result.confidence == pytest.approx(0.6 / 0.9)
    assert result.raw_response["valid_mass"] == pytest.approx(0.9)
    assert result.usage == {"prompt_tokens": 321, "completion_tokens": 0}
    assert result.model_id == result.served_model == "fake/model@abc123"


def test_prompt_lists_codes_not_intent_names():
    backend = FakeBackend({"B": math.log(0.9)})
    result = OptionProbClient(backend).classify("my card", CRITERIA, PROMPT)
    assert "- A: card arrival\n- B: card linking\n- C: age limit" in backend.seen_prompts[0]
    assert "card_arrival" not in backend.seen_prompts[0]
    assert result.predicted_label == "card_linking" and result.raw_response["codes"]["card_linking"] == "B"


def test_codes_skip_multi_token_codes_and_stay_stable_across_calls():
    backend = FakeBackend({"C": math.log(0.9)}, multi_token={"B"})
    client = OptionProbClient(backend)
    first = client.classify("one", CRITERIA, PROMPT)
    second = client.classify("two", CRITERIA, PROMPT)
    assert first.raw_response["codes"] == {"card_arrival": "A", "card_linking": "C", "age_limit": "D"}
    assert second.raw_response["codes"] == first.raw_response["codes"]
    assert first.predicted_label == "card_linking"  # "C" is the second option here


def test_ties_choose_the_first_option():
    result = OptionProbClient(FakeBackend({"A": -1.0, "B": -1.0, "C": -1.0})).classify("x", CRITERIA, PROMPT)
    assert result.predicted_label == "card_arrival" and result.confidence == pytest.approx(1 / 3)


def test_non_finite_logprobs_are_an_invalid_response():
    with pytest.raises(InvalidResponse, match="Non-finite"):
        OptionProbClient(FakeBackend({"A": float("nan")})).classify("x", CRITERIA, PROMPT)


MAIN = build_task_prompt("banking77", CONDITIONS["C1"])


def test_verbalized_client_parses_the_json_answer():
    backend = FakeBackend(answer='Sure: {"intent": "card_linking", "confidence": 0.85}')
    result = HfVerbalClient(backend).classify("link my card", CRITERIA, MAIN)
    assert (result.predicted_label, result.confidence) == ("card_linking", 0.85)
    assert result.usage == {"prompt_tokens": 50, "completion_tokens": 12}
    assert "- card_linking: card linking" in backend.seen_prompts[0]  # main prompt keeps intent names


def test_verbalized_client_keeps_the_raw_text_of_invalid_answers():
    backend = FakeBackend(answer="I think it is card arrival.")
    with pytest.raises(InvalidResponse) as caught:
        HfVerbalClient(backend).classify("x", CRITERIA, MAIN)
    assert caught.value.raw_response == "I think it is card arrival." and caught.value.usage["prompt_tokens"] == 50


def test_mock_clients_return_valid_results():
    result = MockOptionProbClient("gemma").classify("card arrival", CRITERIA, PROMPT)
    assert result.model_id == "mock-gemma-prob" and result.raw_response["valid_mass"] == 1.0
    assert result.predicted_label in CRITERIA


def test_hf_backend_does_not_load_weights_until_used():
    import sys

    from jev_eval.clients.hf_backend import HfBackend
    backend = HfBackend("google/gemma-4-E4B-it", "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c")
    assert backend.model_id == "google/gemma-4-E4B-it@4ecc2ba63bca"
    assert backend._model is None and backend._tokenizer is None
    assert "jev_eval.clients.hf_backend" in sys.modules
