import pytest

from jev_eval.clients.llm_client import parse_llm_json


def test_parse_llm_json_with_extra_text():
    assert parse_llm_json('Here is the result: {"intent":"foo","confidence":0.7}', {"foo"}) == ("foo", 0.7)


def test_parse_llm_json_rejects_invalid_intent():
    with pytest.raises(ValueError, match="Invalid intent"):
        parse_llm_json('{"intent":"bar","confidence":0.7}', {"foo"})
