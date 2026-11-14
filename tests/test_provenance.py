"""Jev confidence = probabilities[choice] (proposal 4.2), versions and context guard, run
metadata, input-hash resume guard, raw output kept on invalid answers, legacy re-analysis."""
import json
from types import SimpleNamespace

import pytest

from jev_eval.clients.base import ClassifierClient, InvalidResponse
from jev_eval.clients.jev_client import MockJevClient, chosen_probability
from jev_eval.clients.llm_client import LlmClient
from jev_eval.conditions import CONDITIONS
from jev_eval.data.io import read_jsonl
from jev_eval.pipeline import metadata as md
from jev_eval.pipeline import run_inference as ri
from jev_eval.pipeline.analyze import analyze, with_chosen_probability, with_conditions
from jev_eval.prompts.templates import build_task_prompt
from test_pipeline import RecordingClient, make_config, patch_clients

C1 = build_task_prompt("banking77", CONDITIONS["C1"])


# --- Jev confidence -----------------------------------------------------------------------

def test_chosen_probability_reads_the_chosen_option():
    assert chosen_probability({"a": 0.7, "b": 0.3}, "a") == 0.7
    with pytest.raises(ValueError):
        chosen_probability({"a": 0.7}, "b")
    with pytest.raises(ValueError):
        chosen_probability([0.7, 0.3], "a")


def test_mock_jev_confidence_is_probability_of_choice():
    result = MockJevClient().classify("activate my card", {"activate_my_card": "x", "age_limit": "y"}, C1)
    assert result.confidence == result.raw_response["probabilities"][result.predicted_label]


def legacy_jev(eid, conf, probs, pred="a", correct=True):
    # Shape of records written before the fix: no confidence_sdk key, SDK value in `confidence`.
    return {"example_id": eid, "dataset": "banking77", "condition": "C3", "system": "jev",
            "predicted_label_name": pred, "confidence": conf, "correct": correct, "error": None,
            "raw_response": {"choice": pred, "confidence": conf, "probabilities": probs}}


def test_legacy_records_get_probability_of_choice_and_keep_sdk_value():
    rows = [legacy_jev("e0", 0.5, {"a": 0.9, "b": 0.1}), legacy_jev("e1", 0.4, None)]
    out = with_chosen_probability(rows)
    assert out[0]["confidence"] == 0.9 and out[0]["confidence_sdk"] == 0.5
    assert out[1]["confidence"] == 0.4 and out[1]["confidence_source"].startswith("sdk_confidence")
    assert "confidence_sdk" not in rows[0]  # input untouched
    assert with_chosen_probability(out) == out  # idempotent


def test_analyze_reports_sdk_confidence_check(tmp_path):
    llm = {"example_id": "e0", "condition": "C3", "system": "llm", "confidence": 1.0, "correct": True,
           "error": None}
    metrics = analyze([legacy_jev("e0", 0.5, {"a": 0.9, "b": 0.1}), llm], 1, tmp_path / "m.json")
    c3 = metrics["conditions"]["C3"]
    assert c3["paired"]["jev"]["ece"] == pytest.approx(0.1)  # |1 - 0.9|, from probabilities[choice]
    assert c3["jev_confidence_sdk_check"]["ece"] == pytest.approx(0.5)  # |1 - 0.5|, SDK field
    assert metrics["jev_confidence_source"] == {"probabilities[choice] (derived from raw)": 1}


def test_preliminary_language_records_map_to_conditions():
    rows = with_conditions([{"example_id": "e0", "language": "en"}, {"example_id": "e0", "language": "pt_br"},
                            {"example_id": "e0", "dataset": "banking77", "condition": "C2"}])
    assert [(r["dataset"], r["condition"]) for r in rows] == [
        ("banking77", "C1"), ("banking77", "PT-preliminar"), ("banking77", "C2")]


# --- resume only over identical inputs ------------------------------------------------------

def test_resume_refuses_records_made_from_other_inputs(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    patch_clients(monkeypatch, {"jev": RecordingClient(), "llm": RecordingClient()})
    ri.run_inference(config)
    sample = config.datasets[0].paths["sample"]
    rows = [json.loads(line) for line in sample.read_text().splitlines()]
    rows[1]["text_pt_br"] = "olá revisado"
    sample.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="different inputs"):
        ri.run_inference(config)
    assert len(read_jsonl(config.paths["raw_predictions"])) == 8  # nothing overwritten


def test_resume_refuses_records_without_hash(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    old = {"example_id": "e0", "dataset": "banking77", "condition": "C1", "system": "jev", "error": None,
           "confidence": 0.5}
    config.paths["raw_predictions"].parent.mkdir(parents=True)
    config.paths["raw_predictions"].write_text(json.dumps(old) + "\n", encoding="utf-8")
    patch_clients(monkeypatch, {"jev": RecordingClient(), "llm": RecordingClient()})
    with pytest.raises(ValueError, match="input_hash"):
        ri.run_inference(config)


def test_input_hash_depends_on_message_options_model_and_prompt():
    base = ri.input_hash("m", {"a": "x"}, "rec", C1)
    assert base == ri.input_hash("m", {"a": "x"}, "rec", C1)
    assert base != ri.input_hash("m2", {"a": "x"}, "rec", C1)
    assert base != ri.input_hash("m", {"a": "y"}, "rec", C1)
    assert base != ri.input_hash("m", {"a": "x"}, "other", C1)
    assert base != ri.input_hash("m", {"a": "x"}, "rec", build_task_prompt("banking77", CONDITIONS["C3"]))
    assert base != ri.input_hash("m", {"a": "x"}, "rec", build_task_prompt("banking77", CONDITIONS["C1"], "topk"))


# --- raw output kept on invalid answers -----------------------------------------------------

class InvalidClient(ClassifierClient):
    model_id = "bad"

    def classify(self, message, criteria, prompt):
        raise InvalidResponse("Invalid intent returned by LLM: 'zzz'", '{"intent": "zzz"}', "served-x",
                              {"prompt_tokens": 10, "completion_tokens": 3})


def test_invalid_answer_keeps_raw_text(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    patch_clients(monkeypatch, {"jev": RecordingClient(), "llm": InvalidClient()})
    ri.run_inference(config)
    bad = [r for r in read_jsonl(config.paths["raw_predictions"]) if r["system"] == "llm"]
    assert bad and all(r["raw_response"] == '{"intent": "zzz"}' and r["served_model"] == "served-x"
                       and r["error"].startswith("InvalidResponse") and r["latency_s"] is not None
                       for r in bad)


# --- LLM client: served model, usage, context guard ---------------------------------------

def fake_llm(text, prompt_tokens, completion_tokens=5, num_ctx=8192):
    client = object.__new__(LlmClient)
    client.model = client.model_id = "m"
    client.num_ctx = num_ctx
    response = SimpleNamespace(
        model="m-served", choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    return client


def test_llm_client_records_served_model_and_usage():
    result = fake_llm('{"intent": "a", "confidence": 0.8}', 1500).classify("msg", {"a": "x"}, C1)
    assert (result.predicted_label, result.confidence, result.served_model) == ("a", 0.8, "m-served")
    assert result.usage == {"prompt_tokens": 1500, "completion_tokens": 5} and result.prompt_version == C1.version


def test_llm_client_flags_context_overflow():
    with pytest.raises(InvalidResponse, match="ContextOverflow") as caught:
        fake_llm('{"intent": "a", "confidence": 0.8}', 8000).classify("msg", {"a": "x"}, C1)
    assert caught.value.raw_response == '{"intent": "a", "confidence": 0.8}'


def test_llm_client_invalid_json_keeps_text():
    with pytest.raises(InvalidResponse) as caught:
        fake_llm("not json", 100).classify("msg", {"a": "x"}, C1)
    assert caught.value.raw_response == "not json" and caught.value.usage["prompt_tokens"] == 100


# --- clients per config, run metadata -------------------------------------------------------

def test_build_clients_only_for_configured_systems(tmp_path):
    config = make_config(tmp_path)
    config.systems = ["llm"]
    assert set(ri.build_clients(True, config)) == {"llm"}


def test_run_writes_metadata_history(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    patch_clients(monkeypatch, {"jev": RecordingClient(), "llm": RecordingClient()})
    ri.run_inference(config, mock=True)
    ri.run_inference(config, mock=True)
    history = json.loads((config.paths["raw_predictions"].parent / "run_metadata.json").read_text())
    assert len(history) == 2 and history[0]["mock"] is True and history[0]["ollama"] is None
    assert "python" in history[0]["machine"] and "typesafe-sdk" in history[0]["packages"]
    assert history[0]["config"]["conditions"] == ["C1", "C3"]
    assert history[0]["config"]["datasets"][0]["name"] == "banking77"


def test_ollama_helpers(monkeypatch):
    assert md.ollama_root("http://localhost:11434/v1") == "http://localhost:11434"
    assert md.ollama_root("https://api.openai.com/v1") is None
    import httpx

    def boom(*args, **kwargs):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx.Client, "get", boom)
    info = md.ollama_info("http://localhost:11434/v1", "m")
    assert info["error"].startswith("ConnectError")


def test_metadata_redacts_keys(tmp_path):
    config = make_config(tmp_path)
    config.llm = {"model": "m", "api_key": "secret"}
    entry = md.collect_run_metadata(config, mock=True)
    assert entry["config"]["llm"]["api_key"] == "<redacted>"
