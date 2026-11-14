import json
from pathlib import Path

import pytest

from jev_eval.clients.base import ClassifierClient
from jev_eval.conditions import get_condition
from jev_eval.config import EvalConfig
from jev_eval.data.banking77 import Banking77Adapter
from jev_eval.data.datasets import DatasetSpec, TaskType
from jev_eval.data.io import read_jsonl, to_jsonable
from jev_eval.pipeline import run_inference as ri
from jev_eval.pipeline.analyze import analyze
from jev_eval.types import ClassificationResult


class SdkLike:
    """Stand-in for a non-serializable SDK response object."""

    def __init__(self):
        self.choice = "a"
        self.confidence = 0.8


class RecordingClient(ClassifierClient):
    model_id = "rec"

    def __init__(self, fail_ids=()):
        self.calls, self.fail_ids = [], set(fail_ids)

    def classify(self, message, criteria, prompt):
        self.calls.append((message, dict(criteria), prompt.condition))
        if message in self.fail_ids:
            raise ValueError("boom")
        return ClassificationResult("a", 0.8, SdkLike(), self.model_id, "test")


def make_config(tmp_path: Path, pt_desc=None, conditions=("C1", "C3")) -> EvalConfig:
    intents = [{"label_id": 0, "label_name": "a", "description_en": "option A", "description_pt_br": "opção A" if pt_desc is None else pt_desc},
               {"label_id": 1, "label_name": "b", "description_en": "option B", "description_pt_br": "opção B" if pt_desc is None else pt_desc}]
    (tmp_path / "intents.json").write_text(json.dumps(intents), encoding="utf-8")
    rows = [{"example_id": f"e{i}", "split": "test", "label_id": 0, "label_name": "a",
             "text_en": f"hello {i}", "text_pt_br": f"olá {i}"} for i in range(2)]
    (tmp_path / "sample.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    spec = DatasetSpec("banking77", TaskType.CHOICE,
                       {"intents": tmp_path / "intents.json", "sample": tmp_path / "sample.jsonl"})
    return EvalConfig(42, 10, 2, list(conditions), ["jev", "llm"], [spec],
                      {"raw_predictions": tmp_path / "out" / "raw.jsonl"}, tmp_path / "c.yaml")


def patch_clients(monkeypatch, clients):
    monkeypatch.setattr(ri, "build_clients", lambda mock, config=None: clients)
    monkeypatch.setattr(ri.time, "sleep", lambda s: None)


def criteria_for(config, conditions):
    adapter = Banking77Adapter(config.datasets[0])
    return ri.criteria_by_language(adapter, [get_condition(c) for c in conditions])


def test_options_follow_instruction_language(tmp_path, monkeypatch):
    jev, llm = RecordingClient(), RecordingClient()
    patch_clients(monkeypatch, {"jev": jev, "llm": llm})
    ri.run_inference(make_config(tmp_path, conditions=("C1", "C2", "C3")))
    expected = {"C1": "option A", "C2": "option A", "C3": "opção A"}
    for message, criteria, condition in jev.calls:
        assert criteria["a"] == expected[condition]


def test_content_follows_content_language(tmp_path, monkeypatch):
    jev, llm = RecordingClient(), RecordingClient()
    patch_clients(monkeypatch, {"jev": jev, "llm": llm})
    ri.run_inference(make_config(tmp_path, conditions=("C1", "C2")))
    assert {(m.split()[0], c) for m, _, c in jev.calls} == {("hello", "C1"), ("olá", "C2")}


def test_c1_c2_do_not_need_pt_descriptions(tmp_path, monkeypatch):
    patch_clients(monkeypatch, {"jev": RecordingClient(), "llm": RecordingClient()})
    ri.run_inference(make_config(tmp_path, pt_desc="", conditions=("C1", "C2")))


def test_missing_pt_br_descriptions_fail_fast(tmp_path):
    with pytest.raises(ValueError, match="description_pt_br"):
        criteria_for(make_config(tmp_path, pt_desc=""), ["C3"])


def test_untranslated_text_fails_fast(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    row = {"example_id": "e0", "split": "test", "label_id": 0, "label_name": "a",
           "text_en": "same", "text_pt_br": "same"}
    config.datasets[0].paths["sample"].write_text(json.dumps(row), encoding="utf-8")
    patch_clients(monkeypatch, {"jev": RecordingClient(), "llm": RecordingClient()})
    with pytest.raises(ValueError, match="untranslated"):
        ri.run_inference(config)


def test_non_choice_tasks_are_rejected(tmp_path):
    config = make_config(tmp_path)
    config.datasets = [DatasetSpec("boolq", TaskType.BINARY)]
    with pytest.raises(NotImplementedError, match="binary"):
        ri.run_inference(config)


def test_records_are_serialized_and_run_resumes(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    jev, llm = RecordingClient(), RecordingClient(fail_ids={"olá 1"})
    patch_clients(monkeypatch, {"jev": jev, "llm": llm})
    ri.run_inference(config)
    rows = read_jsonl(config.paths["raw_predictions"])
    assert len(rows) == 8
    assert rows[0]["raw_response"] == {"choice": "a", "confidence": 0.8}
    assert rows[0]["dataset"] == "banking77" and rows[0]["condition"] == "C1"
    assert sum(r["error"] is not None for r in rows) == 1

    jev2, llm2 = RecordingClient(), RecordingClient()
    patch_clients(monkeypatch, {"jev": jev2, "llm": llm2})
    ri.run_inference(config)
    assert jev2.calls == [] and llm2.calls == [("olá 1", llm2.calls[0][1], "C3")]
    rows = read_jsonl(config.paths["raw_predictions"])
    assert len(rows) == 8 and all(r["error"] is None for r in rows)


def test_resume_refuses_records_without_condition(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config.paths["raw_predictions"].parent.mkdir(parents=True)
    config.paths["raw_predictions"].write_text(json.dumps({"example_id": "e0", "language": "en"}) + "\n")
    patch_clients(monkeypatch, {"jev": RecordingClient(), "llm": RecordingClient()})
    with pytest.raises(ValueError, match="--fresh"):
        ri.run_inference(config)


def test_analyze_reports_errors_and_uses_paired_subset(tmp_path):
    def rec(eid, system, conf, correct, error=None):
        return {"example_id": eid, "condition": "C2", "system": system, "confidence": conf,
                "correct": correct, "error": error}
    records = [rec("e0", "jev", 0.9, True), rec("e0", "llm", 0.9, True),
               rec("e1", "jev", 0.9, False), rec("e1", "llm", 0.0, False, error="parse")]
    metrics = analyze(records, 1, tmp_path / "m.json")["conditions"]["C2"]
    assert metrics["errors"]["llm"]["n"] == 1 and metrics["errors"]["jev"]["n"] == 0
    assert metrics["n_paired"] == 1
    assert metrics["paired"]["jev"]["n"] == 1 and metrics["all_valid"]["jev"]["n"] == 2


def test_to_jsonable_handles_pydantic_like():
    class Model:
        def model_dump(self):
            return {"x": 1}
    assert to_jsonable(Model()) == {"x": 1}


def test_duplicate_descriptions_fail_fast(tmp_path):
    with pytest.raises(ValueError, match="share the same"):
        criteria_for(make_config(tmp_path, pt_desc="mesma descrição"), ["C3"])
