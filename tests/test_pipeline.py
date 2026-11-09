import json
from pathlib import Path

import pytest

from jev_eval.clients.base import ClassifierClient
from jev_eval.config import EvalConfig
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

    def classify(self, message, criteria, language):
        self.calls.append((message, dict(criteria), language))
        if message in self.fail_ids:
            raise ValueError("boom")
        return ClassificationResult("a", 0.8, SdkLike(), self.model_id, "test")


def make_config(tmp_path: Path, pt_desc=None) -> EvalConfig:
    intents = [{"label_id": 0, "label_name": "a", "description_en": "option A", "description_pt_br": "opção A" if pt_desc is None else pt_desc},
               {"label_id": 1, "label_name": "b", "description_en": "option B", "description_pt_br": "opção B" if pt_desc is None else pt_desc}]
    (tmp_path / "intents.json").write_text(json.dumps(intents), encoding="utf-8")
    rows = [{"example_id": f"e{i}", "split": "test", "label_id": 0, "label_name": "a",
             "text_en": f"hello {i}", "text_pt_br": f"olá {i}"} for i in range(2)]
    (tmp_path / "sample.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return EvalConfig(42, 10, 2, ["en", "pt_br"], ["jev", "llm"], {
        "intents": tmp_path / "intents.json", "sample": tmp_path / "sample.jsonl",
        "raw_predictions": tmp_path / "out" / "raw.jsonl"}, tmp_path / "c.yaml")


def patch_clients(monkeypatch, clients):
    monkeypatch.setattr(ri, "build_clients", lambda mock: clients)
    monkeypatch.setattr(ri.time, "sleep", lambda s: None)


def test_pt_br_uses_pt_br_descriptions(tmp_path, monkeypatch):
    jev, llm = RecordingClient(), RecordingClient()
    patch_clients(monkeypatch, {"jev": jev, "llm": llm})
    ri.run_inference(make_config(tmp_path))
    for message, criteria, language in jev.calls:
        expected = "opção A" if language == "pt_br" else "option A"
        assert criteria["a"] == expected


def test_missing_pt_br_descriptions_fail_fast(tmp_path):
    with pytest.raises(ValueError, match="description_pt_br"):
        ri.criteria_by_language(make_config(tmp_path, pt_desc=""))


def test_untranslated_text_fails_fast(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    row = {"example_id": "e0", "split": "test", "label_id": 0, "label_name": "a",
           "text_en": "same", "text_pt_br": "same"}
    config.paths["sample"].write_text(json.dumps(row), encoding="utf-8")
    patch_clients(monkeypatch, {"jev": RecordingClient(), "llm": RecordingClient()})
    with pytest.raises(ValueError, match="untranslated"):
        ri.run_inference(config)


def test_records_are_serialized_and_run_resumes(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    jev, llm = RecordingClient(), RecordingClient(fail_ids={"olá 1"})
    patch_clients(monkeypatch, {"jev": jev, "llm": llm})
    ri.run_inference(config)
    rows = read_jsonl(config.paths["raw_predictions"])
    assert len(rows) == 8
    assert rows[0]["raw_response"] == {"choice": "a", "confidence": 0.8}
    assert sum(r["error"] is not None for r in rows) == 1

    jev2, llm2 = RecordingClient(), RecordingClient()
    patch_clients(monkeypatch, {"jev": jev2, "llm": llm2})
    ri.run_inference(config)
    assert jev2.calls == [] and llm2.calls == [("olá 1", llm2.calls[0][1], "pt_br")]
    rows = read_jsonl(config.paths["raw_predictions"])
    assert len(rows) == 8 and all(r["error"] is None for r in rows)


def test_analyze_reports_errors_and_uses_paired_subset(tmp_path):
    def rec(eid, system, conf, correct, error=None):
        return {"example_id": eid, "language": "pt_br", "system": system, "confidence": conf,
                "correct": correct, "error": error}
    records = [rec("e0", "jev", 0.9, True), rec("e0", "llm", 0.9, True),
               rec("e1", "jev", 0.9, False), rec("e1", "llm", 0.0, False, error="parse")]
    metrics = analyze(records, 1, tmp_path / "m.json")["languages"]["pt_br"]
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
        ri.criteria_by_language(make_config(tmp_path, pt_desc="mesma descrição"))
