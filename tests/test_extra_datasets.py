"""CLINC150 and SIB-200 input builders, merging raw predictions from two machines, and resuming
an open-model-only run without losing other systems' failed calls."""
import importlib.util
import json
from pathlib import Path

import pytest

from jev_eval.config import load_config
from jev_eval.data.datasets import build_adapter
from jev_eval.pipeline import run_inference as ri
from jev_eval.prompts.templates import FROZEN_FINGERPRINTS, TEMPLATES, VARIANTS, fingerprint
from test_pipeline import RecordingClient, make_config, patch_clients

ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clinc = load_script("build_clinc150_data")
sib = load_script("build_sib200_data")
merge_script = load_script("merge_raw_predictions")

NAMES = ["a_intent", "oos", "b_intent"]
PT_ROWS = ([{"text": f"a {i}", "text_pt": f"pt a {i}", "intent": 0} for i in range(4)]
           + [{"text": f"b {i}", "text_pt": f"pt b {i}", "intent": 2} for i in range(4)]
           + [{"text": "x", "text_pt": "x", "intent": 1},          # out of scope: ignored
              {"text": "b same", "text_pt": "B same", "intent": 2},  # PT identical to EN
              {"text": "a 0", "text_pt": "outra", "intent": 0}])     # duplicated EN text


def test_clinc_alignment_exclusions_and_stratified_sample():
    clinc.check_aligned([r["text"] for r in PT_ROWS], [r["intent"] for r in PT_ROWS], PT_ROWS)
    with pytest.raises(ValueError, match="not aligned"):
        clinc.check_aligned(["other"] + [r["text"] for r in PT_ROWS[1:]], [r["intent"] for r in PT_ROWS], PT_ROWS)
    excluded = clinc.exclusions(PT_ROWS, NAMES)
    assert excluded == {0: "duplicate_en_text", 9: "pt_identical_to_en", 10: "duplicate_en_text"}
    picked = clinc.stratified_sample(PT_ROWS, NAMES, excluded, per_intent=2)
    assert len(picked) == 4 and not set(picked) & set(excluded) and {PT_ROWS[i]["intent"] for i in picked} == {0, 2}
    assert picked == clinc.stratified_sample(PT_ROWS, NAMES, excluded, per_intent=2)  # seeded


def test_clinc_intents_need_exact_and_distinct_pt_names():
    intents = clinc.build_intents(NAMES, {"a_intent": "intenção a", "b_intent": "intenção b"})
    assert [r["label_name"] for r in intents] == ["a_intent", "b_intent"] and intents[1]["label_id"] == 2
    with pytest.raises(ValueError, match="cover exactly"):
        clinc.build_intents(NAMES, {"a_intent": "x"})
    with pytest.raises(ValueError, match="not distinct"):
        clinc.build_intents(NAMES, {"a_intent": "igual", "b_intent": "igual"})


def test_shipped_clinc_pt_names_cover_the_versioned_intents():
    names = json.loads((ROOT / "data/clinc150/intent_names_pt.json").read_text(encoding="utf-8"))
    intents = json.loads((ROOT / "data/clinc150/intents_clinc150.json").read_text(encoding="utf-8"))
    assert {r["label_name"] for r in intents} == {k for k in names if not k.startswith("_")} and len(intents) == 150


def test_sib_pairs_by_id_and_refuses_mismatches():
    en = [{"index_id": 1, "category": "travel", "text": "Go"}, {"index_id": 2, "category": "health", "text": "Eat"}]
    pt = [{"index_id": 2, "category": "health", "text": "Coma"}, {"index_id": 1, "category": "travel", "text": "go"}]
    pairs, excluded = sib.pair_by_id(en, pt)
    assert [p[0]["index_id"] for p in pairs] == [2] and excluded[0]["reason"] == "pt_identical_to_en"
    with pytest.raises(ValueError, match="category differs"):
        sib.pair_by_id(en, [{**pt[0], "category": "travel"}, pt[1]])
    with pytest.raises(ValueError, match="different index_id"):
        sib.pair_by_id(en, pt[:1])
    sample = sib.build_sample(pairs)
    assert sample[0]["label_name"] == "health" and sample[0]["translation_status"] == "human_translated"


@pytest.mark.parametrize("name", ["clinc150", "sib200"])
def test_new_datasets_load_and_have_frozen_templates(name):
    config = load_config(ROOT / "configs" / "final.yaml")
    spec = next(s for s in config.datasets if s.name == name)
    adapter = build_adapter(spec)
    examples = adapter.load_examples()
    assert examples and set(adapter.options("en")) == set(adapter.options("pt_br")) >= {e.label for e in examples}
    for template in (TEMPLATES[name], VARIANTS["codes"][name]):
        assert fingerprint(template) == FROZEN_FINGERPRINTS[(name, template.version)]


def test_merge_keeps_valid_answers_and_refuses_conflicts():
    def row(system, error=None, pred="a", h="h"):
        return {"dataset": "d", "example_id": "e", "condition": "C1", "system": system, "error": error,
                "predicted_label_name": pred, "confidence": 0.5, "input_hash": h}
    merged = merge_script.merge([[row("jev"), row("qwen_prob", error="x")], [row("qwen_prob"), row("jev")]])
    assert sorted((r["system"], r["error"]) for r in merged) == [("jev", None), ("qwen_prob", None)]
    with pytest.raises(ValueError, match="conflicting"):
        merge_script.merge([[row("jev")], [row("jev", h="other")]])


def test_partial_system_run_keeps_other_systems_failed_rows(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    patch_clients(monkeypatch, {"jev": RecordingClient(fail_ids={"olá 1"}), "llm": RecordingClient()})
    ri.run_inference(config)
    config.systems = ["llm"]
    patch_clients(monkeypatch, {"llm": RecordingClient()})
    ri.run_inference(config)
    rows = ri.read_jsonl(config.paths["raw_predictions"])
    assert sum(r["system"] == "jev" and r["error"] is not None for r in rows) == 1 and len(rows) == 8
