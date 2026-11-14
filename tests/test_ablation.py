"""E1: top-k prompt parsing, and the ablation reusing a baseline run's Jev/v1 answers."""
import json

import pytest

from jev_eval.clients.llm_client import parse_topk_json
from jev_eval.pipeline import run_inference as ri
from jev_eval.pipeline.ablation import ablation, ablation_tables, load_baseline, with_baseline_jev
from jev_eval.prompts.llm_prompt import PROMPTS, TOPK_PROMPT_VERSION
from test_pipeline import make_config
from test_provenance import fake_llm

LABELS = {"a", "b", "c"}


def topk(*guesses):
    return json.dumps({"guesses": [{"intent": i, "probability": p} for i, p in guesses]})


def test_topk_parser_takes_highest_probability_guess():
    assert parse_topk_json("```json\n" + topk(("a", 0.6), ("b", 0.3)) + "\n```", LABELS) == ("a", 0.6)
    assert parse_topk_json(topk(("b", 0.2), ("c", 0.5), ("a", 0.5)), LABELS) == ("c", 0.5)  # first max wins


@pytest.mark.parametrize("text", [
    topk(("a", 0.6), ("zzz", 0.3)),          # invalid intent anywhere
    topk(("a", 0.6), ("a", 0.3)),            # repeated intent
    topk(("a", 1.2)),                        # probability out of range
    topk(("a", 0.5), ("b", 0.2), ("c", 0.1), ("a", 0.1)),  # more than k guesses
    '{"intent": "a", "confidence": 0.9}',    # v1 format
])
def test_topk_parser_rejects_bad_answers(text):
    with pytest.raises(ValueError):
        parse_topk_json(text, LABELS)


def test_llm_client_topk_prompt_and_version():
    client = fake_llm(topk(("a", 0.7), ("b", 0.2)), 1500)
    client.prompt, client.prompt_version = "topk", TOPK_PROMPT_VERSION
    result = client.classify("msg", {"a": "x", "b": "y"}, "en")
    assert (result.predicted_label, result.confidence, result.prompt_version) == ("a", 0.7, TOPK_PROMPT_VERSION)
    assert '"guesses"' in PROMPTS["topk"][1]("msg", {"a": "x"}, "en")


def run_baseline_and_ablation(tmp_path):
    base = make_config(tmp_path)
    ri.run_inference(base, mock=True)
    abl = make_config(tmp_path)
    abl.systems, abl.llm = ["llm"], {"prompt": "topk"}
    abl.paths = {**abl.paths, "baseline": base.paths["raw_predictions"],
                 "raw_predictions": tmp_path / "abl" / "raw.jsonl"}
    ri.run_inference(abl, mock=True)
    return base, abl


def test_ablation_reuses_baseline_without_calling_jev(tmp_path):
    base, abl = run_baseline_and_ablation(tmp_path)
    own = ri.read_jsonl(abl.paths["raw_predictions"])
    assert {r["system"] for r in own} == {"llm"} and {r["prompt_version"] for r in own} == {TOPK_PROMPT_VERSION}
    baseline = load_baseline(abl)
    merged = with_baseline_jev(own, baseline)
    assert sorted({r["system"] for r in merged}) == ["jev", "llm"] and len(merged) == 8
    result = ablation(own, baseline, abl.ece_bins, n_bootstrap=50)
    assert set(result["languages"]["pt_br"]["comparisons"]) == {"llm_topk - llm_v1", "jev - llm_topk",
                                                              "jev - llm_v1"}
    header, rows, _, deltas = ablation_tables(result)
    assert len(rows) == 2 * 3 and len(deltas) == 2 * 3 and rows[0][2] == 2


def test_ablation_refuses_baseline_from_other_inputs_or_incomplete(tmp_path):
    base, abl = run_baseline_and_ablation(tmp_path)
    rows = [json.loads(line) for line in abl.paths["sample"].read_text().splitlines()]
    rows[0]["text_pt_br"] = "olá revisado"
    abl.paths["sample"].write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="other messages"):
        load_baseline(abl)
    rows[0]["text_pt_br"] = "olá 0"
    abl.paths["sample"].write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    kept = ri.read_jsonl(base.paths["raw_predictions"])[1:]
    ri.write_jsonl(base.paths["raw_predictions"], kept)
    with pytest.raises(ValueError, match="incomplete"):
        load_baseline(abl)
