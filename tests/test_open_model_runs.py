from pathlib import Path

import pytest

from jev_eval.config import load_config
from jev_eval.data.io import read_jsonl
from jev_eval.pipeline import run_inference as ri
from jev_eval.systems import is_open_model_system, system_parts

ROOT = Path(__file__).resolve().parent.parent


def write_config(tmp_path: Path, systems="[jev, gemma_prob, gemma_verb]", models=None) -> Path:
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "c.yaml"
    models = models if models is not None else (
        "open_models:\n  gemma: {hf_id: org/gemma, revision: abc123}\n")
    path.write_text(f"""seed: 1
sample_size: 2
ece_bins: 5
conditions: [C1, C2]
systems: {systems}
{models}datasets:
  - name: banking77
    task: choice
    paths: {{sample: data/s.jsonl, intents: data/i.json}}
paths: {{raw_predictions: outputs/r.jsonl}}
""", encoding="utf-8")
    return path


def test_system_names_split_into_model_and_readout():
    assert system_parts("jev") == ("jev", "prob") and system_parts("llm") == ("llm", "verb")
    assert system_parts("gemma_prob") == ("gemma", "prob") and system_parts("qwen_x_verb") == ("qwen_x", "verb")
    assert [is_open_model_system(s) for s in ("jev", "llm", "gemma_prob")] == [False, False, True]
    for bad in ("gemma", "gemma_logits", "_prob"):
        with pytest.raises(ValueError, match="Unknown system"):
            system_parts(bad)


def test_config_loads_open_models(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.open_models["gemma"].hf_id == "org/gemma" and config.open_models["gemma"].dtype == "bfloat16"


def test_open_model_system_needs_its_model_entry(tmp_path):
    with pytest.raises(ValueError, match="open_models.gemma"):
        load_config(write_config(tmp_path, models=""))


def test_open_model_revision_must_be_pinned(tmp_path):
    with pytest.raises(ValueError, match="pin a 'revision'"):
        load_config(write_config(tmp_path, models="open_models:\n  gemma: {hf_id: org/gemma}\n"))


def test_shipped_open_models_config_loads():
    config = load_config(ROOT / "configs" / "open_models.yaml")
    assert set(config.open_models) == {"gemma", "qwen"} and "gemma_prob" in config.systems


def mock_open_config(tmp_path):
    config = load_config(ROOT / "configs" / "smoke.yaml")
    config.systems = ["jev", "llm", "gemma_prob", "gemma_verb"]
    config.conditions = ["C1", "C2"]
    config.open_models = load_config(ROOT / "configs" / "open_models.yaml").open_models
    config.paths["raw_predictions"] = tmp_path / "raw.jsonl"
    return config


def test_mock_run_covers_every_system_and_uses_the_right_prompt(tmp_path):
    config = mock_open_config(tmp_path)
    ri.run_inference(config, mock=True)
    rows = read_jsonl(config.paths["raw_predictions"])
    assert len(rows) == 6 * 2 * 4 and all(r["error"] is None for r in rows)
    versions = {r["system"]: r["prompt_version"] for r in rows}
    assert versions["gemma_prob"] == "choice-codes-v1" and versions["gemma_verb"] == "choice-v1"
    assert versions["jev"] == versions["llm"] == "choice-v1"
    assert {r["model_id"] for r in rows if r["system"].startswith("gemma")} == {"mock-gemma-prob", "mock-gemma-verb"}


def test_prompts_differ_by_readout_and_resume_keeps_rows(tmp_path):
    config = mock_open_config(tmp_path)
    prob = ri.prompt_for("banking77", ri.get_condition("C2"), "gemma_prob", config)
    verb = ri.prompt_for("banking77", ri.get_condition("C2"), "gemma_verb", config)
    assert prob.answer == "code" and verb.answer == "single"
    ri.run_inference(config, mock=True)
    again = ri.run_inference(config, mock=True)  # same inputs: nothing recomputed, nothing mixed
    assert len(again) == 6 * 2 * 4
