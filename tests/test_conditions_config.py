from pathlib import Path

import pytest

from jev_eval.conditions import CONDITIONS, get_condition
from jev_eval.config import load_config
from jev_eval.data.datasets import TaskType, build_adapter

ROOT = Path(__file__).resolve().parent.parent


def write_config(tmp_path: Path, conditions="[C1, C2]", source="") -> Path:
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "c.yaml"
    path.write_text(f"""seed: 1
sample_size: 2
ece_bins: 5
conditions: {conditions}
systems: [jev, llm]
datasets:
  - name: banking77
    task: choice
{source}    paths: {{sample: data/s.jsonl, intents: data/i.json}}
paths: {{raw_predictions: outputs/r.jsonl}}
""", encoding="utf-8")
    return path


def test_condition_definitions():
    assert [(c.content_language, c.instruction_language) for c in CONDITIONS.values()] == [
        ("en", "en"), ("pt_br", "en"), ("pt_br", "pt_br")]
    with pytest.raises(ValueError, match="Unknown condition"):
        get_condition("C9")


def test_config_loads_datasets_and_conditions(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.conditions == ["C1", "C2"]
    spec = config.datasets[0]
    assert spec.task is TaskType.CHOICE and spec.paths["sample"] == tmp_path / "data" / "s.jsonl"
    assert config.output_dir == tmp_path / "outputs"


def test_config_rejects_unknown_condition(tmp_path):
    with pytest.raises(ValueError, match="Unknown condition"):
        load_config(write_config(tmp_path, conditions="[C1, C4]"))


def test_config_requires_pinned_revision(tmp_path):
    source = "    source: {hf_id: PolyAI/banking77, split: test}\n"
    with pytest.raises(ValueError, match="pin a 'revision'"):
        load_config(write_config(tmp_path, source=source))


@pytest.mark.parametrize("name", ["default", "smoke", "pilot", "full_local", "final", "final_timing", "ablation_topk"])
def test_shipped_configs_load(name):
    config = load_config(ROOT / "configs" / f"{name}.yaml")
    assert config.datasets and set(config.conditions) <= set(CONDITIONS)


def test_banking77_adapter_reads_both_languages():
    adapter = build_adapter(load_config(ROOT / "configs" / "smoke.yaml").datasets[0])
    example = adapter.load_examples(1)[0]
    assert example.text["en"] and example.text["pt_br"] and example.label
    assert set(adapter.options("en")) == set(adapter.options("pt_br"))
