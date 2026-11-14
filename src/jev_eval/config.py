from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from jev_eval.conditions import get_condition
from jev_eval.data.datasets import DatasetSpec, SourceSpec, TaskType


@dataclass
class EvalConfig:
    seed: int
    sample_size: int
    ece_bins: int
    conditions: list[str]
    systems: list[str]
    datasets: list[DatasetSpec]
    paths: dict[str, Path]
    config_path: Path
    llm: dict[str, Any] = field(default_factory=dict)
    jev: dict[str, Any] = field(default_factory=dict)

    @property
    def output_dir(self) -> Path:
        return self.paths["raw_predictions"].parent


def _dataset_spec(raw: dict[str, Any], base: Path) -> DatasetSpec:
    source = None
    if raw.get("source"):
        if not raw["source"].get("revision"):
            raise ValueError(f"Dataset {raw['name']!r}: pin a 'revision' in its source "
                             f"({raw['source'].get('hf_id')}) so the snapshot is reproducible.")
        source = SourceSpec(**raw["source"])
    return DatasetSpec(
        name=raw["name"],
        task=TaskType(raw["task"]),
        paths={key: base / value for key, value in (raw.get("paths") or {}).items()},
        source=source,
    )


def load_config(path: str | Path) -> EvalConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)
    base = config_path.parent.parent
    for name in raw["conditions"]:
        get_condition(name)  # fail fast on typos
    datasets = [_dataset_spec(item, base) for item in raw["datasets"]]
    if not datasets:
        raise ValueError("Config must list at least one dataset.")
    return EvalConfig(
        seed=raw["seed"],
        sample_size=raw["sample_size"],
        ece_bins=raw["ece_bins"],
        conditions=raw["conditions"],
        systems=raw["systems"],
        datasets=datasets,
        paths={key: base / value for key, value in raw["paths"].items()},
        config_path=config_path,
        llm=raw.get("llm") or {},
        jev=raw.get("jev") or {},
    )
