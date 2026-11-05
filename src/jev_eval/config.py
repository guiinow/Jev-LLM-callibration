from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvalConfig:
    seed: int
    sample_size: int
    ece_bins: int
    languages: list[str]
    systems: list[str]
    paths: dict[str, Path]
    config_path: Path


def load_config(path: str | Path) -> EvalConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)
    base = config_path.parent.parent
    paths = {key: base / value for key, value in raw["paths"].items()}
    return EvalConfig(
        seed=raw["seed"],
        sample_size=raw["sample_size"],
        ece_bins=raw["ece_bins"],
        languages=raw["languages"],
        systems=raw["systems"],
        paths=paths,
        config_path=config_path,
    )

