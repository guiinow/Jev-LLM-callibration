from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


class TaskType(str, Enum):
    CHOICE = "choice"
    BINARY = "binary"
    MULTILABEL = "multilabel"


@dataclass(frozen=True)
class SourceSpec:
    """Where a dataset comes from on the Hugging Face Hub; `revision` pins the exact snapshot."""
    hf_id: str
    split: str
    revision: str
    config: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: TaskType
    paths: dict[str, Path] = field(default_factory=dict)
    source: SourceSpec | None = None


@dataclass
class Example:
    """One item with its content in every available language."""
    example_id: str
    label: str
    text: dict[str, str]
    translation_status: str | None = None


class DatasetAdapter(ABC):
    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @abstractmethod
    def load_examples(self, limit: int | None = None) -> list[Example]:
        raise NotImplementedError

    @abstractmethod
    def options(self, language: str) -> dict[str, str]:
        """Label -> description in `language` (choice tasks only)."""
        raise NotImplementedError


_ADAPTERS: dict[str, Callable[[DatasetSpec], DatasetAdapter]] = {}


def register_adapter(name: str):
    def decorator(factory):
        _ADAPTERS[name] = factory
        return factory
    return decorator


def build_adapter(spec: DatasetSpec) -> DatasetAdapter:
    # Imported for its side effect: adapters register themselves.
    from jev_eval.data import banking77  # noqa: F401
    if spec.name not in _ADAPTERS:
        raise ValueError(f"No adapter registered for dataset {spec.name!r}; known: {sorted(_ADAPTERS)}.")
    return _ADAPTERS[spec.name](spec)
