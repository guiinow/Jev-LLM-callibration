import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from jev_eval.types import SampleExample


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(asdict(row) if hasattr(row, "__dataclass_fields__") else row,
                                  ensure_ascii=False) + "\n")


def load_examples(path: Path, limit: int | None = None) -> list[SampleExample]:
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    return [SampleExample(**row) for row in rows]

