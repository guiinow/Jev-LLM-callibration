import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from jev_eval.types import SampleExample


def to_jsonable(value: Any) -> Any:
    """Best-effort conversion of SDK response objects into JSON-serializable data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    for method in ("model_dump", "dict", "to_dict"):
        if callable(getattr(value, method, None)):
            try:
                return to_jsonable(getattr(value, method)())
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        return to_jsonable({k: v for k, v in vars(value).items() if not k.startswith("_")})
    return repr(value)


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

