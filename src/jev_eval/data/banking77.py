import json

from jev_eval.data.datasets import DatasetAdapter, Example, register_adapter
from jev_eval.data.io import read_jsonl

DESCRIPTION_FIELD = {"en": "description_en", "pt_br": "description_pt_br"}


@register_adapter("banking77")
class Banking77Adapter(DatasetAdapter):
    """Banking77 from the local stratified sample (`sample`) plus the intent descriptions (`intents`)."""

    def load_examples(self, limit: int | None = None) -> list[Example]:
        rows = read_jsonl(self.spec.paths["sample"])
        if limit is not None:
            rows = rows[:limit]
        return [Example(row["example_id"], row["label_name"],
                        {"en": row["text_en"], "pt_br": row["text_pt_br"]},
                        row.get("translation_status")) for row in rows]

    def options(self, language: str) -> dict[str, str]:
        intents = json.loads(self.spec.paths["intents"].read_text(encoding="utf-8"))
        field = DESCRIPTION_FIELD[language]
        return {row["label_name"]: row.get(field, "") for row in intents}
