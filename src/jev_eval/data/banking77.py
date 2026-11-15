import json

from jev_eval.data.datasets import DatasetAdapter, Example, register_adapter
from jev_eval.data.io import read_jsonl

DESCRIPTION_FIELD = {"en": "description_en", "pt_br": "description_pt_br"}


@register_adapter("banking77")
@register_adapter("clinc150")
@register_adapter("sib200")
class Banking77Adapter(DatasetAdapter):
    """A local EN/PT sample (`sample`: text_en, text_pt_br, label_name) plus the option descriptions
    (`intents`: description_en, description_pt_br). Banking77 (scripts/build_usp_data.py), CLINC150
    (scripts/build_clinc150_data.py) and SIB-200 (scripts/build_sib200_data.py) share this format."""

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
