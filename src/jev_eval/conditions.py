from dataclasses import dataclass

EN, PT_BR = "en", "pt_br"


@dataclass(frozen=True)
class Condition:
    """One linguistic setup. `content_language` is the language of the item text;
    `instruction_language` is the language of the task instructions and option descriptions."""
    name: str
    content_language: str
    instruction_language: str
    description: str


CONDITIONS: dict[str, Condition] = {
    "C1": Condition("C1", EN, EN, "EN content + EN instructions (baseline)"),
    "C2": Condition("C2", PT_BR, EN, "PT content + EN instructions/options (primary)"),
    "C3": Condition("C3", PT_BR, PT_BR, "PT content + PT instructions/options (secondary)"),
}


def get_condition(name: str) -> Condition:
    try:
        return CONDITIONS[name]
    except KeyError:
        raise ValueError(f"Unknown condition {name!r}; expected one of {sorted(CONDITIONS)}.") from None
