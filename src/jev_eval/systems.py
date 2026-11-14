"""System names: "jev", "llm", and "<model>_prob" / "<model>_verb" for the open models."""

READOUTS = ("prob", "verb")  # option probabilities vs confidence written by the model
BUILTIN = {"jev": ("jev", "prob"), "llm": ("llm", "verb")}


def system_parts(system: str) -> tuple[str, str]:
    """(model, readout) of a system name; the built-in Jev reads probabilities, the other LLM verbalizes."""
    if system in BUILTIN:
        return BUILTIN[system]
    model, _, readout = system.rpartition("_")
    if not model or readout not in READOUTS:
        raise ValueError(f"Unknown system {system!r}: use 'jev', 'llm' or '<model>_prob' / '<model>_verb'.")
    return model, readout


def is_open_model_system(system: str) -> bool:
    return system not in BUILTIN
