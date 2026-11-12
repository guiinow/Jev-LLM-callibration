PROMPT_VERSION = "llm-intent-json-v1"

LANGUAGE_NAMES = {"en": "English", "pt_br": "Brazilian Portuguese"}


def build_prompt(message: str, criteria: dict[str, str], language: str) -> str:
    """Frozen v1 prompt text. Do not edit: variants get a new PROMPT_VERSION and function."""
    options = "\n".join(f"- {label}: {description}" for label, description in criteria.items())
    return (
        f"Classify this banking message written in {LANGUAGE_NAMES.get(language, language)}. Return only JSON with exact keys "
        f'"intent" and "confidence", where confidence is 0 to 1.\nOptions:\n{options}\nMessage: {message}'
    )
