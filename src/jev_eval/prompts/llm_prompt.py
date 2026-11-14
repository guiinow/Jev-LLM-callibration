PROMPT_VERSION = "llm-intent-json-v1"

LANGUAGE_NAMES = {"en": "English", "pt_br": "Brazilian Portuguese"}


def build_prompt(message: str, criteria: dict[str, str], language: str) -> str:
    """Frozen v1 prompt text. Do not edit: variants get a new PROMPT_VERSION and function."""
    options = "\n".join(f"- {label}: {description}" for label, description in criteria.items())
    return (
        f"Classify this banking message written in {LANGUAGE_NAMES.get(language, language)}. Return only JSON with exact keys "
        f'"intent" and "confidence", where confidence is 0 to 1.\nOptions:\n{options}\nMessage: {message}'
    )


TOPK_PROMPT_VERSION = "llm-intent-topk-v1"
TOPK = 3


def build_topk_prompt(message: str, criteria: dict[str, str], language: str) -> str:
    """E1 ablation, after Tian et al. (2023) "Verb. 1S top-k": k guesses, each with the
    probability that it is correct. Same options/message layout as v1. Frozen once used."""
    options = "\n".join(f"- {label}: {description}" for label, description in criteria.items())
    return (
        f"Classify this banking message written in {LANGUAGE_NAMES.get(language, language)}. Give your {TOPK} best "
        f"guesses for the intent and, for each, the probability (0 to 1) that it is correct. Return only JSON "
        f'with exact key "guesses": a list of {TOPK} objects with exact keys "intent" and "probability", '
        f"most likely first.\nOptions:\n{options}\nMessage: {message}"
    )


PROMPTS = {"v1": (PROMPT_VERSION, build_prompt), "topk": (TOPK_PROMPT_VERSION, build_topk_prompt)}
