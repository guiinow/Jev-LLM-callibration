"""Frozen, versioned prompt templates.

Each template is keyed by dataset and holds the instruction text for every instruction
language. `TEMPLATES` holds the main prompt (one intent + verbalized confidence);
`VARIANTS` holds secondary prompts for ablations (E1: top-k guesses), keyed by name. Editing a template's text changes its fingerprint: bump `version` and update
`FROZEN_FINGERPRINTS`, otherwise tests/test_templates.py fails.
"""
import hashlib
import json
from dataclasses import dataclass

from jev_eval.conditions import Condition


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    instructions: dict[str, str]  # short question text sent to Jev
    llm: dict[str, str]  # full LLM prompt with {content_language}, {options}, {message}
    answer: str = "single"  # LLM answer format: "single" (intent + confidence) or "topk" (guesses)


# Name of the content language, written in the instruction language.
LANGUAGE_NAMES = {
    "en": {"en": "English", "pt_br": "Brazilian Portuguese"},
    "pt_br": {"en": "inglês", "pt_br": "português do Brasil"},
}

TEMPLATES: dict[str, PromptTemplate] = {
    "banking77": PromptTemplate(
        version="choice-v1",
        instructions={
            "en": "Choose the banking intent.",
            "pt_br": "Escolha a intenção bancária.",
        },
        llm={
            "en": ("Classify this banking message written in {content_language}. Return only JSON with "
                   'exact keys "intent" and "confidence", where confidence is 0 to 1.\n'
                   "Options:\n{options}\nMessage: {message}"),
            "pt_br": ("Classifique esta mensagem bancária escrita em {content_language}. Retorne apenas "
                      'JSON com as chaves exatas "intent" e "confidence", em que confidence é um '
                      "número de 0 a 1.\nOpções:\n{options}\nMensagem: {message}"),
        },
    ),
}

TOPK = 3

# E1 ablation, after Tian et al. (2023) "Verb. 1S top-k": k guesses, each with the probability
# that it is correct. Same options/message layout as the main prompt. Jev is not run with it.
VARIANTS: dict[str, dict[str, PromptTemplate]] = {
    "topk": {
        "banking77": PromptTemplate(
            version="choice-topk-v1",
            instructions=TEMPLATES["banking77"].instructions,
            llm={
                "en": ("Classify this banking message written in {content_language}. Give your 3 best "
                       "guesses for the intent and, for each, the probability (0 to 1) that it is correct. "
                       'Return only JSON with exact key "guesses": a list of 3 objects with exact keys '
                       '"intent" and "probability", most likely first.\n'
                       "Options:\n{options}\nMessage: {message}"),
                "pt_br": ("Classifique esta mensagem bancária escrita em {content_language}. Dê seus 3 "
                          "melhores palpites para a intenção e, para cada um, a probabilidade (de 0 a 1) de "
                          'que esteja correto. Retorne apenas JSON com a chave exata "guesses": uma lista de '
                          '3 objetos com as chaves exatas "intent" e "probability", do mais provável para o '
                          "menos provável.\nOpções:\n{options}\nMensagem: {message}"),
            },
            answer="topk",
        ),
    },
}

FROZEN_FINGERPRINTS = {
    ("banking77", "choice-v1"): "0f9329ed6353",
    ("banking77", "choice-topk-v1"): "11c1d5fc8e77",
}


def fingerprint(template: PromptTemplate) -> str:
    payload = json.dumps({"version": template.version, "instructions": template.instructions,
                          "llm": template.llm}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class TaskPrompt:
    """A template resolved for one (dataset, condition)."""
    condition: str
    version: str
    instructions: str
    llm_template: str
    content_language_name: str
    answer: str = "single"

    def render_llm(self, message: str, criteria: dict[str, str]) -> str:
        options = "\n".join(f"- {label}: {description}" for label, description in criteria.items())
        return self.llm_template.format(content_language=self.content_language_name,
                                        options=options, message=message)


def build_task_prompt(dataset: str, condition: Condition, variant: str | None = None) -> TaskPrompt:
    """`variant=None` is the main prompt; otherwise a key of VARIANTS (e.g. "topk")."""
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"Unknown prompt variant {variant!r}; known: {sorted(VARIANTS)}.")
    templates = TEMPLATES if variant is None else VARIANTS[variant]
    if dataset not in templates:
        raise ValueError(f"No prompt template for dataset {dataset!r}; known: {sorted(templates)}.")
    template = templates[dataset]
    lang = condition.instruction_language
    return TaskPrompt(
        condition=condition.name,
        version=template.version,
        instructions=template.instructions[lang],
        llm_template=template.llm[lang],
        content_language_name=LANGUAGE_NAMES[lang][condition.content_language],
        answer=template.answer,
    )
