import pytest

from jev_eval.conditions import CONDITIONS
from jev_eval.prompts.templates import FROZEN_FINGERPRINTS, TEMPLATES, build_task_prompt, fingerprint


@pytest.mark.parametrize("dataset", sorted(TEMPLATES))
def test_templates_are_frozen(dataset):
    template = TEMPLATES[dataset]
    assert fingerprint(template) == FROZEN_FINGERPRINTS[(dataset, template.version)], (
        "Template text changed: bump its version and add the new fingerprint to FROZEN_FINGERPRINTS.")


def test_c1_prompt_matches_the_original_english_prompt():
    prompt = build_task_prompt("banking77", CONDITIONS["C1"])
    assert prompt.render_llm("hi", {"a": "option A"}) == (
        "Classify this banking message written in English. Return only JSON with exact keys "
        '"intent" and "confidence", where confidence is 0 to 1.\nOptions:\n- a: option A\nMessage: hi')


def test_instruction_language_follows_condition():
    c2 = build_task_prompt("banking77", CONDITIONS["C2"])
    c3 = build_task_prompt("banking77", CONDITIONS["C3"])
    assert "written in Brazilian Portuguese" in c2.render_llm("oi", {"a": "x"})
    assert c2.instructions == "Choose the banking intent."
    assert "português do Brasil" in c3.render_llm("oi", {"a": "x"})
    assert c3.instructions == "Escolha a intenção bancária."


def test_braces_in_message_are_not_interpreted():
    prompt = build_task_prompt("banking77", CONDITIONS["C1"])
    assert "Message: {oops}" in prompt.render_llm("{oops}", {"a": "x"})


def test_unknown_dataset_has_no_template():
    with pytest.raises(ValueError, match="No prompt template"):
        build_task_prompt("nope", CONDITIONS["C1"])
