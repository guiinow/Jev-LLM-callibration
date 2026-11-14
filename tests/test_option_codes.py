import pytest

from jev_eval.clients.option_codes import assign_codes, code_candidates
from jev_eval.conditions import CONDITIONS
from jev_eval.prompts.templates import FROZEN_FINGERPRINTS, VARIANTS, build_task_prompt, fingerprint


def test_candidates_start_with_letters_then_pairs():
    codes = list(code_candidates())
    assert codes[:3] == ["A", "B", "C"] and codes[25] == "Z" and codes[26] == "AA" and codes[27] == "AB"
    assert len(codes) == 26 + 26 * 26 and len(set(codes)) == len(codes)


def test_codes_follow_label_order():
    assert assign_codes(["x", "y", "z"], lambda code: True) == {"x": "A", "y": "B", "z": "C"}


def test_multi_token_codes_are_skipped():
    assert assign_codes(["x", "y", "z"], lambda code: code != "B") == {"x": "A", "y": "C", "z": "D"}


def test_77_options_need_codes_beyond_the_alphabet():
    codes = assign_codes([f"l{i}" for i in range(77)], lambda code: True)
    assert list(codes.values())[25] == "Z" and list(codes.values())[26] == "AA" and len(set(codes.values())) == 77


def test_not_enough_codes_is_an_error():
    with pytest.raises(ValueError, match="single-token codes"):
        assign_codes(["x", "y"], lambda code: code == "A")


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_variant_templates_are_frozen(variant):
    for dataset, template in VARIANTS[variant].items():
        assert fingerprint(template) == FROZEN_FINGERPRINTS[(dataset, template.version)], (
            "Template text changed: bump its version and add the new fingerprint to FROZEN_FINGERPRINTS.")


def test_codes_prompt_lists_codes_and_asks_for_the_code_only():
    prompt = build_task_prompt("banking77", CONDITIONS["C2"], "codes")
    text = prompt.render_llm("oi", {"A": "ativar cartão", "B": "limite de idade"})
    assert prompt.answer == "code" and prompt.version == "choice-codes-v1"
    assert "- A: ativar cartão\n- B: limite de idade" in text
    assert "Reply with the code of the correct option only" in text  # C2: EN instructions
    assert "written in Brazilian Portuguese" in text
    c3 = build_task_prompt("banking77", CONDITIONS["C3"], "codes").render_llm("oi", {"A": "x"})
    assert "Responda apenas com o código" in c3
