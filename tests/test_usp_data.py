"""scripts/build_usp_data.py: refuses misaligned USP data, matches the sample by English text."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "build_usp_data", Path(__file__).parents[1] / "scripts" / "build_usp_data.py")
usp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(usp)

CARD = "| rótulo | intenção |\n|---|---|\n" + "".join(f"| {i} | nome_{i} |\n" for i in range(77))
PT_IDS = usp.card_label_names(CARD)
LABELS = [f"label_{i}" for i in range(77)]
EN_TEXTS = ["hello", "card lost", "fee?"]
EN_LABELS = [0, 1, 1]
PT_ROWS = [{"frase": f, "rotulo": f"nome_{l}", "alterado": a, "modelo_tradutor": "gpt"}
           for f, l, a in [("olá", 0, 0), ("perdi o cartão", 1, 1), ("tarifa?", 1, 0)]]
SAMPLE = [{"example_id": "test-0000", "split": "test", "label_id": 1, "label_name": "label_1",
           "text_en": "fee?", "text_pt_br": "", "translation_status": "pending_translation"}]
INTENTS = [{"label_id": i, "label_name": f"label_{i}", "description_en": f"label {i}",
            "description_pt_br": "x", "description_mt_model": "gemma"} for i in range(77)]


def test_card_table_needs_all_77_intents():
    with pytest.raises(ValueError):
        usp.card_label_names("| 0 | a |\n")


def test_empty_tail_dropped_but_nulls_inside_refused():
    tail = {"frase": None, "rotulo": None}
    assert usp.drop_empty_tail(PT_ROWS + [tail, tail]) == PT_ROWS
    with pytest.raises(ValueError):
        usp.drop_empty_tail([PT_ROWS[0], tail, PT_ROWS[1]])


def test_order_check_refuses_different_label_sequence():
    usp.check_same_order(EN_LABELS, PT_ROWS, PT_IDS)
    with pytest.raises(ValueError, match="order differs"):
        usp.check_same_order([1, 0, 1], PT_ROWS, PT_IDS)


def test_build_takes_sentence_at_matched_index():
    sample, intents, alignment = usp.build(SAMPLE, INTENTS, EN_TEXTS, EN_LABELS, PT_ROWS, PT_IDS, LABELS)
    assert sample[0]["text_pt_br"] == "tarifa?" and sample[0]["translation_status"] == "human_reviewed"
    assert alignment[0] == {"example_id": "test-0000", "test_index": 2, "rotulo": "nome_1",
                            "alterado": 0, "modelo_tradutor": "gpt"}
    assert intents[5]["description_pt_br"] == "nome 5" and "description_mt_model" not in intents[5]


def test_build_refuses_label_mismatch_and_unknown_text():
    with pytest.raises(ValueError, match="label mismatch"):
        usp.build([{**SAMPLE[0], "label_id": 0, "label_name": "label_0"}], INTENTS, EN_TEXTS, EN_LABELS,
                  PT_ROWS, PT_IDS, LABELS)
    with pytest.raises(ValueError, match="not found"):
        usp.build([{**SAMPLE[0], "text_en": "nope"}], INTENTS, EN_TEXTS, EN_LABELS, PT_ROWS, PT_IDS, LABELS)
