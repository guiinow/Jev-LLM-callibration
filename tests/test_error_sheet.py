"""E3: error sheet for manual categorization and its counts."""
import pytest

from jev_eval.data.datasets import Example
from jev_eval.pipeline.error_sheet import (CATEGORIES, error_rows, read_sheet, summarize, summary_table,
                                           write_sheet)


def rec(eid, system, pred, ok, conf, condition="C3"):
    return {"dataset": "banking77", "condition": condition, "example_id": eid, "system": system,
            "true_label_name": "a", "predicted_label_name": pred, "correct": ok, "confidence": conf, "error": None}


EXAMPLES = [Example("e0", "a", {"en": "hello", "pt_br": "olá"}), Example("e1", "a", {"en": "bye", "pt_br": "tchau"})]
RECORDS = [rec("e0", "jev", "b", False, 0.95), rec("e0", "llm", "a", True, 0.9),
           rec("e1", "jev", "a", True, 0.8), rec("e1", "llm", "c", False, 0.5),
           rec("e1", "llm", "c", False, 0.5, condition="C1"),
           {"dataset": "banking77", "condition": "C3", "example_id": "e1", "system": "gemma_prob", "error": "x"}]


def test_error_rows_list_only_wrong_answers_of_the_condition(tmp_path):
    rows = error_rows(RECORDS, EXAMPLES, "banking77", "C3", ["jev", "llm"])
    assert [(r["example_id"], r["system"]) for r in rows] == [("e0", "jev"), ("e1", "llm")]
    assert rows[0]["text"] == "olá" and rows[0]["high_confidence"] is True and "llm: a (ok, 0.90)" in rows[0]["others"]
    path = tmp_path / "s.csv"
    write_sheet(path, rows)
    with pytest.raises(FileExistsError):
        write_sheet(path, rows)  # never overwrite a sheet that may already be filled


def test_summary_counts_and_refuses_blank_categories(tmp_path):
    path = tmp_path / "s.csv"
    write_sheet(path, error_rows(RECORDS, EXAMPLES, "banking77", "C3", ["jev", "llm"]))
    rows = read_sheet(path)
    with pytest.raises(ValueError, match="blank or unknown"):
        summarize(rows)
    rows[0]["category"], rows[1]["category"] = "intencoes_proximas", "traducao"
    summary = summarize(rows)
    assert summary["by_system"] == {"jev": {"intencoes_proximas": 1}, "llm": {"traducao": 1}}
    assert summary["high_confidence_by_system"] == {"jev": {"intencoes_proximas": 1}}
    header, table = summary_table(summary)
    assert len(table) == 2 * len(CATEGORIES)
