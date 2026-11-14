import pytest

from jev_eval.pipeline.language_shift import language_shift


def row(example_id, condition, system, confidence, correct, predicted="a", error=None):
    return {"example_id": example_id, "condition": condition, "system": system, "confidence": confidence,
            "correct": correct, "predicted_label_name": predicted, "error": error}


def records():
    rows = []
    for i in range(40):
        rows.append(row(f"e{i}", "C1", "jev", 0.9, i % 10 != 0))            # 90% accurate, 0.9 confident
        rows.append(row(f"e{i}", "C2", "jev", 0.9, i % 2 == 0, "b" if i % 2 else "a"))  # 50% accurate
        rows.append(row(f"e{i}", "C1", "llm", 0.8, True))
    return rows


def test_shift_is_pt_minus_en_with_paired_interval():
    shift = language_shift(records(), n_bins=5, n_bootstrap=200)["jev"]["C2"]
    assert shift["n"] == 40
    assert shift["ece_en"] == pytest.approx(0.0)  # 0.9 confidence at 0.9 accuracy
    assert shift["ece_pt"] == pytest.approx(0.4)
    assert shift["delta_ece"] == pytest.approx(0.4)
    assert shift["delta_accuracy"] == pytest.approx(0.5 - 0.9)
    assert shift["prediction_change_rate"] == pytest.approx(0.5)
    assert shift["mean_confidence_change"] == pytest.approx(0.0)
    low, high = shift["delta_ece_ci95"]
    assert low <= shift["delta_ece"] <= high and shift["delta_ece_ci_excludes_zero"]


def test_only_systems_and_conditions_with_both_sides_appear():
    shifts = language_shift(records(), n_bins=5, n_bootstrap=20)
    assert set(shifts) == {"jev"} and set(shifts["jev"]) == {"C2"}  # llm has no PT answers; no C3 run


def test_only_examples_valid_in_both_conditions_are_used():
    def is_jev(r, example_id, condition):
        return r["example_id"] == example_id and r["condition"] == condition and r["system"] == "jev"
    rows = [r for r in records() if not is_jev(r, "e0", "C2") and not is_jev(r, "e1", "C1")]
    rows.append(row("e0", "C2", "jev", 0.0, False, error="boom"))  # e0 failed in PT, e1 has no EN answer
    assert language_shift(rows, n_bins=5, n_bootstrap=20)["jev"]["C2"]["n"] == 38


def test_identical_answers_give_no_shift():
    rows = [row(f"e{i}", c, "jev", 0.7, i % 3 == 0) for i in range(30) for c in ("C1", "C3")]
    shift = language_shift(rows, n_bins=5, n_bootstrap=50)["jev"]["C3"]
    assert shift["delta_ece"] == 0 and shift["delta_accuracy"] == 0
    assert shift["prediction_change_rate"] == 0 and not shift["delta_ece_ci_excludes_zero"]
