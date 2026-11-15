import random

import pytest

from jev_eval.pipeline.readout_table import readout_analysis, write_readout_outputs


def make_records(n=80, seed=3):
    """gemma_prob is calibrated in both languages; gemma_verb is overconfident, and more so in PT."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        for condition, accuracy in (("C1", 0.8), ("C2", 0.6)):
            for system, confidence in (("jev", accuracy), ("llm", 0.95), ("gemma_prob", accuracy),
                                       ("gemma_verb", 0.95)):
                rows.append({"example_id": f"e{i}", "dataset": "banking77", "condition": condition, "system": system,
                             "confidence": confidence, "correct": rng.random() < accuracy,
                             "predicted_label_name": "a", "error": None})
    return rows


@pytest.fixture(scope="module")
def result():
    return readout_analysis(make_records(), 10, n_bootstrap=150)


def test_cells_cover_every_system_and_condition(result):
    assert {(c["system"], c["condition"]) for c in result["cells"]} == {
        (s, c) for s in ("jev", "llm", "gemma_prob", "gemma_verb") for c in ("C1", "C2")}
    cell = next(c for c in result["cells"] if c["system"] == "gemma_verb" and c["condition"] == "C2")
    assert (cell["model"], cell["readout"], cell["n"]) == ("gemma", "verb", 80)
    assert cell["ece_ci95"][0] <= cell["ece"] <= cell["ece_ci95"][1]


def test_verbalized_is_worse_calibrated_than_probabilities(result):
    rows = {r["condition"]: r for r in result["readout_within_model"]}
    assert set(rows) == {"C1", "C2"} and rows["C2"]["model"] == "gemma"
    assert rows["C2"]["estimate"] == pytest.approx(rows["C2"]["ece_verb"] - rows["C2"]["ece_prob"])
    assert rows["C2"]["estimate"] > 0.2 and rows["C2"]["ci_excludes_zero"]
    assert rows["C1"]["n"] == 80


def test_readout_shift_contrast_compares_the_two_en_to_pt_changes(result):
    (row,) = result["readout_shift_within_model"]
    assert row["pt_condition"] == "C2"
    assert row["estimate"] == pytest.approx(row["shift_verb"] - row["shift_prob"])
    assert row["shift_verb"] > row["shift_prob"]  # overconfident verbalization suffers more from the drop


def test_jev_is_compared_with_every_other_system(result):
    systems = {(r["system"], r["condition"]) for r in result["jev_vs_system"]}
    assert systems == {(s, c) for s in ("llm", "gemma_prob", "gemma_verb") for c in ("C1", "C2")}
    vs_llm = next(r for r in result["jev_vs_system"] if r["system"] == "llm" and r["condition"] == "C2")
    assert vs_llm["estimate"] < 0 and vs_llm["ci_excludes_zero"]  # Jev (calibrated) has the lower ECE
    assert next(r for r in result["jev_vs_system"] if r["system"] == "gemma_prob")["readout"] == "prob"


def test_only_examples_valid_in_every_cell_are_paired():
    rows = [r for r in make_records(30) if not (r["example_id"] == "e0" and r["system"] == "gemma_prob"
                                                and r["condition"] == "C2")]
    rows.append({"example_id": "e1", "dataset": "banking77", "condition": "C1", "system": "gemma_verb",
                 "confidence": 0.0, "correct": False, "predicted_label_name": "", "error": "boom"})
    rows = [r for r in rows if not (r["example_id"] == "e1" and r["system"] == "gemma_verb"
                                    and r["condition"] == "C1" and r["error"] is None)]
    out = readout_analysis(rows, 10, n_bootstrap=20)
    assert next(r for r in out["readout_within_model"] if r["condition"] == "C1")["n"] == 29  # e1 failed in verb C1
    assert next(r for r in out["readout_within_model"] if r["condition"] == "C2")["n"] == 29  # e0 missing in prob C2
    assert out["readout_shift_within_model"][0]["n"] == 28  # both


def test_no_open_models_means_no_readout_contrasts():
    rows = [r for r in make_records(20) if r["system"] in ("jev", "llm")]
    out = readout_analysis(rows, 10, n_bootstrap=20)
    assert out["readout_within_model"] == [] and out["readout_shift_within_model"] == []
    assert len(out["jev_vs_system"]) == 2


def test_outputs_are_written(tmp_path, result):
    written = write_readout_outputs(result, tmp_path)
    assert [p.name for p in written] == ["readout.json", "readout_cells.csv", "readout_contrasts.csv"]
    contrasts = (tmp_path / "tables" / "readout_contrasts.csv").read_text(encoding="utf-8-sig").splitlines()
    assert contrasts[0].startswith("contrast;model;condition;a;b;n;a_minus_b")
    assert len(contrasts) == 1 + 2 + 1 + 6  # within (2 conditions) + shift (1) + jev vs 3 systems x 2 conditions
    assert any(line.startswith("verb_minus_prob_shift;gemma;C2 vs C1") for line in contrasts)
