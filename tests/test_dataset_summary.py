"""Equal-weight summary across datasets: per-dataset estimates, their mean, and an interval from
averaging the per-dataset bootstrap draws (predictions are never pooled)."""
import pytest

from jev_eval.pipeline.dataset_summary import dataset_summary, summary_rows


def rows(dataset, condition, system, conf, correct, n=20):
    return [{"dataset": dataset, "condition": condition, "system": system, "example_id": f"{dataset}-{i}",
             "confidence": conf, "correct": i < correct, "error": None} for i in range(n)]


def records():
    out = []
    for dataset, (jev_hits, llm_hits) in {"a": (18, 10), "b": (10, 10)}.items():
        for condition in ("C1", "C3"):
            out += rows(dataset, condition, "jev", 0.9, jev_hits) + rows(dataset, condition, "llm", 1.0, llm_hits)
    out.append({"dataset": "a", "condition": "C3", "system": "llm", "example_id": "a-99", "error": "parse"})
    return out


def test_mean_of_datasets_not_pooled():
    result = dataset_summary(records(), n_bins=10, n_bootstrap=200)
    # ECE with one bin per confidence value: |acc - conf|. Jev: a 0.0, b 0.4 -> mean 0.2.
    jev = result["ece"]["C3"]["jev"]
    assert jev["per_dataset"] == {"a": pytest.approx(0.0), "b": pytest.approx(0.4)} and jev["mean"] == pytest.approx(0.2)
    # Jev - LLM: a 0.0 - 0.5 = -0.5, b 0.4 - 0.5 = -0.1 -> mean -0.3, interval around it.
    delta = result["jev_minus"]["C3"]["llm"]
    assert delta["mean"] == pytest.approx(-0.3) and delta["ci95"][0] <= -0.3 <= delta["ci95"][1]
    # Same accuracy and confidence in EN and PT -> no change.
    assert result["en_to_pt"]["jev"]["C3-C1"]["mean"] == pytest.approx(0.0)


def test_summary_table_has_one_column_per_dataset():
    header, table = summary_rows(dataset_summary(records(), n_bins=10, n_bootstrap=50))
    assert header[-2:] == ["a", "b"]
    assert {r[0] for r in table} == {"ece", "ece_jev_minus_system", "ece_pt_minus_en"}
