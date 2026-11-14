import random

import pytest

from jev_eval.pipeline.benchmark_check import PUBLISHED, benchmark_check, format_report


def jev_rows(n=600, accuracy=0.797, seed=0, condition="C1", dataset="banking77"):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        confidence = round(rng.uniform(0.3, 1.0), 2)
        rows.append({"example_id": f"e{i}", "dataset": dataset, "condition": condition, "system": "jev",
                     "confidence": confidence, "correct": rng.random() < min(1.0, accuracy + (confidence - 0.65) / 2),
                     "error": None})
    return rows


def test_check_passes_when_published_numbers_are_inside_our_intervals():
    result = benchmark_check(jev_rows(), 15, n_bootstrap=100)
    assert result["n"] == 600
    assert result["metrics"]["accuracy"]["published"] == PUBLISHED["accuracy"]
    assert result["metrics"]["accuracy"]["published_in_ci"]
    assert set(result["metrics"]) == {"accuracy", "ece", "accuracy_at_0.8", "accuracy_at_0.5", "aurc"}


def test_check_flags_a_much_worse_run_and_names_the_metric():
    result = benchmark_check(jev_rows(accuracy=0.4), 15, n_bootstrap=100)
    assert not result["all_within_ci"] and not result["metrics"]["accuracy"]["published_in_ci"]
    assert "Difere em" in format_report(result) and "Acurácia" in format_report(result)


def test_only_valid_jev_c1_banking77_rows_count():
    rows = jev_rows(50) + jev_rows(50, condition="C2") + jev_rows(50, dataset="other")
    rows.append({**jev_rows(1)[0], "error": "boom"})
    assert benchmark_check(rows, 15, n_bootstrap=20)["n"] == 50


def test_missing_c1_is_an_error_and_other_bins_warn():
    with pytest.raises(ValueError, match="C1"):
        benchmark_check(jev_rows(20, condition="C2"), 15, n_bootstrap=20)
    result = benchmark_check(jev_rows(30), 6, n_bootstrap=20)
    assert "ece_bins desta execução é 6" in result["warnings"][0] and "Aviso" in format_report(result)


def test_report_lists_every_metric():
    report = format_report(benchmark_check(jev_rows(), 15, n_bootstrap=50))
    for label in ("Acurácia", "ECE", "AURC", "50% mais confiantes", "80% mais confiantes"):
        assert label in report
