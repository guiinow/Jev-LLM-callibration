import json
import random

import pytest

from jev_eval.pipeline.analyze import _verdict, analyze


def fake_records(n=60, seed=1):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        for condition, accuracy in (("C1", 0.9), ("C2", 0.6)):
            for system, confidence in (("jev", 0.85), ("llm", 0.99)):
                ok = rng.random() < accuracy
                rows.append({"example_id": f"e{i}", "dataset": "banking77", "condition": condition,
                             "system": system, "confidence": confidence, "correct": ok,
                             "predicted_label_name": "a" if ok else "b", "error": None})
    return rows


def test_analyze_reports_selection_bin_sensitivity_and_language_shift(tmp_path):
    out = tmp_path / "m.json"
    metrics = analyze(fake_records(), 15, out, n_bootstrap=50)
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(json.dumps(metrics))

    c2 = metrics["conditions"]["C2"]
    for system in ("jev", "llm"):
        selective = c2["selective"][system]
        assert 0 <= selective["aurc"] <= 1
        assert set(selective["accuracy_at_coverage"]) == {"0.5", "0.8"}
        assert selective["aurc_ci95"][0] <= selective["aurc"] <= selective["aurc_ci95"][1]
    # constant confidence carries no ranking information: accuracy at any coverage is the overall accuracy
    assert c2["selective"]["jev"]["accuracy_at_coverage"]["0.5"] == pytest.approx(
        c2["paired"]["jev"]["accuracy"])

    assert set(c2["bin_sensitivity"]["by_bins"]) == {"10", "15", "20", "30"}
    assert isinstance(c2["bin_sensitivity"]["conclusion_stable"], bool)

    shift = metrics["language_shift"]["jev"]["C2"]
    assert shift["n"] == 60 and shift["delta_accuracy"] < 0  # PT is less accurate in this fake data
    assert "C3" not in metrics["language_shift"]["jev"]


def test_bin_sensitivity_includes_the_configured_bins(tmp_path):
    metrics = analyze(fake_records(20), 6, tmp_path / "m.json", n_bootstrap=20)
    assert set(metrics["conditions"]["C1"]["bin_sensitivity"]["by_bins"]) == {"6", "10", "20", "30"}


def test_verdict_labels():
    assert _verdict(-0.2, -0.01) == "jev_lower"
    assert _verdict(0.01, 0.2) == "llm_lower"
    assert _verdict(-0.05, 0.05) == "inconclusive"
