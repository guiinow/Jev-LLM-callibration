import random

from jev_eval.eval.figures import confusion_pairs, make_report_outputs, paired_by_condition, selective_curve


def fake_records(n=40, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        for lang in ("C1", "C2"):
            for system in ("jev", "llm"):
                conf = rng.random() if system == "jev" else rng.choice([0.95, 1.0])
                ok = rng.random() < 0.7
                rows.append({"example_id": f"e{i}", "condition": lang, "system": system, "true_label_name": "a",
                             "predicted_label_name": "a" if ok else rng.choice(["b", "c"]), "confidence": conf,
                             "correct": ok, "error": None, "model_id": system})
    rows.append({"example_id": "e0", "condition": "C1", "system": "llm", "error": "ValueError: x"})
    return rows


def test_paired_excludes_examples_missing_a_system():
    rows = [r for r in fake_records() if not (r["example_id"] == "e1" and r["system"] == "llm")]
    paired = paired_by_condition(rows)
    assert len(paired["C1"]["jev"]) == len(paired["C1"]["llm"]) == 39


def test_selective_curve_groups_ties():
    rows = [{"confidence": c, "correct": ok} for c, ok in [(1.0, True), (1.0, False), (0.9, True), (0.5, False)]]
    cov, acc, thr = selective_curve(rows)
    assert thr == [1.0, 0.9, 0.5] and cov == [0.5, 0.75, 1.0] and acc[0] == 0.5


def test_confusion_pairs_counts_both_systems():
    paired = paired_by_condition(fake_records())
    total = sum(sum(c.values()) for _, _, c, _ in confusion_pairs(paired, "C2"))
    assert total == sum(not r["correct"] for s in ("jev", "llm") for r in paired["C2"][s])


def test_make_report_outputs_writes_all_files(tmp_path):
    written = make_report_outputs(tmp_path, [r for r in fake_records() if r.get("error") is None] +
                                  [fake_records()[-1]], 10, n_bootstrap=200)
    names = {p.name for p in written}
    assert {"reliability.png", "ece_accuracy_ci.png", "delta_ece_bootstrap.png", "risk_coverage.png",
            "confidence_correct_vs_wrong.png", "top_confusions.png", "summary.csv", "reliability_bins.csv",
            "high_confidence_errors.csv", "confusion_pairs.csv", "language_flips.csv"} <= names
