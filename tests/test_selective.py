import random

import pytest

from jev_eval.eval.metrics import bootstrap_selective_ci, expected_risk_by_coverage, selective_summary


def test_risk_by_coverage_and_aurc():
    conf, ok = [1.0, 0.9, 0.8, 0.7], [True, True, False, True]
    assert expected_risk_by_coverage(conf, ok).tolist() == pytest.approx([0, 0, 1 / 3, 1 / 4])
    summary = selective_summary(conf, ok, coverages=(0.5, 1.0))
    assert summary["aurc"] == pytest.approx((1 / 3 + 1 / 4) / 4)
    assert summary["accuracy_at_coverage"] == pytest.approx({"0.5": 1.0, "1.0": 0.75})


def test_ties_do_not_depend_on_input_order():
    conf = [0.9, 0.9, 0.9, 0.5, 0.5]
    ok = [True, False, True, False, True]
    pairs = list(zip(conf, ok))
    reference = selective_summary(conf, ok)
    for seed in range(5):
        random.Random(seed).shuffle(pairs)
        shuffled = selective_summary([c for c, _ in pairs], [o for _, o in pairs])
        assert shuffled["aurc"] == pytest.approx(reference["aurc"])
        assert shuffled["risk_at_coverage"] == pytest.approx(reference["risk_at_coverage"])
    # one tie group with 1 error in 2: risk is 0.5 at k=1 and k=2
    assert expected_risk_by_coverage([1.0, 1.0], [True, False]).tolist() == [0.5, 0.5]


def test_coverage_rounds_up_without_float_noise():
    # 0.8 * 5 == 4.000000000000001 in floating point: must still mean the top 4 items.
    conf, ok = [0.9, 0.8, 0.7, 0.6, 0.5], [True, True, True, True, False]
    assert selective_summary(conf, ok, coverages=(0.8,))["accuracy_at_coverage"]["0.8"] == 1.0


def test_confident_correct_beats_confident_wrong():
    good = selective_summary([0.9, 0.8, 0.2, 0.1], [True, True, False, False])
    bad = selective_summary([0.9, 0.8, 0.2, 0.1], [False, False, True, True])
    assert good["aurc"] < bad["aurc"]


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        expected_risk_by_coverage([], [])


def test_selective_bootstrap_interval_contains_estimate():
    rng = random.Random(0)
    conf = [rng.random() for _ in range(80)]
    ok = [rng.random() < c for c in conf]
    ci = bootstrap_selective_ci(conf, ok, n_bootstrap=200)
    low, high = ci["aurc_ci95"]
    assert low <= selective_summary(conf, ok)["aurc"] <= high
    assert set(ci) == {"aurc_ci95", "accuracy_at_0.5_ci95", "accuracy_at_0.8_ci95"}
