from jev_eval.eval.metrics import expected_calibration_error


def test_ece_perfect_calibration():
    assert expected_calibration_error([0.0, 1.0], [False, True], 2) == 0


def test_ece_miscalibrated_one_bin():
    assert abs(expected_calibration_error([0.9] * 4, [True, True, False, False], 1) - 0.4) < 1e-9



def test_robustness_metrics():
    import pytest
    from jev_eval.eval.metrics import auroc, brier_top1, equal_mass_ece
    conf, ok = [0.9, 0.8, 0.6, 0.6], [True, True, False, True]
    assert auroc(conf, ok) == pytest.approx((1 + 1 + 0.5) / 3)  # correct items vs the one wrong (0.6)
    assert auroc([0.5, 0.7], [True, True]) is None
    assert brier_top1([1.0, 0.5], [True, False]) == pytest.approx(0.125)
    # 2 equal-mass bins: {0.6 wrong, 0.6 right} -> |0.5-0.6|, {0.8, 0.9 right} -> |1-0.85|
    assert equal_mass_ece(conf, ok, 2) == pytest.approx(0.5 * 0.1 + 0.5 * 0.15)
