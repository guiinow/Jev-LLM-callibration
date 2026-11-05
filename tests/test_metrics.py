from jev_eval.eval.metrics import expected_calibration_error


def test_ece_perfect_calibration():
    assert expected_calibration_error([0.0, 1.0], [False, True], 2) == 0


def test_ece_miscalibrated_one_bin():
    assert abs(expected_calibration_error([0.9] * 4, [True, True, False, False], 1) - 0.4) < 1e-9

