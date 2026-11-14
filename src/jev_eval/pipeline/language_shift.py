"""EN -> PT change within each system (Q2): how much do ECE and accuracy move when only the
language changes?

For every system, each PT condition is compared with the English baseline C1 on the examples
the system answered validly in BOTH conditions. The bootstrap resamples examples, so the EN
and PT answers of the same example always stay together.
"""
from typing import Any

import numpy as np

from jev_eval.eval.metrics import expected_calibration_error

BASELINE = "C1"
PT_CONDITIONS = ("C2", "C3")  # C2 (PT content, EN instructions) is the primary contrast


def _interval(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def shift_between(en: list[dict[str, Any]], pt: list[dict[str, Any]], n_bins: int,
                  n_bootstrap: int = 5000, seed: int = 42) -> dict[str, Any]:
    """`en` and `pt` are the same examples, in the same order, in the two conditions."""
    conf_en = np.asarray([r["confidence"] for r in en], dtype=float)
    conf_pt = np.asarray([r["confidence"] for r in pt], dtype=float)
    ok_en = np.asarray([bool(r["correct"]) for r in en])
    ok_pt = np.asarray([bool(r["correct"]) for r in pt])
    changed = np.asarray([a["predicted_label_name"] != b["predicted_label_name"] for a, b in zip(en, pt)])
    n = len(en)

    def ece(conf: np.ndarray, ok: np.ndarray) -> float:
        return expected_calibration_error(conf.tolist(), ok.tolist(), n_bins)

    delta_ece = ece(conf_pt, ok_pt) - ece(conf_en, ok_en)
    rng = np.random.default_rng(seed)
    ece_samples, acc_samples = np.empty(n_bootstrap), np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        pick = rng.integers(0, n, n)
        ece_samples[index] = ece(conf_pt[pick], ok_pt[pick]) - ece(conf_en[pick], ok_en[pick])
        acc_samples[index] = ok_pt[pick].mean() - ok_en[pick].mean()
    delta_acc = ok_pt.mean() - ok_en.mean()
    ece_ci, acc_ci = _interval(ece_samples), _interval(acc_samples)
    return {
        "n": n,
        "ece_en": ece(conf_en, ok_en), "ece_pt": ece(conf_pt, ok_pt),
        "delta_ece": float(delta_ece), "delta_ece_ci95": ece_ci,
        "delta_ece_ci_excludes_zero": not ece_ci[0] <= 0 <= ece_ci[1],
        "accuracy_en": float(ok_en.mean()), "accuracy_pt": float(ok_pt.mean()),
        "delta_accuracy": float(delta_acc), "delta_accuracy_ci95": acc_ci,
        "delta_accuracy_ci_excludes_zero": not acc_ci[0] <= 0 <= acc_ci[1],
        "prediction_change_rate": float(changed.mean()),
        "mean_confidence_change": float((conf_pt - conf_en).mean()),
        "mean_abs_confidence_change": float(np.abs(conf_pt - conf_en).mean()),
        "n_bootstrap": n_bootstrap, "seed": seed,
    }


def language_shift(records: list[dict[str, Any]], n_bins: int, n_bootstrap: int = 5000,
                   seed: int = 42) -> dict[str, Any]:
    """{system: {pt_condition: shift}} for every system with answers in C1 and a PT condition."""
    valid: dict[tuple[str, str], dict[str, dict]] = {}
    for row in records:
        if row.get("error") is None:
            valid.setdefault((row["system"], row["condition"]), {})[row["example_id"]] = row
    shifts: dict[str, dict[str, Any]] = {}
    for system in sorted({system for system, _ in valid}):
        for condition in PT_CONDITIONS:
            en, pt = valid.get((system, BASELINE), {}), valid.get((system, condition), {})
            ids = sorted(set(en) & set(pt))
            if ids:
                shifts.setdefault(system, {})[condition] = shift_between(
                    [en[i] for i in ids], [pt[i] for i in ids], n_bins, n_bootstrap, seed)
    return shifts
