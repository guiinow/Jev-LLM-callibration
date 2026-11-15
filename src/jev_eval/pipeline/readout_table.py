"""The central comparison: model x how confidence is read x language.

Each open model is measured two ways from the same weights ("<model>_prob": option
probabilities; "<model>_verb": confidence written by the model). That separates "which model"
from "how the confidence is read", which a plain Jev vs LLM comparison cannot do. Three paired
contrasts, all with a bootstrap over examples (an example's answers in every cell stay together):

- `readout_within_model`: ECE(verb) - ECE(prob) of one model, per condition.
- `readout_shift_within_model`: does the EN->PT change of ECE depend on the readout?
  (delta ECE of verb) - (delta ECE of prob), for each PT condition against C1.
- `jev_vs_system`: ECE(jev) - ECE(other system), per condition, including each open model read
  by probabilities (same readout as Jev) and by words.
"""
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from jev_eval.eval.metrics import bootstrap_system_ci, expected_calibration_error, save_metrics
from jev_eval.pipeline.language_shift import BASELINE, PT_CONDITIONS
from jev_eval.systems import system_parts

Arrays = dict[str, tuple[np.ndarray, np.ndarray]]  # cell name -> (confidence, correct)


def _valid(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, dict]]:
    valid: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in records:
        if row.get("error") is None:
            valid[(row["system"], row["condition"])][row["example_id"]] = row
    return valid


def _aligned(valid: dict, cells: dict[str, tuple[str, str]]) -> Arrays | None:
    """Arrays for the named (system, condition) cells over the examples valid in ALL of them."""
    maps = {name: valid.get(key, {}) for name, key in cells.items()}
    ids = sorted(set.intersection(*(set(m) for m in maps.values())))
    if not ids:
        return None
    return {name: (np.asarray([m[i]["confidence"] for i in ids], dtype=float),
                   np.asarray([bool(m[i]["correct"]) for i in ids]))
            for name, m in maps.items()}


def _ece(cell: tuple[np.ndarray, np.ndarray], n_bins: int) -> float:
    return expected_calibration_error(cell[0].tolist(), cell[1].tolist(), n_bins)


def _bootstrap(arrays: Arrays, stat: Callable[[Arrays], float], n_bootstrap: int, seed: int) -> dict[str, Any]:
    n = len(next(iter(arrays.values()))[0])
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        pick = rng.integers(0, n, n)
        draws[index] = stat({name: (conf[pick], ok[pick]) for name, (conf, ok) in arrays.items()})
    low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {"n": n, "estimate": float(stat(arrays)), "ci95": [low, high], "ci_excludes_zero": not low <= 0 <= high,
            "n_bootstrap": n_bootstrap, "seed": seed}


def _cells(valid: dict, n_bins: int, n_bootstrap: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for (system, condition), by_id in sorted(valid.items()):
        confidences = [r["confidence"] for r in by_id.values()]
        correct = [bool(r["correct"]) for r in by_id.values()]
        model, readout = system_parts(system)
        ci = bootstrap_system_ci(confidences, correct, n_bins, n_bootstrap, seed)
        rows.append({"model": model, "readout": readout, "system": system, "condition": condition,
                     "n": len(by_id), "accuracy": sum(correct) / len(correct),
                     "ece": expected_calibration_error(confidences, correct, n_bins), "ece_ci95": ci["ece_ci95"],
                     "mean_confidence": float(np.mean(confidences))})
    return rows


def readout_analysis(records: list[dict[str, Any]], n_bins: int, n_bootstrap: int = 5000,
                     seed: int = 42) -> dict[str, Any]:
    valid = _valid(records)
    systems = sorted({system for system, _ in valid})
    conditions = sorted({condition for _, condition in valid})
    by_model: dict[str, dict[str, str]] = defaultdict(dict)
    for system in systems:
        model, readout = system_parts(system)
        by_model[model][readout] = system

    within, shift_within, jev_vs = [], [], []
    for model, readouts in sorted(by_model.items()):
        if "prob" in readouts and "verb" in readouts:
            prob, verb = readouts["prob"], readouts["verb"]
            for condition in conditions:
                arrays = _aligned(valid, {"prob": (prob, condition), "verb": (verb, condition)})
                if arrays:
                    within.append({"model": model, "condition": condition, **_bootstrap(
                        arrays, lambda a: _ece(a["verb"], n_bins) - _ece(a["prob"], n_bins), n_bootstrap, seed),
                        "ece_prob": _ece(arrays["prob"], n_bins), "ece_verb": _ece(arrays["verb"], n_bins)})
            for pt in PT_CONDITIONS:
                arrays = _aligned(valid, {"prob_en": (prob, BASELINE), "prob_pt": (prob, pt),
                                          "verb_en": (verb, BASELINE), "verb_pt": (verb, pt)})
                if arrays:
                    def stat(a: Arrays) -> float:
                        return ((_ece(a["verb_pt"], n_bins) - _ece(a["verb_en"], n_bins))
                                - (_ece(a["prob_pt"], n_bins) - _ece(a["prob_en"], n_bins)))
                    shift_within.append({"model": model, "pt_condition": pt, **_bootstrap(arrays, stat, n_bootstrap, seed),
                                         "shift_prob": _ece(arrays["prob_pt"], n_bins) - _ece(arrays["prob_en"], n_bins),
                                         "shift_verb": _ece(arrays["verb_pt"], n_bins) - _ece(arrays["verb_en"], n_bins)})
    if "jev" in systems:
        for other in (s for s in systems if s != "jev"):
            for condition in conditions:
                arrays = _aligned(valid, {"jev": ("jev", condition), "other": (other, condition)})
                if arrays:
                    model, readout = system_parts(other)
                    jev_vs.append({"system": other, "model": model, "readout": readout, "condition": condition,
                                   **_bootstrap(arrays, lambda a: _ece(a["jev"], n_bins) - _ece(a["other"], n_bins),
                                                n_bootstrap, seed)})
    return {"n_bins": n_bins, "cells": _cells(valid, n_bins, n_bootstrap, seed), "readout_within_model": within,
            "readout_shift_within_model": shift_within, "jev_vs_system": jev_vs}


def _r(value: float, digits: int = 4) -> float:
    return round(value, digits)


def write_readout_outputs(result: dict[str, Any], out_dir: Path) -> list[Path]:
    from jev_eval.eval.figures import _write_csv
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(out_dir / "readout.json", result)
    _write_csv(table_dir / "readout_cells.csv",
               ["model", "readout", "system", "condition", "n", "accuracy", "ece", "ece_ci_low", "ece_ci_high",
                "mean_confidence"],
               [[c["model"], c["readout"], c["system"], c["condition"], c["n"], _r(c["accuracy"]), _r(c["ece"]),
                 _r(c["ece_ci95"][0]), _r(c["ece_ci95"][1]), _r(c["mean_confidence"])] for c in result["cells"]])
    contrasts = (
        [["verb_minus_prob_ece", r["model"], r["condition"], "verb", "prob", r] for r in result["readout_within_model"]]
        + [["verb_minus_prob_shift", r["model"], r["pt_condition"] + " vs " + BASELINE, "verb", "prob", r]
           for r in result["readout_shift_within_model"]]
        + [["jev_minus_system_ece", r["system"], r["condition"], "jev", r["system"], r]
           for r in result["jev_vs_system"]])
    _write_csv(table_dir / "readout_contrasts.csv",
               ["contrast", "model", "condition", "a", "b", "n", "a_minus_b", "ci_low", "ci_high", "ci_excludes_zero"],
               [[name, model, condition, a, b, r["n"], _r(r["estimate"]), _r(r["ci95"][0]), _r(r["ci95"][1]),
                 r["ci_excludes_zero"]] for name, model, condition, a, b, r in contrasts])
    return [out_dir / "readout.json", table_dir / "readout_cells.csv", table_dir / "readout_contrasts.csv"]
