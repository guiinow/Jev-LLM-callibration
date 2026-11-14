"""CSV tables for the EN->PT shift, the selection metrics and the ECE bin sensitivity.

They are built from the dict returned by `analyze`, so they exist right after `analyze`
(no need to run `plot`).
"""
from pathlib import Path
from typing import Any

from jev_eval.eval.figures import _write_csv


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _shift_rows(metrics: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for system, conditions in metrics.get("language_shift", {}).items():
        for condition, s in conditions.items():
            rows.append([
                system, condition, s["n"], _r(s["ece_en"]), _r(s["ece_pt"]), _r(s["delta_ece"]),
                _r(s["delta_ece_ci95"][0]), _r(s["delta_ece_ci95"][1]), s["delta_ece_ci_excludes_zero"],
                _r(s["accuracy_en"]), _r(s["accuracy_pt"]), _r(s["delta_accuracy"]),
                _r(s["delta_accuracy_ci95"][0]), _r(s["delta_accuracy_ci95"][1]),
                _r(s["prediction_change_rate"]), _r(s["mean_confidence_change"]),
                _r(s["mean_abs_confidence_change"]),
            ])
    return rows


def _selective_rows(metrics: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for condition, data in metrics["conditions"].items():
        for system, s in data.get("selective", {}).items():
            rows.append([
                condition, system, data["n_paired"], _r(s["aurc"]), _r(s["aurc_ci95"][0]), _r(s["aurc_ci95"][1]),
                _r(s["accuracy_at_coverage"]["0.5"]), _r(s["accuracy_at_0.5_ci95"][0]),
                _r(s["accuracy_at_0.5_ci95"][1]),
                _r(s["accuracy_at_coverage"]["0.8"]), _r(s["accuracy_at_0.8_ci95"][0]),
                _r(s["accuracy_at_0.8_ci95"][1]),
            ])
    return rows


def _bin_rows(metrics: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for condition, data in metrics["conditions"].items():
        sensitivity = data.get("bin_sensitivity")
        if not sensitivity:
            continue
        for bins, entry in sensitivity["by_bins"].items():
            rows.append([condition, int(bins), _r(entry["ece_jev"]), _r(entry["ece_llm"]), _r(entry["delta_ece"]),
                         _r(entry["ci95_low"]), _r(entry["ci95_high"]), entry["verdict"],
                         sensitivity["conclusion_stable"]])
    return rows


def write_analysis_tables(metrics: dict[str, Any], table_dir: Path) -> list[Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "language_shift.csv": (
            ["system", "pt_condition", "n", "ece_en", "ece_pt", "delta_ece", "delta_ece_ci_low",
             "delta_ece_ci_high", "delta_ece_ci_excludes_zero", "accuracy_en", "accuracy_pt", "delta_accuracy",
             "delta_accuracy_ci_low", "delta_accuracy_ci_high", "prediction_change_rate",
             "mean_confidence_change", "mean_abs_confidence_change"], _shift_rows(metrics)),
        "selective.csv": (
            ["condition", "system", "n_paired", "aurc", "aurc_ci_low", "aurc_ci_high", "accuracy_at_50",
             "accuracy_at_50_ci_low", "accuracy_at_50_ci_high", "accuracy_at_80", "accuracy_at_80_ci_low",
             "accuracy_at_80_ci_high"], _selective_rows(metrics)),
        "bin_sensitivity.csv": (
            ["condition", "n_bins", "ece_jev", "ece_llm", "delta_ece_jev_minus_llm", "ci_low", "ci_high",
             "verdict", "conclusion_stable"], _bin_rows(metrics)),
    }
    written = []
    for name, (header, rows) in tables.items():
        _write_csv(table_dir / name, header, rows)
        written.append(table_dir / name)
    return written
