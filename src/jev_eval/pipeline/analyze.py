from collections import defaultdict
from typing import Any

from jev_eval.eval.metrics import expected_calibration_error, paired_bootstrap_ece, save_metrics


def analyze(records: list[dict[str, Any]], n_bins: int, output_path) -> dict[str, Any]:
    by_language: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row["error"] is None:
            by_language[row["language"]][row["system"]].append(row)
    metrics: dict[str, Any] = {"n_bins": n_bins, "languages": {}}
    for language, systems in by_language.items():
        language_metrics = {}
        for system, rows in systems.items():
            language_metrics[system] = {
                "ece": expected_calibration_error(
                    [row["confidence"] for row in rows], [row["correct"] for row in rows], n_bins
                ),
                "accuracy": sum(row["correct"] for row in rows) / len(rows),
                "n": len(rows),
            }
        paired = []
        by_id = defaultdict(dict)
        for system, rows in systems.items():
            for row in rows:
                by_id[row["example_id"]][system] = row
        for values in by_id.values():
            if {"jev", "llm"} <= values.keys():
                paired.append({
                    "jev": values["jev"]["confidence"], "llm": values["llm"]["confidence"],
                    "correct_jev": values["jev"]["correct"], "correct_llm": values["llm"]["correct"],
                })
        if paired:
            language_metrics["paired_bootstrap"] = paired_bootstrap_ece(paired, n_bins)
        metrics["languages"][language] = language_metrics
    save_metrics(output_path, metrics)
    return metrics
