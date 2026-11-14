from collections import defaultdict
from typing import Any

from jev_eval.eval.metrics import (bootstrap_system_ci, expected_calibration_error, paired_bootstrap_ece,
                                   save_metrics)

HIGH_CONFIDENCE = 0.9

SYSTEMS = ("jev", "llm")


def _summary(rows: list[dict[str, Any]], n_bins: int, with_ci: bool = False) -> dict[str, Any]:
    if not rows:
        return {"ece": None, "accuracy": None, "n": 0}
    confidences = [r["confidence"] for r in rows]
    correct = [bool(r["correct"]) for r in rows]
    accuracy = sum(correct) / len(rows)
    mean_confidence = sum(confidences) / len(rows)
    summary = {
        "ece": expected_calibration_error(confidences, correct, n_bins),
        "accuracy": accuracy,
        "n": len(rows),
        "mean_confidence": mean_confidence,
        "overconfidence": mean_confidence - accuracy,
        "high_confidence_errors": sum(1 for c, ok in zip(confidences, correct) if not ok and c >= HIGH_CONFIDENCE),
        "distinct_confidence_values": len({round(c, 4) for c in confidences}),
    }
    if with_ci:
        summary.update(bootstrap_system_ci(confidences, correct, n_bins))
    return summary


def analyze(records: list[dict[str, Any]], n_bins: int, output_path) -> dict[str, Any]:
    """Per-language metrics.

    The headline numbers (`paired`, `paired_bootstrap`) use only examples where BOTH
    systems returned a valid answer, so ECE_jev and ECE_llm are computed on the same
    items. `all_valid` keeps every valid answer per system, and `errors` counts the
    failed calls (parse failures, API errors) that were excluded.
    """
    by_language: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    errors: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row["error"] is None:
            by_language[row["language"]][row["system"]][row["example_id"]] = row
        else:
            errors[row["language"]][row["system"]].append(row["example_id"])

    metrics: dict[str, Any] = {"n_bins": n_bins, "high_confidence_threshold": HIGH_CONFIDENCE, "languages": {}}
    for language in sorted(set(by_language) | set(errors)):
        systems = by_language[language]
        paired_ids = sorted(set.intersection(*(set(systems.get(s, {})) for s in SYSTEMS)))
        language_metrics: dict[str, Any] = {
            "errors": {s: {"n": len(errors[language][s]), "example_ids": errors[language][s]} for s in SYSTEMS},
            "all_valid": {s: _summary(list(systems.get(s, {}).values()), n_bins) for s in SYSTEMS},
            "paired": {s: _summary([systems[s][i] for i in paired_ids], n_bins, with_ci=bool(paired_ids))
                       for s in SYSTEMS},
            "n_paired": len(paired_ids),
        }
        if paired_ids:
            paired = [{
                "jev": systems["jev"][i]["confidence"], "llm": systems["llm"][i]["confidence"],
                "correct_jev": systems["jev"][i]["correct"], "correct_llm": systems["llm"][i]["correct"],
            } for i in paired_ids]
            language_metrics["paired_bootstrap"] = paired_bootstrap_ece(paired, n_bins)
        metrics["languages"][language] = language_metrics
    save_metrics(output_path, metrics)
    return metrics
