import sys
from collections import Counter, defaultdict
from typing import Any

from jev_eval.clients.jev_client import chosen_probability
from jev_eval.eval.metrics import (bootstrap_selective_ci, bootstrap_system_ci, expected_calibration_error,
                                   paired_bootstrap_ece, save_metrics, selective_summary)
from jev_eval.pipeline.language_shift import language_shift

HIGH_CONFIDENCE = 0.9
SENSITIVITY_BINS = (10, 20, 30)  # extra ECE bin counts, to check the H1 conclusion does not hinge on 15

SYSTEMS = ("jev", "llm")

# Records written before the conditions refactor have `language` instead of `condition`.
# The preliminary PT-BR setup (PT message, PT options, EN instructions) is none of C1-C3.
LEGACY_CONDITIONS = {"en": "C1", "pt_br": "PT-preliminar"}


def with_conditions(records: list[dict[str, Any]], dataset: str = "banking77") -> list[dict[str, Any]]:
    """Give pre-refactor records (`language`, no `condition`/`dataset`) the current keys.

    Only for re-analysing saved runs (e.g. outputs/full_ministral); returns new dicts."""
    return [row if "condition" in row else
            {**row, "dataset": row.get("dataset", dataset), "condition": LEGACY_CONDITIONS[row["language"]]}
            for row in records]


def with_chosen_probability(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Proposal 4.2 for records written before the fix (no `confidence_sdk` key).

    Those stored Jev's SDK `confidence`; move it to `confidence_sdk` and use
    probabilities[choice] from the saved raw response. Returns new dicts; inputs and files
    are left untouched. Rows whose raw response lacks probabilities keep the SDK value
    and are flagged in `confidence_source`.
    """
    out, missing = [], 0
    for row in records:
        if row.get("system") == "jev" and row.get("error") is None and "confidence_sdk" not in row:
            row = {**row, "confidence_sdk": row["confidence"]}
            try:
                raw = row.get("raw_response") or {}
                row["confidence"] = chosen_probability(raw.get("probabilities") if isinstance(raw, dict) else None,
                                                       row.get("predicted_label_name"))
                row["confidence_source"] = "probabilities[choice] (derived from raw)"
            except ValueError:
                row["confidence_source"] = "sdk_confidence (probabilities missing)"
                missing += 1
        out.append(row)
    if missing:
        print(f"WARNING: {missing} legacy Jev records have no probabilities[choice] in raw_response; "
              "they still use the SDK `confidence` field.", file=sys.stderr)
    return out


def _confidence_source(row: dict[str, Any]) -> str:
    return row.get("confidence_source", "probabilities[choice]")


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


def _selective(rows: list[dict[str, Any]], n_bootstrap: int) -> dict[str, Any]:
    """AURC and accuracy at 50%/80% coverage when only the most confident answers are accepted."""
    confidences, correct = [r["confidence"] for r in rows], [bool(r["correct"]) for r in rows]
    return {**selective_summary(confidences, correct),
            **bootstrap_selective_ci(confidences, correct, n_bootstrap=n_bootstrap)}


def _verdict(interval_low: float, interval_high: float) -> str:
    if interval_high < 0:
        return "jev_lower"
    if interval_low > 0:
        return "llm_lower"
    return "inconclusive"


def _bin_sensitivity(paired: list[dict[str, Any]], n_bins: int, n_bootstrap: int) -> dict[str, Any]:
    """delta ECE (Jev - LLM) with other bin counts; `conclusion_stable` if the verdict never changes."""
    by_bins: dict[str, Any] = {}
    for bins in sorted({n_bins, *SENSITIVITY_BINS}):
        result = paired_bootstrap_ece(paired, bins, n_bootstrap)
        by_bins[str(bins)] = {
            "ece_jev": expected_calibration_error([r["jev"] for r in paired], [r["correct_jev"] for r in paired], bins),
            "ece_llm": expected_calibration_error([r["llm"] for r in paired], [r["correct_llm"] for r in paired], bins),
            "delta_ece": result["delta_ece"], "ci95_low": result["ci95_low"], "ci95_high": result["ci95_high"],
            "verdict": _verdict(result["ci95_low"], result["ci95_high"]),
        }
    return {"by_bins": by_bins,
            "conclusion_stable": len({entry["verdict"] for entry in by_bins.values()}) == 1}


def analyze(records: list[dict[str, Any]], n_bins: int, output_path, n_bootstrap: int = 5000) -> dict[str, Any]:
    """Per-condition metrics for one dataset.

    The headline numbers (`paired`, `paired_bootstrap`) use only examples where BOTH
    systems returned a valid answer, so ECE_jev and ECE_llm are computed on the same
    items. `all_valid` keeps every valid answer per system, and `errors` counts the
    failed calls (parse failures, API errors) that were excluded. `selective` and
    `bin_sensitivity` use the same paired examples; `language_shift` compares each PT condition
    with C1 inside each system.
    """
    records = with_chosen_probability(records)
    by_condition: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    errors: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row["error"] is None:
            by_condition[row["condition"]][row["system"]][row["example_id"]] = row
        else:
            errors[row["condition"]][row["system"]].append(row["example_id"])

    metrics: dict[str, Any] = {
        "n_bins": n_bins, "high_confidence_threshold": HIGH_CONFIDENCE,
        "jev_confidence_source": dict(Counter(_confidence_source(r) for r in records
                                              if r.get("system") == "jev" and r.get("error") is None)),
        "conditions": {},
    }
    for condition in sorted(set(by_condition) | set(errors)):
        systems = by_condition[condition]
        paired_ids = sorted(set.intersection(*(set(systems.get(s, {})) for s in SYSTEMS)))
        condition_metrics: dict[str, Any] = {
            "errors": {s: {"n": len(errors[condition][s]), "example_ids": errors[condition][s]} for s in SYSTEMS},
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
            condition_metrics["paired_bootstrap"] = paired_bootstrap_ece(paired, n_bins, n_bootstrap)
            condition_metrics["selective"] = {s: _selective([systems[s][i] for i in paired_ids], n_bootstrap)
                                              for s in SYSTEMS}
            condition_metrics["bin_sensitivity"] = _bin_sensitivity(paired, n_bins, n_bootstrap)
            sdk = [systems["jev"][i] for i in paired_ids if systems["jev"][i].get("confidence_sdk") is not None]
            # Check only: ECE if Jev's SDK `confidence` statistic were used instead (not the H1 metric).
            condition_metrics["jev_confidence_sdk_check"] = {
                "n": len(sdk),
                "ece": expected_calibration_error([r["confidence_sdk"] for r in sdk],
                                                  [bool(r["correct"]) for r in sdk], n_bins) if sdk else None,
            }
        metrics["conditions"][condition] = condition_metrics
    metrics["language_shift"] = language_shift(records, n_bins, n_bootstrap)
    save_metrics(output_path, metrics)
    return metrics
