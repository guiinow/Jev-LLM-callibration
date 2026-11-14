import json
from pathlib import Path
from typing import Any

import numpy as np


def expected_calibration_error(confidences: list[float], correct: list[bool], n_bins: int) -> float:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if not confidences:
        return 0.0
    confidence = np.asarray(confidences, dtype=float)
    accuracy = np.asarray(correct, dtype=float)
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence values must be between 0 and 1")
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for index in range(n_bins):
        mask = ((confidence >= edges[index]) & (confidence <= edges[index + 1])
                if index == n_bins - 1
                else (confidence >= edges[index]) & (confidence < edges[index + 1]))
        if mask.any():
            total += mask.mean() * abs(accuracy[mask].mean() - confidence[mask].mean())
    return float(total)


def equal_mass_ece(confidences: list[float], correct: list[bool], n_bins: int) -> float:
    """ECE with bins of (nearly) equal size over the confidence-sorted items (adaptive ECE).

    Ties are split by original order (stable sort), so with coarse confidences (e.g. Jev's
    2 decimals) tied items can land in neighbouring bins.
    """
    if not confidences:
        return 0.0
    confidence = np.asarray(confidences, dtype=float)
    accuracy = np.asarray(correct, dtype=float)
    order = np.argsort(confidence, kind="stable")
    return float(sum(len(b) / len(order) * abs(accuracy[b].mean() - confidence[b].mean())
                     for b in np.array_split(order, min(n_bins, len(order)))))


def brier_top1(confidences: list[float], correct: list[bool]) -> float:
    """Mean squared error of the top-1 confidence against correctness."""
    return float(np.mean((np.asarray(confidences, dtype=float) - np.asarray(correct, dtype=float)) ** 2))


def auroc(confidences: list[float], correct: list[bool]) -> float | None:
    """P(conf of a correct item > conf of a wrong item), ties count 1/2. None if only one class."""
    confidence = np.asarray(confidences, dtype=float)
    hits = np.asarray(correct, dtype=bool)
    pos, neg = confidence[hits], confidence[~hits]
    if not len(pos) or not len(neg):
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def bootstrap_delta_samples(records: list[dict[str, Any]], n_bins: int, n_bootstrap: int = 5000,
                            seed: int = 42) -> np.ndarray:
    """Paired bootstrap samples of ECE_jev - ECE_llm (resampling example ids)."""
    rng = np.random.default_rng(seed)
    jev = np.asarray([row["jev"] for row in records], dtype=float)
    llm = np.asarray([row["llm"] for row in records], dtype=float)
    correct_jev = np.asarray([row["correct_jev"] for row in records], dtype=bool)
    correct_llm = np.asarray([row["correct_llm"] for row in records], dtype=bool)
    if len(jev) == 0:
        raise ValueError("Cannot bootstrap an empty paired sample")
    differences = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(jev), len(jev))
        differences[index] = (
            expected_calibration_error(jev[sample].tolist(), correct_jev[sample].tolist(), n_bins)
            - expected_calibration_error(llm[sample].tolist(), correct_llm[sample].tolist(), n_bins)
        )
    return differences


def paired_bootstrap_ece(records: list[dict[str, Any]], n_bins: int, n_bootstrap: int = 5000,
                         seed: int = 42) -> dict[str, float]:
    differences = bootstrap_delta_samples(records, n_bins, n_bootstrap, seed)
    delta = (expected_calibration_error([r["jev"] for r in records], [r["correct_jev"] for r in records], n_bins)
             - expected_calibration_error([r["llm"] for r in records], [r["correct_llm"] for r in records], n_bins))
    return {
        "delta_ece": float(delta),
        "ci95_low": float(np.percentile(differences, 2.5)),
        "ci95_high": float(np.percentile(differences, 97.5)),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def bootstrap_system_ci(confidences: list[float], correct: list[bool], n_bins: int, n_bootstrap: int = 5000,
                        seed: int = 42) -> dict[str, list[float]]:
    """Percentile 95% CIs for one system's ECE and accuracy."""
    rng = np.random.default_rng(seed)
    confidence = np.asarray(confidences, dtype=float)
    hits = np.asarray(correct, dtype=bool)
    eces, accuracies = np.empty(n_bootstrap), np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(confidence), len(confidence))
        eces[index] = expected_calibration_error(confidence[sample].tolist(), hits[sample].tolist(), n_bins)
        accuracies[index] = hits[sample].mean()
    return {
        "ece_ci95": [float(np.percentile(eces, 2.5)), float(np.percentile(eces, 97.5))],
        "accuracy_ci95": [float(np.percentile(accuracies, 2.5)), float(np.percentile(accuracies, 97.5))],
    }


def save_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

