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


def paired_bootstrap_ece(records: list[dict[str, Any]], n_bins: int, n_bootstrap: int = 5000,
                         seed: int = 42) -> dict[str, float]:
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
    return {
        "delta_ece": float(np.mean(jev * 0 + np.array([
            expected_calibration_error(jev.tolist(), correct_jev.tolist(), n_bins)
        ])) - expected_calibration_error(llm.tolist(), correct_llm.tolist(), n_bins)),
        "ci95_low": float(np.percentile(differences, 2.5)),
        "ci95_high": float(np.percentile(differences, 97.5)),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def save_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

