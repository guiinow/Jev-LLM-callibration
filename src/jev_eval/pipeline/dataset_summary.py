"""Equal-weight summary across datasets (refined proposal: results per dataset, summarised as the
mean over datasets with equal weights, never pooling predictions).

For every system present (Jev, the Ministral LLM, and open models such as gemma_prob/gemma_verb):
- `ece`: ECE per condition, on the answers that system got right or wrong (all valid answers);
- `jev_minus`: ECE_jev - ECE_system per condition, on the items both answered (Q1 style);
- `en_to_pt`: ECE(PT condition) - ECE(C1) per system, on the items it answered in both (Q2).

Each dataset gets a paired bootstrap (resampling its example ids, `n_bootstrap` draws, fixed seed);
the summary is the mean of the per-dataset estimates, and its 95% interval comes from averaging
the b-th draw of every dataset (datasets are resampled independently).
"""
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from jev_eval.eval.metrics import expected_calibration_error, save_metrics

PT_CONDITIONS = ("C2", "C3")


def _ece(conf: np.ndarray, hit: np.ndarray, n_bins: int) -> float:
    return expected_calibration_error(conf.tolist(), hit.tolist(), n_bins)


def _paired_draws(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], n_bins: int,
                  n_bootstrap: int, seed: int) -> tuple[float, np.ndarray]:
    """ECE(a) - ECE(b) and its bootstrap draws, resampling the shared items together."""
    rng = np.random.default_rng(seed)
    n = len(a[0])
    draws = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        draws[i] = _ece(a[0][idx], a[1][idx], n_bins) - _ece(b[0][idx], b[1][idx], n_bins)
    return _ece(*a, n_bins) - _ece(*b, n_bins), draws


def _arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray([r["confidence"] for r in rows], dtype=float),
            np.asarray([bool(r["correct"]) for r in rows], dtype=bool))


def _summary(per_dataset: dict[str, float], draws: list[np.ndarray] | None) -> dict[str, Any]:
    out: dict[str, Any] = {"per_dataset": {k: round(v, 4) for k, v in per_dataset.items()},
                           "n_datasets": len(per_dataset), "mean": round(float(np.mean(list(per_dataset.values()))), 4)}
    if draws:
        mean_draws = np.mean(np.vstack(draws), axis=0)
        out["ci95"] = [round(float(np.percentile(mean_draws, 2.5)), 4), round(float(np.percentile(mean_draws, 97.5)), 4)]
        out["ci_excludes_zero"] = not out["ci95"][0] <= 0 <= out["ci95"][1]
    return out


def dataset_summary(records: list[dict[str, Any]], n_bins: int, n_bootstrap: int = 5000,
                    seed: int = 42) -> dict[str, Any]:
    valid: dict[str, dict[str, dict[str, dict[str, dict]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in records:
        if r.get("error") is None:
            valid[r["dataset"]][r["condition"]][r["system"]][r["example_id"]] = r
    datasets = sorted(valid)
    conditions = sorted({c for d in datasets for c in valid[d]})
    systems = sorted({s for d in datasets for c in valid[d] for s in valid[d][c]})
    result: dict[str, Any] = {"datasets": datasets, "n_bins": n_bins, "n_bootstrap": n_bootstrap, "seed": seed,
                              "ece": {}, "jev_minus": {}, "en_to_pt": {}}
    for condition in conditions:
        for system in systems:
            per = {d: _ece(*_arrays(list(valid[d][condition][system].values())), n_bins)
                   for d in datasets if valid[d][condition].get(system)}
            if per:
                result["ece"].setdefault(condition, {})[system] = _summary(per, None)
            if system == "jev":
                continue
            per, draws = {}, []
            for d in datasets:
                ids = sorted(set(valid[d][condition].get("jev", {})) & set(valid[d][condition].get(system, {})))
                if ids:
                    jev = _arrays([valid[d][condition]["jev"][i] for i in ids])
                    other = _arrays([valid[d][condition][system][i] for i in ids])
                    per[d], sample = _paired_draws(jev, other, n_bins, n_bootstrap, seed)
                    draws.append(sample)
            if per:
                result["jev_minus"].setdefault(condition, {})[system] = _summary(per, draws)
    for system in systems:
        for condition in PT_CONDITIONS:
            per, draws = {}, []
            for d in datasets:
                en, pt = valid[d].get("C1", {}).get(system, {}), valid[d].get(condition, {}).get(system, {})
                ids = sorted(set(en) & set(pt))
                if ids:
                    per[d], sample = _paired_draws(_arrays([pt[i] for i in ids]), _arrays([en[i] for i in ids]),
                                                   n_bins, n_bootstrap, seed)
                    draws.append(sample)
            if per:
                result["en_to_pt"].setdefault(system, {})[f"{condition}-C1"] = _summary(per, draws)
    return result


def summary_rows(result: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    datasets = result["datasets"]
    header = ["measure", "condition", "system", "n_datasets", "mean", "ci95_low", "ci95_high"] + datasets
    rows = []

    def add(measure: str, condition: str, system: str, s: dict[str, Any]) -> None:
        ci = s.get("ci95", [None, None])
        rows.append([measure, condition, system, s["n_datasets"], s["mean"], ci[0], ci[1]]
                    + [s["per_dataset"].get(d) for d in datasets])

    for condition, by_system in result["ece"].items():
        for system, s in by_system.items():
            add("ece", condition, system, s)
    for condition, by_system in result["jev_minus"].items():
        for system, s in by_system.items():
            add("ece_jev_minus_system", condition, system, s)
    for system, by_contrast in result["en_to_pt"].items():
        for contrast, s in by_contrast.items():
            add("ece_pt_minus_en", contrast, system, s)
    return header, rows


def write_dataset_summary(records: list[dict[str, Any]], n_bins: int, output_dir: Path,
                          n_bootstrap: int = 5000, seed: int = 42) -> list[Path]:
    result = dataset_summary(records, n_bins, n_bootstrap, seed)
    json_path, csv_path = output_dir / "summary_datasets.json", output_dir / "tables" / "summary_datasets.csv"
    save_metrics(json_path, result)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header, rows = summary_rows(result)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)
    return [json_path, csv_path]
