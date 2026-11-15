"""Categorized error analysis (E3): a sheet the group fills by hand, and the counts from it.

`error_rows` lists every wrong (valid) answer of the chosen systems in one condition, with the
message in PT and EN, the gold and predicted options, the confidence, and what the other systems
answered. The group fills `category` with one of CATEGORIES (and `notes`); `summarize` counts them
by system and confidence band, refusing unknown or missing categories.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jev_eval.data.datasets import Example

HIGH_CONFIDENCE = 0.9

# Fixed taxonomy (the group may add to it before filling, never after counting).
CATEGORIES = {
    "intencoes_proximas": "Confundiu opções vizinhas de sentido parecido (ex.: tarifa x taxa de câmbio).",
    "rotulo_ambiguo": "A mensagem cabe em mais de uma opção, ou o rótulo do dataset é discutível.",
    "traducao": "A tradução da mensagem (ou da opção) mudou ou apagou o sentido.",
    "falta_contexto": "Mensagem curta ou vaga demais para decidir.",
    "erro_do_modelo": "Erro claro do sistema, sem justificativa aparente.",
    "outro": "Nenhuma das anteriores (explicar em notes).",
}

HEADER = ["dataset", "condition", "example_id", "system", "text", "text_en", "true_label", "predicted_label",
          "confidence", "high_confidence", "others", "category", "notes"]


def error_rows(records: list[dict[str, Any]], examples: list[Example], dataset: str, condition: str,
               systems: list[str]) -> list[dict[str, Any]]:
    by_id = {ex.example_id: ex for ex in examples}
    answers: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        if r.get("dataset") == dataset and r["condition"] == condition and r.get("error") is None:
            answers[r["example_id"]][r["system"]] = r
    out = []
    for example_id in sorted(answers):
        for system in systems:
            r = answers[example_id].get(system)
            if r is None or r["correct"]:
                continue
            ex = by_id[example_id]
            others = "; ".join(f"{s}: {o['predicted_label_name']} ({'ok' if o['correct'] else 'erro'}, "
                               f"{o['confidence']:.2f})" for s, o in sorted(answers[example_id].items()) if s != system)
            out.append({"dataset": dataset, "condition": condition, "example_id": example_id, "system": system,
                        "text": ex.text.get("pt_br" if condition != "C1" else "en", ""), "text_en": ex.text.get("en", ""),
                        "true_label": r["true_label_name"], "predicted_label": r["predicted_label_name"],
                        "confidence": round(float(r["confidence"]), 4),
                        "high_confidence": float(r["confidence"]) >= HIGH_CONFIDENCE, "others": others,
                        "category": "", "notes": ""})
    return out


def write_sheet(path: Path, rows: list[dict[str, Any]], force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists (it may be filled in); use --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HEADER, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def read_sheet(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=";"))


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Counts by system x category and by confidence band; refuses blank or unknown categories."""
    bad = [(r["example_id"], r["system"], r["category"]) for r in rows if r["category"].strip() not in CATEGORIES]
    if bad:
        raise ValueError(f"{len(bad)} rows have a blank or unknown category (e.g. {bad[:3]}); "
                         f"use one of {sorted(CATEGORIES)}")
    by_system: dict[str, Counter] = defaultdict(Counter)
    high: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_system[r["system"]][r["category"].strip()] += 1
        if str(r["high_confidence"]).lower() == "true":
            high[r["system"]][r["category"].strip()] += 1
    return {"n_errors": {s: sum(c.values()) for s, c in by_system.items()},
            "by_system": {s: dict(c) for s, c in by_system.items()},
            "high_confidence_by_system": {s: dict(c) for s, c in high.items()}}


def summary_table(summary: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    rows = []
    for system, counts in sorted(summary["by_system"].items()):
        total = summary["n_errors"][system]
        for category in CATEGORIES:
            n, n_high = counts.get(category, 0), summary["high_confidence_by_system"].get(system, {}).get(category, 0)
            rows.append([system, category, n, round(n / total, 4) if total else None, n_high])
    return ["system", "category", "n", "share", "n_high_confidence"], rows
