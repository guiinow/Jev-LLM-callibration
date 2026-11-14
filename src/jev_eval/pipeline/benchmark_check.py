"""Does our English Jev run agree with the published Jev benchmark on Banking77?

Compares accuracy, ECE and the selection metrics of Jev in C1 (EN content + EN instructions)
with Deußer et al. (2026), Tables 10 and 11. Our sample is a stratified subset of the test split,
so each of our numbers has a wide 95% bootstrap interval: the check passes when the published
value falls inside it. That only catches large disagreements, and the report says so.
"""
from typing import Any

from jev_eval.eval.metrics import (bootstrap_selective_ci, bootstrap_system_ci, expected_calibration_error,
                                   selective_summary)

PUBLISHED = {
    "source": "Deußer et al. (2026), Tabelas 10 e 11 (Banking77, teste completo, n = 3.076)",
    "dataset": "banking77", "system": "jev", "condition": "C1", "n_bins": 15,
    "accuracy": 0.797, "ece": 0.087, "accuracy_at_0.8": 0.888, "accuracy_at_0.5": 0.963, "aurc": 0.061,
}
METRICS = (
    ("accuracy", "Acurácia"), ("ece", "ECE (15 faixas)"), ("accuracy_at_0.8", "Acurácia nos 80% mais confiantes"),
    ("accuracy_at_0.5", "Acurácia nos 50% mais confiantes"), ("aurc", "AURC"),
)


def benchmark_check(records: list[dict[str, Any]], n_bins: int, n_bootstrap: int = 5000) -> dict[str, Any]:
    rows = [r for r in records if r.get("error") is None and r.get("dataset") == PUBLISHED["dataset"]
            and r["system"] == PUBLISHED["system"] and r["condition"] == PUBLISHED["condition"]]
    if not rows:
        raise ValueError("No valid Jev answers in C1 for banking77: run the C1 condition with the Jev first.")
    confidences, correct = [r["confidence"] for r in rows], [bool(r["correct"]) for r in rows]
    system_ci = bootstrap_system_ci(confidences, correct, PUBLISHED["n_bins"], n_bootstrap)
    selective = selective_summary(confidences, correct)
    selective_ci = bootstrap_selective_ci(confidences, correct, n_bootstrap=n_bootstrap)
    ours = {
        "accuracy": (sum(correct) / len(correct), system_ci["accuracy_ci95"]),
        "ece": (expected_calibration_error(confidences, correct, PUBLISHED["n_bins"]), system_ci["ece_ci95"]),
        "accuracy_at_0.8": (selective["accuracy_at_coverage"]["0.8"], selective_ci["accuracy_at_0.8_ci95"]),
        "accuracy_at_0.5": (selective["accuracy_at_coverage"]["0.5"], selective_ci["accuracy_at_0.5_ci95"]),
        "aurc": (selective["aurc"], selective_ci["aurc_ci95"]),
    }
    metrics = {name: {"ours": value, "ci95": ci, "published": PUBLISHED[name],
                      "published_in_ci": ci[0] <= PUBLISHED[name] <= ci[1]}
               for name, (value, ci) in ours.items()}
    return {
        "source": PUBLISHED["source"], "n": len(rows), "metrics": metrics,
        "all_within_ci": all(m["published_in_ci"] for m in metrics.values()),
        "warnings": ([f"o ece_bins desta execução é {n_bins}; a conferência sempre usa {PUBLISHED['n_bins']} faixas, "
                      "como o benchmark."] if n_bins != PUBLISHED["n_bins"] else []),
    }


def format_report(result: dict[str, Any]) -> str:
    lines = [
        "# Conferência com o benchmark publicado (Jev, Banking77, inglês)", "",
        f"Nosso Jev em C1 (EN + instruções EN), {result['n']} exemplos, contra {result['source']}.", "",
        "| Métrica | Nosso | IC 95% | Publicado | Publicado dentro do IC? |", "|---|---|---|---|---|",
    ]
    for name, label in METRICS:
        m = result["metrics"][name]
        lines.append(f"| {label} | {m['ours']:.3f} | [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] | "
                     f"{m['published']:.3f} | {'sim' if m['published_in_ci'] else '**não**'} |")
    outside = [label for name, label in METRICS if not result["metrics"][name]["published_in_ci"]]
    lines += ["", ("**Bate:** todos os valores publicados caem dentro dos nossos intervalos." if not outside
                   else f"**Difere em:** {', '.join(outside)}. Investigar antes de confiar nos resultados em PT."),
              "", "Limites: com poucos exemplos os intervalos são largos, então este teste só detecta diferenças "
              "grandes. O benchmark usa o teste completo (`mteb/banking77`, 3.076 exemplos) e a nossa amostra é "
              "um subconjunto estratificado do teste do PolyAI (3.080)."]
    if result["warnings"]:
        lines += ["", *(f"Aviso: {w}" for w in result["warnings"])]
    return "\n".join(lines) + "\n"
