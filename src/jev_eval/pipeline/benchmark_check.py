"""Do our English runs agree with the published Jev benchmark on Banking77?

Compares the English condition (C1: EN content + EN instructions) with Deußer et al. (2026):
Tables 10 and 11 for Jev (accuracy, ECE and the selection metrics) and Table 2 for Qwen3.8-27B
and Gemma-4-E4B read by option probabilities (accuracy and ECE only). Our sample is a stratified
subset of the test split, so each number has a wide 95% bootstrap interval: the check passes when
the published value falls inside it. That only catches large disagreements, and the report says so.
"""
from typing import Any

from jev_eval.eval.metrics import (bootstrap_selective_ci, bootstrap_system_ci, expected_calibration_error,
                                   selective_summary)

SOURCE = "Deußer et al. (2026), Tabelas 2, 10 e 11 (Banking77, teste completo, n = 3.076)"
N_BINS = 15
CONDITION = "C1"
DATASET = "banking77"

# Published numbers by system name; only Jev has the selection metrics in the paper.
PUBLISHED_BY_SYSTEM: dict[str, dict[str, float]] = {
    "jev": {"accuracy": 0.797, "ece": 0.087, "accuracy_at_0.8": 0.888, "accuracy_at_0.5": 0.963, "aurc": 0.061},
    "qwen_prob": {"accuracy": 0.774, "ece": 0.094},
    "gemma_prob": {"accuracy": 0.670, "ece": 0.267},
}
PUBLISHED = {"source": SOURCE, "dataset": DATASET, "system": "jev", "condition": CONDITION, "n_bins": N_BINS,
             **PUBLISHED_BY_SYSTEM["jev"]}
METRICS = (
    ("accuracy", "Acurácia"), ("ece", "ECE (15 faixas)"), ("accuracy_at_0.8", "Acurácia nos 80% mais confiantes"),
    ("accuracy_at_0.5", "Acurácia nos 50% mais confiantes"), ("aurc", "AURC"),
)


def checkable_systems(records: list[dict[str, Any]]) -> list[str]:
    """Systems with a published Banking77 reference that have valid C1 answers in `records`."""
    present = {r["system"] for r in records if r.get("error") is None and r.get("dataset") == DATASET
               and r["condition"] == CONDITION}
    return [system for system in PUBLISHED_BY_SYSTEM if system in present]


def benchmark_check(records: list[dict[str, Any]], n_bins: int, n_bootstrap: int = 5000,
                    system: str = "jev") -> dict[str, Any]:
    published = PUBLISHED_BY_SYSTEM[system]
    rows = [r for r in records if r.get("error") is None and r.get("dataset") == DATASET
            and r["system"] == system and r["condition"] == CONDITION]
    if not rows:
        raise ValueError(f"No valid {system} answers in C1 for banking77: run the C1 condition with it first.")
    confidences, correct = [r["confidence"] for r in rows], [bool(r["correct"]) for r in rows]
    system_ci = bootstrap_system_ci(confidences, correct, N_BINS, n_bootstrap)
    ours = {
        "accuracy": (sum(correct) / len(correct), system_ci["accuracy_ci95"]),
        "ece": (expected_calibration_error(confidences, correct, N_BINS), system_ci["ece_ci95"]),
    }
    if "aurc" in published:
        selective = selective_summary(confidences, correct)
        selective_ci = bootstrap_selective_ci(confidences, correct, n_bootstrap=n_bootstrap)
        ours.update({
            "accuracy_at_0.8": (selective["accuracy_at_coverage"]["0.8"], selective_ci["accuracy_at_0.8_ci95"]),
            "accuracy_at_0.5": (selective["accuracy_at_coverage"]["0.5"], selective_ci["accuracy_at_0.5_ci95"]),
            "aurc": (selective["aurc"], selective_ci["aurc_ci95"]),
        })
    metrics = {name: {"ours": value, "ci95": ci, "published": published[name],
                      "published_in_ci": ci[0] <= published[name] <= ci[1]}
               for name, (value, ci) in ours.items()}
    return {
        "system": system, "source": SOURCE, "n": len(rows), "metrics": metrics,
        "all_within_ci": all(m["published_in_ci"] for m in metrics.values()),
        "warnings": ([f"o ece_bins desta execução é {n_bins}; a conferência sempre usa {N_BINS} faixas, "
                      "como o benchmark."] if n_bins != N_BINS else []),
    }


def format_report(result: dict[str, Any]) -> str:
    system = result.get("system", "jev")
    lines = [
        f"# Conferência com o benchmark publicado ({system}, Banking77, inglês)", "",
        f"Nosso {system} em C1 (EN + instruções EN), {result['n']} exemplos, contra {result['source']}.", "",
        "| Métrica | Nosso | IC 95% | Publicado | Publicado dentro do IC? |", "|---|---|---|---|---|",
    ]
    shown = [(name, label) for name, label in METRICS if name in result["metrics"]]
    for name, label in shown:
        m = result["metrics"][name]
        lines.append(f"| {label} | {m['ours']:.3f} | [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] | "
                     f"{m['published']:.3f} | {'sim' if m['published_in_ci'] else '**não**'} |")
    outside = [label for name, label in shown if not result["metrics"][name]["published_in_ci"]]
    lines += ["", ("**Bate:** todos os valores publicados caem dentro dos nossos intervalos." if not outside
                   else f"**Difere em:** {', '.join(outside)}. Investigar antes de confiar nos resultados em PT."),
              "", "Limites: com poucos exemplos os intervalos são largos, então este teste só detecta diferenças "
              "grandes. O benchmark usa o teste completo (`mteb/banking77`, 3.076 exemplos) e a nossa amostra é "
              "um subconjunto estratificado do teste do PolyAI (3.080)."]
    if result["warnings"]:
        lines += ["", *(f"Aviso: {w}" for w in result["warnings"])]
    return "\n".join(lines) + "\n"
