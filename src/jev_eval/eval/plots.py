from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LANGUAGE_TITLES = {"en": "EN (exploratório)", "pt_br": "PT-BR (principal)"}


def _bin_points(rows: list[dict], n_bins: int) -> tuple[list[float], list[float]]:
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for index in range(n_bins):
        upper_ok = (lambda c: c <= edges[index + 1]) if index == n_bins - 1 else (lambda c: c < edges[index + 1])
        selected = [row for row in rows if edges[index] <= row["confidence"] and upper_ok(row["confidence"])]
        if selected:
            xs.append(float(np.mean([row["confidence"] for row in selected])))
            ys.append(float(np.mean([row["correct"] for row in selected])))
    return xs, ys


def reliability_diagram(path: Path, rows: list[dict], n_bins: int) -> None:
    """One reliability panel per language; failed calls are excluded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = [row for row in rows if row.get("error") is None]
    languages = [lang for lang in ("pt_br", "en") if any(r["language"] == lang for r in valid)]
    languages += sorted({r["language"] for r in valid} - set(languages))
    fig, axes = plt.subplots(1, max(1, len(languages)), figsize=(5 * max(1, len(languages)), 5), squeeze=False)
    for axis, language in zip(axes[0], languages):
        for system in ("jev", "llm"):
            system_rows = [r for r in valid if r["language"] == language and r["system"] == system]
            xs, ys = _bin_points(system_rows, n_bins)
            axis.plot(xs, ys, marker="o", label=f"{system} (n={len(system_rows)})")
        axis.plot([0, 1], [0, 1], "--", color="gray")
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confiança média", ylabel="Acurácia",
                 title=LANGUAGE_TITLES.get(language, language))
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
