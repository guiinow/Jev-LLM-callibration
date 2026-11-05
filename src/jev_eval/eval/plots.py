from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def reliability_diagram(path: Path, rows: list[dict], n_bins: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5, 5))
    edges = np.linspace(0, 1, n_bins + 1)
    for system in ("jev", "llm"):
        system_rows = [row for row in rows if row["system"] == system]
        bins_x, bins_y = [], []
        for index in range(n_bins):
            selected = [row for row in system_rows
                        if edges[index] <= row["confidence"] <= edges[index + 1]
                        if index == n_bins - 1 or row["confidence"] < edges[index + 1]]
            if selected:
                bins_x.append(float(np.mean([row["confidence"] for row in selected])))
                bins_y.append(float(np.mean([row["correct"] for row in selected])))
        axis.plot(bins_x, bins_y, marker="o", label=system)
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confiança média", ylabel="Acurácia")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

