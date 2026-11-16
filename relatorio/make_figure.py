"""Figura do relatório: ECE (15 faixas) com IC 95% por dataset, sistema e condição.

Lê os `readout_cells.csv` gravados pelo `jev-eval analyze` (execução final, branch
`resultados/final-l40s`) e grava `relatorio/figs/ece_datasets.pdf`. Nenhum número novo é
calculado aqui: só os valores já presentes nas tabelas.

Uso: python relatorio/make_figure.py [pasta_da_execucao_final]   (padrão: outputs/final)
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/final")
OUT = Path(__file__).parent / "figs" / "ece_datasets.pdf"
DATASETS = [("banking77", "Banking77 (77)"), ("clinc150", "CLINC150 (150)"), ("sib200", "SIB-200 (7)")]
SYSTEMS = [("jev", "Jev (prob.)", "#2a6fdb"), ("gemma_prob", "Gemma (prob.)", "#7fb3e6"),
           ("gemma_verb", "Gemma (verb.)", "#e8a33d"), ("llm", "Ministral (verb.)", "#d9534f")]
CONDITIONS = ["C1", "C2", "C3"]


def load(dataset: str) -> dict[tuple[str, str], dict[str, float]]:
    path = RUN / dataset / "tables" / "readout_cells.csv"
    with path.open(encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle, delimiter=";")
        return {(r["system"], r["condition"]): {k: float(r[k]) for k in ("ece", "ece_ci_low", "ece_ci_high")}
                for r in rows}


def main() -> None:
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.75), sharey=True)
    width = 0.19
    for ax, (dataset, title) in zip(axes, DATASETS):
        cells = load(dataset)
        for i, (system, label, color) in enumerate(SYSTEMS):
            for j, condition in enumerate(CONDITIONS):
                c = cells[(system, condition)]
                x = j + (i - 1.5) * width
                # IC percentil do bootstrap; pode não conter a estimativa pontual (viés do ECE)
                low, high = min(c["ece_ci_low"], c["ece"]), max(c["ece_ci_high"], c["ece"])
                ax.errorbar(x, c["ece"], yerr=[[c["ece"] - low], [high - c["ece"]]], fmt="o", ms=2.2,
                            color=color, elinewidth=0.7, capsize=0, label=label if j == 0 else None)
        ax.set_xticks(range(3), CONDITIONS)
        ax.set_title(title, fontsize=7)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("ECE (15 faixas)")
    axes[0].set_ylim(0, 0.33)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=5.6,
               frameon=False, handletextpad=0.2, columnspacing=0.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), pad=0.3, w_pad=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    print(f"gravado: {OUT}")


if __name__ == "__main__":
    main()
