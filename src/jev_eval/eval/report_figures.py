"""One figure per question of the study, sized for a two-column ACM page.

- Q1 (H1): ECE of each system per condition, with the paired difference Jev - LLM.
- Q2: change of ECE and accuracy from EN (C1) to each PT condition, per system.
- Q3: accuracy when only the most confident answers are accepted (risk-coverage curves).

Inputs are the `metrics.json` of `analyze` (Q1, Q2) and the run's records (Q3), so every
number on a figure is the same one the tables report. Labels come in Portuguese or English;
each figure is written as .png and as vector .pdf (for LaTeX).
"""
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter, ScalarFormatter  # noqa: E402

from jev_eval.eval.figures import COLORS, GRID, NEUTRAL, SURFACE, TEXT, TEXT2, _style, paired_by_condition  # noqa: E402
from jev_eval.eval.metrics import expected_risk_by_coverage  # noqa: E402

COLUMN_IN = 3.4  # ACM sigconf column width
EXTRA_COLORS = ["#2f9e44", "#8e44ad", "#b8860b", "#0e9aa7", "#c2185b"]
CONDITION_ORDER = ("C1", "C2", "C3")

STRINGS = {
    "pt": {
        "ece": "ECE (menor é melhor)", "accuracy": "Acurácia", "coverage": "Cobertura (respostas aceitas)",
        "content_instr": "conteúdo / instruções", "change_ece": "Δ ECE (PT − EN)",
        "change_acc": "Δ acurácia (PT − EN)", "prob": "prob.", "verb": "verbalizada",
        "q1": "Q1 · O Jev tem ECE menor que o LLM?", "q2": "Q2 · O que muda de EN para PT?",
        "q3": "Q3 · A confiança ajuda a escolher quando responder?", "delta": "Δ", "aurc": "AURC",
        "vs_c1": "vs C1",
    },
    "en": {
        "ece": "ECE (lower is better)", "accuracy": "Accuracy", "coverage": "Coverage (accepted answers)",
        "content_instr": "content / instructions", "change_ece": "Δ ECE (PT − EN)",
        "change_acc": "Δ accuracy (PT − EN)", "prob": "prob.", "verb": "verbalized",
        "q1": "Q1 · Is Jev's ECE lower than the LLM's?", "q2": "Q2 · What changes from EN to PT?",
        "q3": "Q3 · Does confidence help decide when to answer?", "delta": "Δ", "aurc": "AURC",
        "vs_c1": "vs C1",
    },
}


def _num(value: float, lang: str, digits: int = 2, sign: bool = False) -> str:
    text = f"{value:{'+' if sign else ''}.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "pt" else text


def condition_label(condition: str) -> str:
    return {"C1": "C1\nEN/EN", "C2": "C2\nPT/EN", "C3": "C3\nPT/PT"}.get(condition, condition)


def system_label(system: str, lang: str = "pt") -> str:
    if system in ("jev", "llm"):
        return {"jev": "Jev", "llm": "LLM"}[system]
    model, _, readout = system.rpartition("_")  # open models: "<model>_prob" / "<model>_verb"
    return f"{model.capitalize()} ({STRINGS[lang][readout]})" if model and readout in ("prob", "verb") else system


def system_colors(systems: list[str]) -> dict[str, str]:
    colors = {s: COLORS[s] for s in systems if s in COLORS}
    spare = iter(EXTRA_COLORS)
    for system in systems:
        colors.setdefault(system, next(spare, NEUTRAL))
    return colors


def _setup() -> None:
    _style()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 1.4})


def _finish(fig, path: Path, title: str, lang: str) -> list[Path]:
    if lang == "pt":  # decimal comma on every numeric axis
        comma = FuncFormatter(lambda value, _: f"{value:g}".replace("-", "−").replace(".", ","))
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                if isinstance(axis.get_major_formatter(), ScalarFormatter):
                    axis.set_major_formatter(comma)
    fig.suptitle(title, x=0.01, ha="left", fontsize=8, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    written = []
    for suffix in (".png", ".pdf"):
        target = path.with_suffix(suffix)
        fig.savefig(target, dpi=300, bbox_inches="tight", facecolor=SURFACE)
        written.append(target)
    plt.close(fig)
    return written


def q1_figure(metrics: dict[str, Any], path: Path, lang: str = "pt") -> list[Path]:
    """ECE of Jev and the LLM per condition (paired examples) and their difference with its 95% CI."""
    text = STRINGS[lang]
    conditions = [c for c in CONDITION_ORDER if metrics["conditions"].get(c, {}).get("paired_bootstrap")]
    if not conditions:
        raise ValueError("No condition with paired Jev and LLM answers: nothing to plot for Q1.")
    _setup()
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.5))
    colors = system_colors(["jev", "llm"])
    top = 0.0
    for offset, system in ((-0.11, "jev"), (0.11, "llm")):
        xs, ys, low, high = [], [], [], []
        for index, condition in enumerate(conditions):
            summary = metrics["conditions"][condition]["paired"][system]
            xs.append(index + offset)
            ys.append(summary["ece"])
            low.append(summary["ece"] - summary["ece_ci95"][0])
            high.append(summary["ece_ci95"][1] - summary["ece"])
            top = max(top, summary["ece_ci95"][1])
        ax.errorbar(xs, ys, yerr=[low, high], fmt="o", color=colors[system], capsize=2.5, markersize=4.5,
                    label=system_label(system, lang))
    for index, condition in enumerate(conditions):
        boot = metrics["conditions"][condition]["paired_bootstrap"]
        ax.text(index, top * 1.08, f"{text['delta']} = {_num(boot['delta_ece'], lang, 3, True)}\n"
                f"[{_num(boot['ci95_low'], lang, 3)}; {_num(boot['ci95_high'], lang, 3)}]",
                ha="center", va="bottom", fontsize=6, color=TEXT2)
    ax.set_xticks(range(len(conditions)), [condition_label(c) for c in conditions])
    ax.set_xlim(-0.5, len(conditions) - 0.5)
    ax.set_ylim(0, top * 1.38)
    ax.set_ylabel(text["ece"])
    ax.set_xlabel(text["content_instr"])
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.74), ncols=1, handletextpad=0.3)
    return _finish(fig, path, text["q1"], lang)


def q2_figure(metrics: dict[str, Any], path: Path, lang: str = "pt") -> list[Path]:
    """Change from C1 to each PT condition, per system: delta ECE (left) and delta accuracy (right)."""
    text = STRINGS[lang]
    shifts = metrics.get("language_shift") or {}
    builtin = ["jev", "llm"]
    systems = sorted(shifts, key=lambda s: (s not in builtin, builtin.index(s) if s in builtin else 0, s))
    rows = [(system, condition) for system in systems for condition in ("C2", "C3") if condition in shifts[system]]
    if not rows:
        raise ValueError("No system answered both C1 and a PT condition: nothing to plot for Q2.")
    _setup()
    colors = system_colors(systems)
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_IN * 2.05, 0.55 + 0.32 * len(rows)), sharey=True)
    ys = np.arange(len(rows))[::-1]
    for ax, key, label in ((axes[0], "ece", text["change_ece"]), (axes[1], "accuracy", text["change_acc"])):
        for y, (system, condition) in zip(ys, rows):
            shift = shifts[system][condition]
            value, (low, high) = shift[f"delta_{key}"], shift[f"delta_{key}_ci95"]
            ax.errorbar(value, y, xerr=[[value - low], [high - value]], fmt="o" if condition == "C2" else "s",
                        color=colors[system], capsize=2.5, markersize=4.5,
                        markerfacecolor=colors[system] if condition == "C2" else SURFACE)
        ax.axvline(0, color=NEUTRAL, linewidth=0.8)
        ax.set_xlabel(label)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [f"{system_label(s, lang)}  {c} {text['vs_c1']}" for s, c in rows])
    return _finish(fig, path, text["q2"], lang)


def q3_figure(records: list[dict[str, Any]], path: Path, lang: str = "pt") -> list[Path]:
    """Accuracy of the answers kept at each coverage, ranking by confidence (paired examples)."""
    text = STRINGS[lang]
    paired = paired_by_condition(records)
    conditions = [c for c in CONDITION_ORDER if c in paired]
    if not conditions:
        raise ValueError("No condition with paired Jev and LLM answers: nothing to plot for Q3.")
    _setup()
    systems = sorted({s for c in conditions for s in paired[c]}, key=lambda s: (s not in ("jev", "llm"), s))
    colors = system_colors(systems)
    fig, axes = plt.subplots(1, len(conditions), figsize=(COLUMN_IN * 2.05, 2.1), sharey=True, squeeze=False)
    lows = []
    aurc: dict[str, list[str]] = {c: [] for c in conditions}
    for ax, condition in zip(axes[0], conditions):
        for system in systems:
            rows = paired[condition].get(system)
            if not rows:
                continue
            risks = expected_risk_by_coverage([r["confidence"] for r in rows], [bool(r["correct"]) for r in rows])
            coverage = np.arange(1, len(risks) + 1) / len(risks)
            accuracy = 1 - risks
            lows.append(accuracy.min())
            ax.plot(coverage, accuracy, color=colors[system], label=system_label(system, lang))
            aurc[condition].append(f"{system_label(system, lang)} {_num(float(risks.mean()), lang, 3)}")
            for share in (0.5, 0.8):
                k = max(1, int(np.ceil(round(share * len(risks), 9)))) - 1
                ax.plot(coverage[k], accuracy[k], "o", color=colors[system], markersize=3.5)
        header = condition_label(condition).replace("\n", " · ")
        ax.set_title(f"{header}\n{text['aurc']}: {' · '.join(aurc[condition])}", loc="left", fontsize=6.5,
                     color=TEXT)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(text["coverage"])
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.axvline(0.5, color=GRID, linewidth=0.8, zorder=0)
        ax.axvline(0.8, color=GRID, linewidth=0.8, zorder=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=len(labels), bbox_to_anchor=(0.995, 0.995), fontsize=6.5)
    axes[0][0].set_ylabel(text["accuracy"])
    axes[0][0].set_ylim(max(0.0, min(lows) - 0.03), 1.01)
    axes[0][0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    return _finish(fig, path, text["q3"], lang)


def make_report_figures(out_dir: Path, metrics: dict[str, Any], records: list[dict[str, Any]],
                        lang: str = "pt") -> list[Path]:
    """Write q1_h1, q2_en_pt and q3_selection (.png and .pdf) into `out_dir`."""
    if lang not in STRINGS:
        raise ValueError(f"lang must be one of {sorted(STRINGS)}, got {lang!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return (q1_figure(metrics, out_dir / "q1_h1", lang) + q2_figure(metrics, out_dir / "q2_en_pt", lang)
            + q3_figure(records, out_dir / "q3_selection", lang))
