
"""Report figures and tables for one run (raw_predictions.jsonl -> figures/ + tables/).

All figures use the paired subset (examples where both systems returned a valid answer),
the same items behind the headline ECE and the bootstrap, so the systems are always
compared on identical inputs. Failed calls are listed in tables/summary.csv.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jev_eval.eval.metrics import (
    bootstrap_delta_samples,
    bootstrap_system_ci,
    expected_calibration_error,
)


SYSTEMS = ("jev", "llm")
SYSTEM_LABELS = {"jev": "Jev", "llm": "LLM"}

LANGUAGE_LABELS = {
    "pt_br": "PT-BR (principal)",
    "en": "EN (exploratório)",
}
LANGUAGE_ORDER = ("pt_br", "en")

HIGH_CONFIDENCE = 0.9

# Consistent palette across all figures.
COLORS = {
    "jev": "#2a78d6",
    "llm": "#eb6834",
}

CORRECT = "#6250d6"
WRONG = "#e34948"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8984"


# ---------------------------------------------------------------------
# Global style and formatting
# ---------------------------------------------------------------------

def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,

        "font.family": "DejaVu Sans",
        "font.size": 9,

        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,

        "axes.edgecolor": NEUTRAL,
        "axes.labelcolor": TEXT2,
        "xtick.color": TEXT2,
        "ytick.color": TEXT2,

        "text.color": TEXT,
        "axes.titlecolor": TEXT,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,

        "legend.frameon": False,
        "lines.linewidth": 2,

        "savefig.dpi": 200,
    })


def _save(fig, path: Path) -> None:
    """Apply Brazilian decimal formatting and save the figure."""

    from matplotlib.ticker import FuncFormatter, ScalarFormatter

    comma = FuncFormatter(
        lambda value, _: f"{value:g}".replace(".", ",")
    )

    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(comma)

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        facecolor=SURFACE,
    )

    plt.close(fig)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _finish(fig, path: Path, title: str, bottom_in: float = 0.0) -> None:
    """Apply consistent title, spacing and export.

    Header and footer space are reserved in inches (not as a fraction of the
    figure), so short and tall figures get the same gap under the title.
    `bottom_in` reserves extra room for a figure-level legend at the bottom.
    """

    import warnings

    _, height_in = fig.get_size_inches()

    # fig.text (not suptitle): tight_layout already reserves room for a
    # suptitle, which would double the gap with the rect below.
    fig.text(
        0.06,
        1 - 0.08 / height_in,
        title,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )

    top = 1 - 0.42 / height_in
    bottom = (0.05 + bottom_in) / height_in

    with warnings.catch_warnings():
        # Mixed-height grids (reliability diagram) trigger a harmless warning.
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=(0.02, bottom, 0.98, top))

    _save(fig, path)


def _panel_title(
    ax,
    lang: str,
    n: int | None = None,
) -> None:
    label = LANGUAGE_LABELS.get(lang, lang)

    ax.set_title(
        label if n is None else f"{label} · n = {n}",
        loc="left",
        fontsize=10,
        color=TEXT,
        fontweight="bold",
    )


def _annotate_bar(
    ax,
    bars,
    values,
    color=TEXT2,
    fmt="{:.0f}",
) -> None:
    """Add readable values at the end of horizontal bars."""

    for bar, value in zip(bars, values):
        if value > 0:
            ax.annotate(
                fmt.format(value).replace(".", ","),
                xy=(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                ),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=color,
            )


# ---------------------------------------------------------------------
# Paired data and metrics
# ---------------------------------------------------------------------

def paired_by_language(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict]]]:
    """
    Return {language: {system: rows}} restricted to example_ids
    where both systems returned valid predictions.

    Both systems use exactly the same example ordering.
    """

    valid: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for row in records:
        if row.get("error") is None:
            valid[row["language"]][row["system"]][row["example_id"]] = row

    out: dict[str, dict[str, list[dict]]] = {}

    languages = (
        [lang for lang in LANGUAGE_ORDER if lang in valid]
        + sorted(set(valid) - set(LANGUAGE_ORDER))
    )

    for language in languages:
        ids = sorted(
            set.intersection(
                *(
                    set(valid[language].get(system, {}))
                    for system in SYSTEMS
                )
            )
        )

        if ids:
            out[language] = {
                system: [
                    valid[language][system][example_id]
                    for example_id in ids
                ]
                for system in SYSTEMS
            }

    return out


def _bins(
    rows: list[dict],
    n_bins: int,
) -> list[dict]:
    conf = np.array(
        [r["confidence"] for r in rows],
        dtype=float,
    )

    hit = np.array(
        [r["correct"] for r in rows],
        dtype=float,
    )

    edges = np.linspace(0, 1, n_bins + 1)

    out = []

    for i in range(n_bins):
        mask = (
            (conf >= edges[i])
            & (
                (conf <= edges[i + 1])
                if i == n_bins - 1
                else (conf < edges[i + 1])
            )
        )

        out.append({
            "lo": edges[i],
            "hi": edges[i + 1],
            "count": int(mask.sum()),
            "conf": float(conf[mask].mean()) if mask.any() else None,
            "acc": float(hit[mask].mean()) if mask.any() else None,
        })

    return out


def _ece(
    rows: list[dict],
    n_bins: int,
) -> float:
    return expected_calibration_error(
        [r["confidence"] for r in rows],
        [r["correct"] for r in rows],
        n_bins,
    )


# ---------------------------------------------------------------------
# Figure 1: Reliability diagram
# ---------------------------------------------------------------------

def reliability_figure(
    path: Path,
    paired: dict,
    n_bins: int,
) -> None:
    """
    Reliability diagram.

    Top: observed accuracy vs. reported confidence.
    Bottom: confidence distribution per bin.
    Marker size represents the number of examples in the bin.
    """

    langs = list(paired)

    fig, axes = plt.subplots(
        2,
        len(langs),
        figsize=(5.2 * len(langs), 6.2),
        squeeze=False,
        gridspec_kw={
            "height_ratios": [3, 1],
            "hspace": 0.12,
        },
    )

    width = 1 / n_bins

    centers = (
        np.linspace(0, 1, n_bins + 1)[:-1]
        + width / 2
    )

    for col, lang in enumerate(langs):
        top = axes[0][col]
        bottom = axes[1][col]

        n = len(paired[lang]["jev"])

        # Perfect calibration reference.
        top.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.5,
            color=NEUTRAL,
            label="Calibração perfeita",
            zorder=1,
        )

        for k, system in enumerate(SYSTEMS):
            rows = paired[lang][system]

            bins = _bins(rows, n_bins)
            populated = [b for b in bins if b["count"]]

            xs = [b["conf"] for b in populated]
            ys = [b["acc"] for b in populated]

            sizes = [
                30 + 220 * b["count"] / n
                for b in populated
            ]

            top.scatter(
                xs,
                ys,
                s=sizes,
                color=COLORS[system],
                edgecolor=SURFACE,
                linewidth=1.2,
                alpha=0.9,
                zorder=3,
                label=(
                    f"{SYSTEM_LABELS[system]} "
                    f"(ECE = {_fmt(_ece(rows, n_bins))})"
                ),
            )

            offset = (k - 0.5) * width * 0.42

            bottom.bar(
                centers + offset,
                [b["count"] for b in bins],
                width=width * 0.4,
                color=COLORS[system],
                edgecolor=SURFACE,
                linewidth=0.8,
            )

        top.set(
            xlim=(0, 1),
            ylim=(0, 1.05),
            ylabel="Acurácia observada",
            xlabel="Confiança reportada",
        )

        _panel_title(top, lang, n)

        top.legend(
            loc="lower right",
            frameon=True,
            facecolor=SURFACE,
            edgecolor="none",
        )

        bottom.set(
            xlim=(0, 1),
            xlabel="Confiança reportada",
            ylabel="Respostas",
        )

        bottom.grid(axis="x", visible=False)

    _finish(
        fig,
        path,
        "Diagrama de confiabilidade: confiança reportada vs. acurácia observada",
    )


# ---------------------------------------------------------------------
# Figure 2: ECE and accuracy comparison
# ---------------------------------------------------------------------

def summary_figure(
    path: Path,
    paired: dict,
    n_bins: int,
) -> None:
    """
    Compare ECE and accuracy across systems and languages.

    Points: observed estimates.
    Error bars: 95% bootstrap confidence intervals.
    """

    langs = list(paired)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 1.5 + 1.0 * len(langs)),
        sharey=True,
    )

    panels = [
        (
            "ece",
            "Expected Calibration Error\n(menor é melhor)",
            0.45,
        ),
        (
            "accuracy",
            "Acurácia\n(maior é melhor)",
            1.0,
        ),
    ]

    for ax, (metric, title, xmax) in zip(axes, panels):

        for row_i, lang in enumerate(langs):

            for k, system in enumerate(SYSTEMS):

                rows = paired[lang][system]

                conf = [
                    r["confidence"]
                    for r in rows
                ]

                hit = [
                    r["correct"]
                    for r in rows
                ]

                value = (
                    _ece(rows, n_bins)
                    if metric == "ece"
                    else float(np.mean(hit))
                )

                ci = bootstrap_system_ci(
                    conf,
                    hit,
                    n_bins,
                    n_bootstrap=2000,
                )

                lo, hi = ci[f"{metric}_ci95"]

                y = row_i + (k - 0.5) * 0.30

                ax.errorbar(
                    value,
                    y,
                    xerr=[
                        [value - lo],
                        [hi - value],
                    ],
                    fmt="o",
                    color=COLORS[system],
                    markersize=7,
                    capsize=4,
                    elinewidth=2,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1,
                    label=(
                        SYSTEM_LABELS[system]
                        if row_i == 0
                        else None
                    ),
                    zorder=3,
                )

                ax.annotate(
                    _fmt(value),
                    (value, y),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=COLORS[system],
                )

        ax.set_title(
            title,
            loc="left",
            fontsize=10,
            fontweight="bold",
        )

        ax.set_xlim(0, xmax)

        ax.grid(axis="y", visible=False)

        if metric == "accuracy":
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(
                    1.0,
                    decimals=0,
                )
            )

    axes[0].set_yticks(
        range(len(langs)),
        [
            LANGUAGE_LABELS.get(lang, lang)
            for lang in langs
        ],
    )

    axes[0].set_ylim(
        len(langs) - 0.4,
        -0.6,
    )

    handles, labels = axes[0].get_legend_handles_labels()

    axes[1].legend(
        handles,
        labels,
        loc="lower right",
        ncols=2,
        frameon=True,
        facecolor=SURFACE,
        edgecolor="none",
    )

    _finish(
        fig,
        path,
        "Comparação de calibração e desempenho entre os sistemas (IC 95%)",
    )


# ---------------------------------------------------------------------
# Figure 3: Paired bootstrap distribution of delta ECE
# ---------------------------------------------------------------------

def delta_bootstrap_figure(
    path: Path,
    paired: dict,
    n_bins: int,
    n_bootstrap: int,
    seed: int,
) -> None:
    """
    Distribution of paired bootstrap delta ECE.

    ΔECE = ECE_jev - ECE_llm.

    Negative values favor LLM.
    H1 requires the entire 95% CI to be below zero.
    """

    langs = list(paired)

    fig, axes = plt.subplots(
        1,
        len(langs),
        figsize=(4.8 * len(langs), 4.0),
        squeeze=False,
        sharey=True,
    )

    for col, lang in enumerate(langs):

        ax = axes[0][col]

        records = [
            {
                "jev": a["confidence"],
                "llm": b["confidence"],
                "correct_jev": a["correct"],
                "correct_llm": b["correct"],
            }
            for a, b in zip(
                paired[lang]["jev"],
                paired[lang]["llm"],
            )
        ]

        samples = bootstrap_delta_samples(
            records,
            n_bins,
            n_bootstrap,
            seed,
        )

        observed = (
            _ece(paired[lang]["jev"], n_bins)
            - _ece(paired[lang]["llm"], n_bins)
        )

        lo, hi = np.percentile(
            samples,
            [2.5, 97.5],
        )

        # Confidence interval.
        ax.axvspan(
            lo,
            hi,
            color=GRID,
            alpha=0.9,
            zorder=0,
            label=f"IC 95% [{_fmt(lo)}; {_fmt(hi)}]",
        )

        ax.hist(
            samples,
            bins=40,
            color=COLORS["jev"],
            alpha=0.7,
            edgecolor=SURFACE,
            linewidth=0.5,
        )

        # Observed difference.
        ax.axvline(
            observed,
            color=TEXT2,
            linewidth=1.5,
            linestyle="--",
            label=f"Δ observado = {_fmt(observed)}",
        )

        # No difference.
        ax.axvline(
            0,
            color=TEXT,
            linewidth=1.5,
            label="Δ = 0",
        )

        _panel_title(
            ax,
            lang,
            len(records),
        )

        ax.legend(
            loc="upper left",
            frameon=True,
            facecolor=SURFACE,
            edgecolor="none",
        )

        ax.set_xlabel(
            "Δ ECE = ECE(Jev) − ECE(LLM)"
        )

        ax.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(5)
        )

        ax.set_xlim(
            min(samples.min(), 0) - 0.01,
            max(samples.max(), 0) + 0.03,
        )

        ax.grid(axis="y", visible=False)

        heights = np.histogram(
            samples,
            bins=40,
        )[0]

        ax.set_ylim(
            0,
            max(heights.max() * 1.35, 1),
        )

    axes[0][0].set_ylabel(
        "Número de reamostragens"
    )

    _finish(
        fig,
        path,
        "Teste da hipótese H1: diferença pareada de ECE entre Jev e LLM",
    )


# ---------------------------------------------------------------------
# Figure 4: Confidence distribution by correctness
# ---------------------------------------------------------------------

def confidence_distribution_figure(
    path: Path,
    paired: dict,
) -> None:
    """
    Confidence histograms split by correct vs incorrect answers.

    Rows: systems.
    Columns: languages.

    High-confidence region is highlighted.
    """

    langs = list(paired)

    fig, axes = plt.subplots(
        len(SYSTEMS),
        len(langs),
        figsize=(5.2 * len(langs), 5.8),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    edges = np.linspace(0, 1, 21)

    for col, lang in enumerate(langs):

        for row_i, system in enumerate(SYSTEMS):

            ax = axes[row_i][col]

            rows = paired[lang][system]

            right = [
                r["confidence"]
                for r in rows
                if r["correct"]
            ]

            wrong = [
                r["confidence"]
                for r in rows
                if not r["correct"]
            ]

            # Highlight confidence >= 0.9.
            ax.axvspan(
                HIGH_CONFIDENCE,
                1,
                color=WRONG,
                alpha=0.07,
                zorder=0,
            )

            ax.hist(
                [right, wrong],
                bins=edges,
                stacked=True,
                color=[CORRECT, WRONG],
                edgecolor=SURFACE,
                linewidth=0.8,
                label=[
                    "Respostas corretas",
                    "Respostas incorretas",
                ],
            )

            high_errors = sum(
                c >= HIGH_CONFIDENCE
                for c in wrong
            )

            ax.axvline(
                HIGH_CONFIDENCE,
                color=WRONG,
                linestyle="--",
                linewidth=1,
                alpha=0.8,
            )

            ax.text(
                0.97,
                0.95,
                f"Erros com confiança ≥ 0,9: {high_errors}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color=WRONG,
                fontweight="bold",
            )

            ax.set_title(
                f"{SYSTEM_LABELS[system]} · "
                f"{LANGUAGE_LABELS.get(lang, lang)}",
                loc="left",
                fontweight="bold",
            )

            if row_i == len(SYSTEMS) - 1:
                ax.set_xlabel(
                    "Confiança reportada"
                )

            if col == 0:
                ax.set_ylabel(
                    "Número de respostas"
                )

    # Headroom so the "Erros com confiança ≥ 0,9" label never sits on a bar
    # (axes share y, so one call applies to every panel).
    axes[0][0].set_ylim(0, axes[0][0].get_ylim()[1] * 1.22)

    handles, labels = axes[0][0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    _finish(
        fig,
        path,
        "Distribuição da confiança segundo a correção das respostas",
        bottom_in=0.3,
    )


# ---------------------------------------------------------------------
# Figure 5: Selective prediction / Risk-Coverage
# ---------------------------------------------------------------------

def selective_curve(
    rows: list[dict],
) -> tuple[list[float], list[float], list[float]]:
    """
    Return coverage, accuracy and threshold.

    Only answers with confidence >= threshold are accepted.

    Ties are kept together.
    """

    conf = np.array(
        [r["confidence"] for r in rows],
        dtype=float,
    )

    hit = np.array(
        [r["correct"] for r in rows],
        dtype=float,
    )

    coverage = []
    accuracy = []
    thresholds = []

    for threshold in sorted(
        set(conf.tolist()),
        reverse=True,
    ):

        mask = conf >= threshold

        coverage.append(
            float(mask.mean())
        )

        accuracy.append(
            float(hit[mask].mean())
        )

        thresholds.append(threshold)

    return coverage, accuracy, thresholds


def selective_figure(
    path: Path,
    paired: dict,
) -> None:
    """
    Exploratory risk-coverage visualization.

    Shows the accuracy of automatically accepted answers
    as a function of the fraction of accepted answers.
    """

    langs = list(paired)

    fig, axes = plt.subplots(
        1,
        len(langs),
        figsize=(5.2 * len(langs), 4.2),
        squeeze=False,
        sharey=True,
    )

    for col, lang in enumerate(langs):

        ax = axes[0][col]

        for system in SYSTEMS:

            rows = paired[lang][system]

            cov, acc, _ = selective_curve(rows)

            baseline = float(
                np.mean([
                    r["correct"]
                    for r in rows
                ])
            )

            ax.step(
                cov,
                acc,
                where="post",
                color=COLORS[system],
                linewidth=2,
                alpha=0.9,
                label=SYSTEM_LABELS[system],
            )

            ax.scatter(
                cov,
                acc,
                s=20,
                color=COLORS[system],
                edgecolor=SURFACE,
                linewidth=0.7,
                zorder=3,
            )

            # Accuracy at full coverage.
            ax.axhline(
                baseline,
                color=COLORS[system],
                linestyle=":",
                linewidth=1,
                alpha=0.7,
            )

        ax.set(
            xlim=(0, 1.02),
            ylim=(0, 1.02),
            xlabel="Cobertura: proporção de respostas aceitas",
        )

        ax.xaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(
                1.0,
                decimals=0,
            )
        )

        ax.yaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(
                1.0,
                decimals=0,
            )
        )

        _panel_title(
            ax,
            lang,
            len(paired[lang]["jev"]),
        )

        ax.legend(
            loc="lower left",
            frameon=True,
            facecolor=SURFACE,
            edgecolor="none",
        )

        ax.grid(alpha=0.7)

    axes[0][0].set_ylabel(
        "Acurácia das respostas aceitas"
    )

    _finish(
        fig,
        path,
        "Risk-Coverage: impacto da seleção de respostas pela confiança",
    )


# ---------------------------------------------------------------------
# Confusion analysis
# ---------------------------------------------------------------------

def confusion_pairs(
    paired: dict,
    language: str,
) -> list[tuple[str, str, dict[str, int], dict[str, int]]]:
    """
    Return:
        (true_label, predicted_label,
         error counts per system,
         high-confidence error counts per system)

    Sorted by total error count.
    """

    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    high: dict[tuple[str, str], Counter] = defaultdict(Counter)

    for system in SYSTEMS:

        for r in paired[language][system]:

            if not r["correct"]:

                key = (
                    r["true_label_name"],
                    r["predicted_label_name"],
                )

                counts[key][system] += 1

                if r["confidence"] >= HIGH_CONFIDENCE:
                    high[key][system] += 1

    ordered = sorted(
        counts.items(),
        key=lambda kv: (
            -sum(kv[1].values()),
            kv[0],
        ),
    )

    return [
        (
            true_label,
            predicted_label,
            dict(count),
            dict(high[(true_label, predicted_label)]),
        )
        for (true_label, predicted_label), count in ordered
    ]


# ---------------------------------------------------------------------
# Figure 6: Most frequent confusions
# ---------------------------------------------------------------------

def confusions_figure(
    path: Path,
    paired: dict,
    top: int = 10,
) -> None:
    """
    Most frequent true -> predicted errors.

    Both systems are displayed side by side.
    """

    langs = list(paired)

    fig, axes = plt.subplots(
        1,
        len(langs),
        figsize=(
            6.5 * len(langs),
            2.0 + 0.48 * top,
        ),
        squeeze=False,
    )

    for col, lang in enumerate(langs):

        ax = axes[0][col]

        pairs = confusion_pairs(
            paired,
            lang,
        )[:top]

        labels = [
            f"{true_label}  →  {predicted_label}"
            for true_label, predicted_label, _, _ in pairs
        ]

        y = np.arange(len(pairs))
        height = 0.34

        for k, system in enumerate(SYSTEMS):

            values = [
                count.get(system, 0)
                for _, _, count, _ in pairs
            ]

            positions = (
                y + (k - 0.5) * height
            )

            bars = ax.barh(
                positions,
                values,
                height=height,
                color=COLORS[system],
                edgecolor=SURFACE,
                linewidth=0.8,
                label=SYSTEM_LABELS[system],
            )

            _annotate_bar(
                ax,
                bars,
                values,
            )

        ax.set_yticks(
            y,
            labels,
            fontsize=8,
        )

        ax.invert_yaxis()

        ax.set_xlabel(
            "Número de erros"
        )

        _panel_title(
            ax,
            lang,
        )

        ax.grid(axis="y", visible=False)

        ax.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True)
        )

        max_value = max(
            [
                max(count.values())
                for _, _, count, _ in pairs
            ] + [1]
        )

        ax.set_xlim(
            0,
            max_value * 1.22,
        )

        ax.legend(
            loc="lower right",
            frameon=True,
            facecolor=SURFACE,
            edgecolor="none",
        )

    _finish(
        fig,
        path,
        "Principais confusões entre intenções: verdadeira → prevista",
    )


# ---------------------------------------------------------------------
# CSV helpers and tables
# ---------------------------------------------------------------------

def _write_csv(
    path: Path,
    header: list[str],
    rows: list[list[Any]],
) -> None:

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.writer(
            file,
            delimiter=";",
        )

        writer.writerow(header)
        writer.writerows(rows)


def write_tables(
    table_dir: Path,
    records: list[dict],
    paired: dict,
    n_bins: int,
    texts: dict[tuple[str, str], str],
) -> None:

    errors = Counter(
        (r["language"], r["system"])
        for r in records
        if r.get("error") is not None
    )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    summary = []

    for lang in paired:

        for system in SYSTEMS:

            rows = paired[lang][system]

            conf = [
                r["confidence"]
                for r in rows
            ]

            hit = [
                bool(r["correct"])
                for r in rows
            ]

            ci = bootstrap_system_ci(
                conf,
                hit,
                n_bins,
            )

            acc = float(np.mean(hit))
            mean_conf = float(np.mean(conf))

            summary.append([
                lang,
                system,
                rows[0].get("model_id", ""),
                len(rows),
                errors[(lang, system)],

                round(_ece(rows, n_bins), 4),
                round(ci["ece_ci95"][0], 4),
                round(ci["ece_ci95"][1], 4),

                round(acc, 4),
                round(ci["accuracy_ci95"][0], 4),
                round(ci["accuracy_ci95"][1], 4),

                round(mean_conf, 4),
                round(mean_conf - acc, 4),

                sum(
                    1
                    for c, ok in zip(conf, hit)
                    if not ok and c >= HIGH_CONFIDENCE
                ),

                len({
                    round(c, 4)
                    for c in conf
                }),
            ])

    _write_csv(
        table_dir / "summary.csv",
        [
            "language",
            "system",
            "model_id",
            "n_paired",
            "n_failed_calls",
            "ece",
            "ece_ci95_low",
            "ece_ci95_high",
            "accuracy",
            "accuracy_ci95_low",
            "accuracy_ci95_high",
            "mean_confidence",
            "overconfidence",
            f"errors_conf_ge_{HIGH_CONFIDENCE}",
            "distinct_confidence_values",
        ],
        summary,
    )

    # -------------------------------------------------
    # Reliability bins
    # -------------------------------------------------

    bins_rows = []

    for lang in paired:

        for system in SYSTEMS:

            for b in _bins(
                paired[lang][system],
                n_bins,
            ):

                bins_rows.append([
                    lang,
                    system,
                    round(b["lo"], 4),
                    round(b["hi"], 4),
                    b["count"],

                    (
                        None
                        if b["conf"] is None
                        else round(b["conf"], 4)
                    ),

                    (
                        None
                        if b["acc"] is None
                        else round(b["acc"], 4)
                    ),
                ])

    _write_csv(
        table_dir / "reliability_bins.csv",
        [
            "language",
            "system",
            "bin_low",
            "bin_high",
            "count",
            "mean_confidence",
            "accuracy",
        ],
        bins_rows,
    )

    # -------------------------------------------------
    # High-confidence errors
    # -------------------------------------------------

    high_rows = []

    for lang in paired:

        for system in SYSTEMS:

            for r in sorted(
                paired[lang][system],
                key=lambda r: -r["confidence"],
            ):

                if (
                    not r["correct"]
                    and r["confidence"] >= HIGH_CONFIDENCE
                ):

                    high_rows.append([
                        lang,
                        system,
                        r["example_id"],
                        texts.get(
                            (r["example_id"], lang),
                            "",
                        ),
                        r["true_label_name"],
                        r["predicted_label_name"],
                        round(r["confidence"], 4),
                    ])

    _write_csv(
        table_dir / "high_confidence_errors.csv",
        [
            "language",
            "system",
            "example_id",
            "text",
            "true_label",
            "predicted_label",
            "confidence",
        ],
        high_rows,
    )

    # -------------------------------------------------
    # Confusion pairs
    # -------------------------------------------------

    conf_rows = []

    for lang in paired:

        for true_label, predicted_label, count, high in confusion_pairs(
            paired,
            lang,
        ):

            conf_rows.append([
                lang,
                true_label,
                predicted_label,

                count.get("jev", 0),
                count.get("llm", 0),

                high.get("jev", 0),
                high.get("llm", 0),
            ])

    _write_csv(
        table_dir / "confusion_pairs.csv",
        [
            "language",
            "true_label",
            "predicted_label",
            "errors_jev",
            "errors_llm",
            f"errors_jev_conf_ge_{HIGH_CONFIDENCE}",
            f"errors_llm_conf_ge_{HIGH_CONFIDENCE}",
        ],
        conf_rows,
    )

    # -------------------------------------------------
    # Language flips
    # -------------------------------------------------

    if {"en", "pt_br"} <= set(paired):

        flip_rows = []

        for system in SYSTEMS:

            en = {
                r["example_id"]: r["correct"]
                for r in paired["en"][system]
            }

            pt = {
                r["example_id"]: r["correct"]
                for r in paired["pt_br"][system]
            }

            ids = sorted(
                set(en) & set(pt)
            )

            flip_rows.append([
                system,
                len(ids),

                sum(
                    en[i] and pt[i]
                    for i in ids
                ),

                sum(
                    en[i] and not pt[i]
                    for i in ids
                ),

                sum(
                    not en[i] and pt[i]
                    for i in ids
                ),

                sum(
                    not en[i] and not pt[i]
                    for i in ids
                ),
            ])

        _write_csv(
            table_dir / "language_flips.csv",
            [
                "system",
                "n",
                "both_correct",
                "en_correct_pt_wrong",
                "en_wrong_pt_correct",
                "both_wrong",
            ],
            flip_rows,
        )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def make_report_outputs(
    output_dir: Path,
    records: list[dict],
    n_bins: int,
    examples: list[dict] | None = None,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> list[Path]:
    """
    Write every figure (figures/*.png) and table (tables/*.csv)
    for a run and return the paths written.
    """

    _style()

    paired = paired_by_language(records)

    if not paired:
        raise ValueError(
            "No paired (both systems valid) predictions to plot."
        )

    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Retrieve example texts
    # -------------------------------------------------

    texts = {}

    for ex in examples or []:

        texts[(ex["example_id"], "en")] = ex.get(
            "text_en",
            "",
        )

        texts[(ex["example_id"], "pt_br")] = ex.get(
            "text_pt_br",
            "",
        )

    # -------------------------------------------------
    # Generate figures
    # -------------------------------------------------

    reliability_figure(
        fig_dir / "reliability.png",
        paired,
        n_bins,
    )

    summary_figure(
        fig_dir / "ece_accuracy_ci.png",
        paired,
        n_bins,
    )

    delta_bootstrap_figure(
        fig_dir / "delta_ece_bootstrap.png",
        paired,
        n_bins,
        n_bootstrap,
        seed,
    )

    confidence_distribution_figure(
        fig_dir / "confidence_correct_vs_wrong.png",
        paired,
    )

    selective_figure(
        fig_dir / "risk_coverage.png",
        paired,
    )

    confusions_figure(
        fig_dir / "top_confusions.png",
        paired,
    )

    # -------------------------------------------------
    # Generate tables
    # -------------------------------------------------

    write_tables(
        table_dir,
        records,
        paired,
        n_bins,
        texts,
    )

    return (
        sorted(fig_dir.glob("*.png"))
        + sorted(table_dir.glob("*.csv"))
    )
