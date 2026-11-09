"""Report figures and tables for one run (raw_predictions.jsonl -> figures/ + tables/).

All figures use the *paired* subset (examples where both systems returned a valid answer),
the same items behind the headline ECE and the bootstrap, so the systems are always
compared on identical inputs. Failed calls are listed in tables/summary.csv.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jev_eval.eval.metrics import bootstrap_delta_samples, bootstrap_system_ci, expected_calibration_error  # noqa: E402

SYSTEMS = ("jev", "llm")
SYSTEM_LABELS = {"jev": "Jev", "llm": "LLM"}
LANGUAGE_LABELS = {"pt_br": "PT-BR (principal)", "en": "EN (exploratório)"}
LANGUAGE_ORDER = ("pt_br", "en")
HIGH_CONFIDENCE = 0.9

# Validated palette (dataviz validator, light surface): systems = categorical slots 1-2;
# correct/incorrect = violet/red (CVD ΔE 24). Text and chrome use neutral inks.
COLORS = {"jev": "#2a78d6", "llm": "#eb6834"}
CORRECT, WRONG = "#6250d6", "#e34948"
SURFACE, TEXT, TEXT2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.edgecolor": NEUTRAL, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "text.color": TEXT, "axes.titlecolor": TEXT, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 2,
    })


def _save(fig, path: Path) -> None:
    """Comma decimal separator on numeric axes (pt-BR), then save."""
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(comma)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def paired_by_language(records: list[dict[str, Any]]) -> dict[str, dict[str, list[dict]]]:
    """{language: {system: rows}} restricted to example_ids valid for both systems, same order."""
    valid: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in records:
        if row.get("error") is None:
            valid[row["language"]][row["system"]][row["example_id"]] = row
    out: dict[str, dict[str, list[dict]]] = {}
    for language in [lang for lang in LANGUAGE_ORDER if lang in valid] + sorted(set(valid) - set(LANGUAGE_ORDER)):
        ids = sorted(set.intersection(*(set(valid[language].get(s, {})) for s in SYSTEMS)))
        if ids:
            out[language] = {s: [valid[language][s][i] for i in ids] for s in SYSTEMS}
    return out


def _bins(rows: list[dict], n_bins: int) -> list[dict]:
    conf = np.array([r["confidence"] for r in rows], dtype=float)
    hit = np.array([r["correct"] for r in rows], dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        mask = (conf >= edges[i]) & ((conf <= edges[i + 1]) if i == n_bins - 1 else (conf < edges[i + 1]))
        out.append({"lo": edges[i], "hi": edges[i + 1], "count": int(mask.sum()),
                    "conf": float(conf[mask].mean()) if mask.any() else None,
                    "acc": float(hit[mask].mean()) if mask.any() else None})
    return out


def _ece(rows: list[dict], n_bins: int) -> float:
    return expected_calibration_error([r["confidence"] for r in rows], [r["correct"] for r in rows], n_bins)


# ---------------------------------------------------------------- figures

def reliability_figure(path: Path, paired: dict, n_bins: int) -> None:
    """Reliability diagram (marker area = bin size) over a confidence histogram, one column per language."""
    langs = list(paired)
    fig, axes = plt.subplots(2, len(langs), figsize=(4.2 * len(langs), 5.4), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08}, sharex=True)
    width = 1 / n_bins
    for col, lang in enumerate(langs):
        top, bottom = axes[0][col], axes[1][col]
        top.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=NEUTRAL, zorder=1)
        n = len(paired[lang]["jev"])
        for k, system in enumerate(SYSTEMS):
            rows = paired[lang][system]
            bins = [b for b in _bins(rows, n_bins) if b["count"]]
            xs, ys = [b["conf"] for b in bins], [b["acc"] for b in bins]
            sizes = [18 + 220 * b["count"] / n for b in bins]
            top.plot(xs, ys, color=COLORS[system], zorder=2, alpha=0.85)
            top.scatter(xs, ys, s=sizes, color=COLORS[system], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                        label=f"{SYSTEM_LABELS[system]}  (ECE {_fmt(_ece(rows, n_bins))})")
            counts = [b["count"] for b in _bins(rows, n_bins)]
            centers = np.linspace(0, 1, n_bins + 1)[:-1] + width / 2
            offset = (k - 0.5) * width * 0.42
            bottom.bar(centers + offset, counts, width=width * 0.4, color=COLORS[system], edgecolor=SURFACE,
                       linewidth=1)
        top.set(xlim=(0, 1), ylim=(0, 1.02), ylabel="Acurácia observada" if col == 0 else None,
                title=f"{LANGUAGE_LABELS.get(lang, lang)} · n = {n}")
        legend = top.legend(loc="upper left", frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.92)
        for handle in legend.legend_handles:
            handle.set_sizes([40])
        bottom.set(xlim=(0, 1), xlabel="Confiança reportada", ylabel="Nº de exemplos" if col == 0 else None)
    fig.text(0.01, 0.005, f"{n_bins} bins uniformes; diagonal tracejada = calibração perfeita; área do marcador ∝ nº de exemplos no bin.",
             color=TEXT2, fontsize=7)
    _save(fig, path)


def summary_figure(path: Path, paired: dict, n_bins: int) -> None:
    """ECE and accuracy per system and language with 95% bootstrap CIs (two panels, one axis each)."""
    langs = list(paired)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 0.9 + 0.75 * len(langs)), sharey=True)
    for panel, (metric, title) in enumerate((("ece", "ECE (menor é melhor)"), ("accuracy", "Acurácia (contexto)"))):
        ax = axes[panel]
        for row_i, lang in enumerate(langs):
            for k, system in enumerate(SYSTEMS):
                rows = paired[lang][system]
                conf, hit = [r["confidence"] for r in rows], [r["correct"] for r in rows]
                value = _ece(rows, n_bins) if metric == "ece" else float(np.mean(hit))
                lo, hi = bootstrap_system_ci(conf, hit, n_bins, n_bootstrap=2000)[f"{metric}_ci95"]
                y = row_i + (k - 0.5) * 0.32
                ax.plot([lo, hi], [y, y], color=COLORS[system], linewidth=2, solid_capstyle="round")
                ax.scatter([value], [y], s=46, color=COLORS[system], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                           label=SYSTEM_LABELS[system] if row_i == 0 else None)
                ax.text(hi + 0.01, y, _fmt(value), va="center", fontsize=7.5, color=TEXT)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 0.45 if metric == "ece" else 1.08)
    axes[0].set_yticks(range(len(langs)), [LANGUAGE_LABELS.get(lang, lang) for lang in langs])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower left", ncols=2)
    fig.text(0.01, -0.02, "Pontos: estimativa; barras: IC 95% bootstrap (2000 reamostragens).",
             color=TEXT2, fontsize=7)
    fig.tight_layout()
    _save(fig, path)


def delta_bootstrap_figure(path: Path, paired: dict, n_bins: int, n_bootstrap: int, seed: int) -> None:
    """Distribution of the paired bootstrap ΔECE = ECE_jev − ECE_llm; H1 needs the CI entirely below 0."""
    langs = list(paired)
    fig, axes = plt.subplots(1, len(langs), figsize=(3.9 * len(langs), 2.8), squeeze=False, sharey=True)
    for col, lang in enumerate(langs):
        ax = axes[0][col]
        records = [{"jev": a["confidence"], "llm": b["confidence"], "correct_jev": a["correct"],
                    "correct_llm": b["correct"]} for a, b in zip(paired[lang]["jev"], paired[lang]["llm"])]
        samples = bootstrap_delta_samples(records, n_bins, n_bootstrap, seed)
        observed = _ece(paired[lang]["jev"], n_bins) - _ece(paired[lang]["llm"], n_bins)
        lo, hi = np.percentile(samples, [2.5, 97.5])
        ax.axvspan(lo, hi, color=GRID, zorder=0)
        ax.hist(samples, bins=40, color=COLORS["jev"], alpha=0.75, edgecolor=SURFACE, linewidth=0.5)
        ax.axvline(0, color=TEXT, linewidth=1)
        ax.axvline(observed, color=TEXT2, linewidth=1.2, linestyle="--")
        verdict = "IC exclui 0" if hi < 0 or lo > 0 else "IC inclui 0"
        ax.set_title(f"{LANGUAGE_LABELS.get(lang, lang)}", loc="left")
        ax.text(0.02, 0.97, f"Δ = {_fmt(observed)}\nIC 95% [{_fmt(lo)}; {_fmt(hi)}]\n{verdict}",
                transform=ax.transAxes, va="top", fontsize=7.5, color=TEXT)
        ax.set_xlabel("Δ ECE = Jev − LLM  (negativo favorece Jev)")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_ylabel("Reamostragens")
    fig.tight_layout()
    _save(fig, path)


def confidence_distribution_figure(path: Path, paired: dict) -> None:
    """Confidence histograms split by correct vs incorrect: rows = system, columns = language."""
    langs = list(paired)
    fig, axes = plt.subplots(len(SYSTEMS), len(langs), figsize=(3.9 * len(langs), 4.6), squeeze=False,
                             sharex=True, sharey=True)
    edges = np.linspace(0, 1, 21)
    for col, lang in enumerate(langs):
        for row_i, system in enumerate(SYSTEMS):
            ax = axes[row_i][col]
            rows = paired[lang][system]
            right = [r["confidence"] for r in rows if r["correct"]]
            wrong = [r["confidence"] for r in rows if not r["correct"]]
            ax.hist([right, wrong], bins=edges, stacked=True, color=[CORRECT, WRONG], edgecolor=SURFACE,
                    linewidth=0.8, label=["acerto", "erro"])
            high = sum(c >= HIGH_CONFIDENCE for c in wrong)
            ax.text(0.03, 0.95, f"{SYSTEM_LABELS[system]}\nerros com conf. ≥ {_fmt(HIGH_CONFIDENCE, 1)}: "
                                f"{high}/{len(wrong)}", transform=ax.transAxes, va="top", fontsize=7.5, color=TEXT)
            ax.grid(axis="x", visible=False)
            if row_i == 0:
                ax.set_title(LANGUAGE_LABELS.get(lang, lang), loc="left")
            if row_i == len(SYSTEMS) - 1:
                ax.set_xlabel("Confiança reportada")
            if col == 0:
                ax.set_ylabel("Nº de exemplos")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(handles, labels, loc="upper right", ncols=2)
    _save(fig, path)


def selective_curve(rows: list[dict]) -> tuple[list[float], list[float], list[float]]:
    """(coverage, accuracy, threshold) when accepting answers with confidence >= each distinct threshold.

    Ties are kept together, so a system with few distinct confidence values has few operating points.
    """
    conf = np.array([r["confidence"] for r in rows], dtype=float)
    hit = np.array([r["correct"] for r in rows], dtype=float)
    coverage, accuracy, thresholds = [], [], []
    for t in sorted(set(conf.tolist()), reverse=True):
        mask = conf >= t
        coverage.append(float(mask.mean()))
        accuracy.append(float(hit[mask].mean()))
        thresholds.append(t)
    return coverage, accuracy, thresholds


def selective_figure(path: Path, paired: dict) -> None:
    """Exploratory: accuracy of the auto-accepted answers vs share of answers accepted (risk-coverage)."""
    langs = list(paired)
    fig, axes = plt.subplots(1, len(langs), figsize=(3.9 * len(langs), 3.2), squeeze=False, sharey=True)
    for col, lang in enumerate(langs):
        ax = axes[0][col]
        for system in SYSTEMS:
            cov, acc, _ = selective_curve(paired[lang][system])
            ax.step(cov, acc, where="post", color=COLORS[system], alpha=0.9)
            ax.scatter(cov, acc, s=16 if len(cov) > 12 else 34, color=COLORS[system], edgecolor=SURFACE,
                       linewidth=1, zorder=3, label=f"{SYSTEM_LABELS[system]} ({len(cov)} limiares)")
        ax.set(xlim=(0, 1.02), ylim=(0.5, 1.01), xlabel="Cobertura (fração aceita automaticamente)",
               title=LANGUAGE_LABELS.get(lang, lang))
        ax.title.set_ha("left")
        ax.title.set_position((0, 1))
        ax.legend(loc="lower left")
    axes[0][0].set_ylabel("Acurácia das respostas aceitas")
    fig.text(0.01, -0.03, "Exploratório: aceitar só respostas com confiança ≥ limiar. Cada ponto é um limiar.",
             color=TEXT2, fontsize=7)
    fig.tight_layout()
    _save(fig, path)


def confusion_pairs(paired: dict, language: str) -> list[tuple[str, str, dict[str, int], dict[str, int]]]:
    """[(true, predicted, counts per system, high-confidence counts per system)] sorted by total count."""
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    high: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for system in SYSTEMS:
        for r in paired[language][system]:
            if not r["correct"]:
                key = (r["true_label_name"], r["predicted_label_name"])
                counts[key][system] += 1
                if r["confidence"] >= HIGH_CONFIDENCE:
                    high[key][system] += 1
    ordered = sorted(counts.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
    return [(t, p, dict(c), dict(high[(t, p)])) for (t, p), c in ordered]


def confusions_figure(path: Path, paired: dict, top: int = 10) -> None:
    """Most frequent (true → predicted) errors, both systems side by side, one panel per language."""
    langs = list(paired)
    fig, axes = plt.subplots(1, len(langs), figsize=(5.0 * len(langs), 0.6 + 0.32 * top), squeeze=False)
    for col, lang in enumerate(langs):
        ax = axes[0][col]
        pairs = confusion_pairs(paired, lang)[:top]
        labels = [f"{t} → {p}" for t, p, _, _ in pairs]
        y = np.arange(len(pairs))
        for k, system in enumerate(SYSTEMS):
            values = [c.get(system, 0) for _, _, c, _ in pairs]
            ax.barh(y + (k - 0.5) * 0.38, values, height=0.36, color=COLORS[system], edgecolor=SURFACE,
                    linewidth=1, label=SYSTEM_LABELS[system])
        ax.set_yticks(y, labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Nº de erros")
        ax.set_title(LANGUAGE_LABELS.get(lang, lang), loc="left")
        ax.grid(axis="y", visible=False)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------- tables

def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)


def write_tables(table_dir: Path, records: list[dict], paired: dict, n_bins: int,
                 texts: dict[tuple[str, str], str]) -> None:
    errors = Counter((r["language"], r["system"]) for r in records if r.get("error") is not None)
    summary = []
    for lang in paired:
        for system in SYSTEMS:
            rows = paired[lang][system]
            conf, hit = [r["confidence"] for r in rows], [bool(r["correct"]) for r in rows]
            ci = bootstrap_system_ci(conf, hit, n_bins)
            acc, mean_conf = float(np.mean(hit)), float(np.mean(conf))
            summary.append([lang, system, rows[0].get("model_id", ""), len(rows), errors[(lang, system)],
                            round(_ece(rows, n_bins), 4), round(ci["ece_ci95"][0], 4), round(ci["ece_ci95"][1], 4),
                            round(acc, 4), round(ci["accuracy_ci95"][0], 4), round(ci["accuracy_ci95"][1], 4),
                            round(mean_conf, 4), round(mean_conf - acc, 4),
                            sum(1 for c, ok in zip(conf, hit) if not ok and c >= HIGH_CONFIDENCE),
                            len({round(c, 4) for c in conf})])
    _write_csv(table_dir / "summary.csv",
               ["language", "system", "model_id", "n_paired", "n_failed_calls", "ece", "ece_ci95_low",
                "ece_ci95_high", "accuracy", "accuracy_ci95_low", "accuracy_ci95_high", "mean_confidence",
                "overconfidence", f"errors_conf_ge_{HIGH_CONFIDENCE}", "distinct_confidence_values"], summary)

    bins_rows = []
    for lang in paired:
        for system in SYSTEMS:
            for b in _bins(paired[lang][system], n_bins):
                bins_rows.append([lang, system, round(b["lo"], 4), round(b["hi"], 4), b["count"],
                                  None if b["conf"] is None else round(b["conf"], 4),
                                  None if b["acc"] is None else round(b["acc"], 4)])
    _write_csv(table_dir / "reliability_bins.csv",
               ["language", "system", "bin_low", "bin_high", "count", "mean_confidence", "accuracy"], bins_rows)

    high_rows = []
    for lang in paired:
        for system in SYSTEMS:
            for r in sorted(paired[lang][system], key=lambda r: -r["confidence"]):
                if not r["correct"] and r["confidence"] >= HIGH_CONFIDENCE:
                    high_rows.append([lang, system, r["example_id"], texts.get((r["example_id"], lang), ""),
                                      r["true_label_name"], r["predicted_label_name"], round(r["confidence"], 4)])
    _write_csv(table_dir / "high_confidence_errors.csv",
               ["language", "system", "example_id", "text", "true_label", "predicted_label", "confidence"], high_rows)

    conf_rows = []
    for lang in paired:
        for t, p, c, h in confusion_pairs(paired, lang):
            conf_rows.append([lang, t, p, c.get("jev", 0), c.get("llm", 0), h.get("jev", 0), h.get("llm", 0)])
    _write_csv(table_dir / "confusion_pairs.csv",
               ["language", "true_label", "predicted_label", "errors_jev", "errors_llm",
                f"errors_jev_conf_ge_{HIGH_CONFIDENCE}", f"errors_llm_conf_ge_{HIGH_CONFIDENCE}"], conf_rows)

    if {"en", "pt_br"} <= set(paired):
        flip_rows = []
        for system in SYSTEMS:
            en = {r["example_id"]: r["correct"] for r in paired["en"][system]}
            pt = {r["example_id"]: r["correct"] for r in paired["pt_br"][system]}
            ids = sorted(set(en) & set(pt))
            flip_rows.append([system, len(ids), sum(en[i] and pt[i] for i in ids),
                              sum(en[i] and not pt[i] for i in ids), sum(not en[i] and pt[i] for i in ids),
                              sum(not en[i] and not pt[i] for i in ids)])
        _write_csv(table_dir / "language_flips.csv",
                   ["system", "n", "both_correct", "en_correct_pt_wrong", "en_wrong_pt_correct", "both_wrong"],
                   flip_rows)


# ---------------------------------------------------------------- entry point

def make_report_outputs(output_dir: Path, records: list[dict], n_bins: int,
                        examples: list[dict] | None = None, n_bootstrap: int = 5000, seed: int = 42) -> list[Path]:
    """Write every figure (figures/*.png) and table (tables/*.csv) for a run; returns the paths written."""
    _style()
    paired = paired_by_language(records)
    if not paired:
        raise ValueError("No paired (both systems valid) predictions to plot.")
    fig_dir, table_dir = output_dir / "figures", output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    texts = {}
    for ex in examples or []:
        texts[(ex["example_id"], "en")] = ex.get("text_en", "")
        texts[(ex["example_id"], "pt_br")] = ex.get("text_pt_br", "")

    reliability_figure(fig_dir / "reliability.png", paired, n_bins)
    summary_figure(fig_dir / "ece_accuracy_ci.png", paired, n_bins)
    delta_bootstrap_figure(fig_dir / "delta_ece_bootstrap.png", paired, n_bins, n_bootstrap, seed)
    confidence_distribution_figure(fig_dir / "confidence_correct_vs_wrong.png", paired)
    selective_figure(fig_dir / "risk_coverage.png", paired)
    confusions_figure(fig_dir / "top_confusions.png", paired)
    write_tables(table_dir, records, paired, n_bins, texts)
    return sorted(fig_dir.glob("*.png")) + sorted(table_dir.glob("*.csv"))
