import random
from pathlib import Path

import pytest

from jev_eval.eval.report_figures import (_num, condition_label, make_report_figures, q1_figure, q2_figure,
                                          q3_figure, system_colors, system_label)
from jev_eval.pipeline.analyze import analyze

ROOT = Path(__file__).resolve().parent.parent


def make_records(n=60, seed=2, conditions=("C1", "C2", "C3"), systems=("jev", "llm")):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        for condition, accuracy in zip(("C1", "C2", "C3"), (0.8, 0.7, 0.68)):
            if condition not in conditions:
                continue
            for system in systems:
                ok = rng.random() < accuracy
                confidence = round(rng.uniform(0.5, 0.99), 2) if system == "jev" else rng.choice([0.9, 0.95, 0.99])
                rows.append({"example_id": f"e{i}", "dataset": "banking77", "condition": condition, "system": system,
                             "confidence": confidence, "correct": ok, "predicted_label_name": "a" if ok else "b",
                             "error": None})
    return rows


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    rows = make_records()
    metrics = analyze(rows, 10, tmp_path_factory.mktemp("m") / "metrics.json", n_bootstrap=30)
    return metrics, rows


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_all_three_figures_are_written_as_png_and_pdf(tmp_path, data, lang):
    metrics, rows = data
    written = make_report_figures(tmp_path, metrics, rows, lang)
    assert sorted(p.name for p in written) == [
        "q1_h1.pdf", "q1_h1.png", "q2_en_pt.pdf", "q2_en_pt.png", "q3_selection.pdf", "q3_selection.png"]
    assert all(p.stat().st_size > 1000 for p in written)
    assert written[0].read_bytes()[:4] == b"\x89PNG" and written[1].read_bytes()[:4] == b"%PDF"


def test_unknown_language_is_rejected(tmp_path, data):
    with pytest.raises(ValueError, match="lang must be"):
        make_report_figures(tmp_path, *data, lang="fr")


def test_figures_say_what_is_missing_instead_of_drawing_nothing(tmp_path):
    only_c1 = make_records(30, conditions=("C1",))
    metrics = analyze(only_c1, 10, tmp_path / "m.json", n_bootstrap=20)
    q1_figure(metrics, tmp_path / "q1")  # C1 alone still has the Jev vs LLM contrast
    with pytest.raises(ValueError, match="Q2"):
        q2_figure(metrics, tmp_path / "q2")  # no PT condition: no EN->PT change
    assert q3_figure(only_c1, tmp_path / "q3")  # one panel is enough
    with pytest.raises(ValueError, match="Q3"):
        q3_figure([], tmp_path / "q3b")
    with pytest.raises(ValueError, match="Q1"):
        q1_figure({"conditions": {}}, tmp_path / "q1b")


def test_q2_handles_extra_systems(tmp_path, data):
    metrics, _ = data
    metrics = {**metrics, "language_shift": {**metrics["language_shift"],
                                            "gemma_prob": metrics["language_shift"]["jev"],
                                            "gemma_verb": metrics["language_shift"]["llm"]}}
    assert q2_figure(metrics, tmp_path / "q2")


def test_labels_and_formatting():
    assert system_label("jev") == "Jev" and system_label("llm") == "LLM"
    assert system_label("gemma_prob") == "Gemma (prob.)" and system_label("qwen_verb", "en") == "Qwen (verbalized)"
    assert system_label("weird") == "weird"
    assert condition_label("C2") == "C2\nPT/EN"
    assert _num(-0.0123, "pt", 3, sign=True) == "−0,012" and _num(0.5, "en", 2, sign=True) == "+0.50"
    colors = system_colors(["jev", "llm", "gemma_prob", "gemma_verb"])
    assert len(set(colors.values())) == 4


def test_report_command_needs_analyze_first_then_writes_figures(tmp_path, monkeypatch):
    from jev_eval import cli
    from jev_eval.config import load_config
    from jev_eval.pipeline import run_inference as ri
    config = load_config(ROOT / "configs" / "smoke.yaml")
    config.paths["raw_predictions"] = tmp_path / "raw.jsonl"
    records = ri.run_inference(config, mock=True)
    with pytest.raises(FileNotFoundError, match="run `analyze` first"):
        cli._report(config, records)
    analyze_ = cli.analyze
    monkeypatch.setattr(cli, "analyze", lambda *a, **k: analyze_(*a, n_bootstrap=20, **k))
    cli._analyze(config, records)
    cli._report(config, records, "en")
    assert (tmp_path / "banking77" / "report" / "en" / "q3_selection.pdf").exists()
