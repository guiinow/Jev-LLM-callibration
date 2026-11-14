"""E1: LLM prompt ablation (top-k) against Jev and the main prompt, reusing the final run.

The ablation config runs only the LLM (systems: [llm], llm.prompt: topk) and points
`paths.baseline` at the final run's raw_predictions.jsonl. Its Jev rows join the analysis as
"jev" and its main-prompt LLM rows feed the comparison table, so Jev is never called again.
Every reused row must have been produced from exactly this config's message, options and the
condition's main prompt (input_hash).
"""
from collections import Counter
from typing import Any

from jev_eval.conditions import get_condition
from jev_eval.clients.llm_client import invalid_lower_guesses
from jev_eval.config import EvalConfig
from jev_eval.data.datasets import build_adapter
from jev_eval.data.io import read_jsonl
from jev_eval.eval.metrics import auroc, brier_top1, expected_calibration_error, paired_bootstrap_ece
from jev_eval.pipeline.run_inference import criteria_by_language, input_hash
from jev_eval.prompts.templates import build_task_prompt

VARIANTS = ("jev", "llm_main", "llm_topk")
COMPARISONS = (("llm_topk", "llm_main"), ("jev", "llm_topk"), ("jev", "llm_main"))


def load_baseline(config: EvalConfig) -> list[dict[str, Any]]:
    """Baseline rows for this config's datasets, examples and conditions; refuses rows made
    from other inputs and an incomplete baseline."""
    conditions = [get_condition(name) for name in config.conditions]
    expected: dict[tuple, str] = {}
    for spec in config.datasets:
        adapter = build_adapter(spec)
        criteria = criteria_by_language(adapter, conditions)
        for ex in adapter.load_examples(config.sample_size):
            for c in conditions:
                expected[(spec.name, ex.example_id, c.name)] = (
                    ex.text[c.content_language], criteria[c.instruction_language], build_task_prompt(spec.name, c))
    rows = [r for r in read_jsonl(config.paths["baseline"])
            if (r.get("dataset"), r["example_id"], r.get("condition")) in expected]
    stale = []
    for r in rows:
        message, options, prompt = expected[(r["dataset"], r["example_id"], r["condition"])]
        if r.get("input_hash") != input_hash(message, options, r.get("model_id", ""), prompt):
            stale.append(r)
    if stale:
        raise ValueError(f"{len(stale)} baseline rows in {config.paths['baseline']} were produced from other "
                         f"messages/options/prompts than this config "
                         f"(e.g. {[(r['example_id'], r['condition'], r['system']) for r in stale[:3]]})")
    have = {(r["dataset"], r["example_id"], r["condition"], r["system"]) for r in rows}
    missing = [key + (s,) for key in expected for s in ("jev", "llm") if key + (s,) not in have]
    if missing:
        raise ValueError(f"baseline is incomplete: {len(missing)} (dataset, example, condition, system) rows "
                         f"missing (e.g. {missing[:3]}); finish the baseline run first")
    return rows


def with_baseline_jev(records: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """This run's LLM rows plus the baseline's Jev rows, ready for analyze/plot."""
    return [r for r in records if r["system"] != "jev"] + [r for r in baseline if r["system"] == "jev"]


def _scores(rows: list[dict[str, Any]], n_bins: int) -> dict[str, Any]:
    conf, hit = [r["confidence"] for r in rows], [bool(r["correct"]) for r in rows]
    area = auroc(conf, hit)
    return {"ece": round(expected_calibration_error(conf, hit, n_bins), 4), "accuracy": round(sum(hit) / len(hit), 4),
            "mean_confidence": round(sum(conf) / len(conf), 4), "brier_top1": round(brier_top1(conf, hit), 4),
            "auroc": None if area is None else round(area, 4)}


def ablation(records: list[dict[str, Any]], baseline: list[dict[str, Any]], n_bins: int,
             n_bootstrap: int = 5000, seed: int = 42, labels: set[str] | None = None) -> dict[str, Any]:
    """For one dataset, per condition: each variant's scores on the items all three answered
    validly, failed calls per variant, and paired-bootstrap ΔECE (a - b) for each comparison.
    With `labels`, also how many accepted top-k answers had a lower guess outside the options."""
    tagged = ([(r, "llm_topk") for r in records if r["system"] == "llm"]
              + [(r, "jev" if r["system"] == "jev" else "llm_main") for r in baseline])
    valid: dict[str, dict[str, dict[str, dict]]] = {}
    failed: Counter = Counter()
    for row, variant in tagged:
        if row.get("error") is None:
            valid.setdefault(row["condition"], {}).setdefault(variant, {})[row["example_id"]] = row
        else:
            failed[(row["condition"], variant)] += 1
    out: dict[str, Any] = {
        "n_bins": n_bins,
        "topk_prompt_versions": dict(Counter(r.get("prompt_version") for r in records if r["system"] == "llm")),
        "baseline_models": dict(Counter(f"{r['system']}: {r.get('model_id')} / {r.get('prompt_version')}"
                                        for r in baseline)),
        "conditions": {},
    }
    for condition, by_variant in sorted(valid.items()):
        ids = sorted(set.intersection(*(set(by_variant.get(v, {})) for v in VARIANTS)))
        if not ids:
            continue
        result = {"n_common": len(ids),
                  "topk_answers_with_invalid_lower_guess": None if labels is None else sum(
                      invalid_lower_guesses(r["raw_response"], labels) > 0
                      for r in by_variant.get("llm_topk", {}).values() if isinstance(r.get("raw_response"), str)),
                  "variants": {v: {**_scores([by_variant[v][i] for i in ids], n_bins),
                                   "n_failed_calls": failed[(condition, v)]} for v in VARIANTS},
                  "comparisons": {}}
        for a, b in COMPARISONS:
            pairs = [{"jev": by_variant[a][i]["confidence"], "llm": by_variant[b][i]["confidence"],
                      "correct_jev": by_variant[a][i]["correct"], "correct_llm": by_variant[b][i]["correct"]}
                     for i in ids]
            result["comparisons"][f"{a} - {b}"] = paired_bootstrap_ece(pairs, n_bins, n_bootstrap, seed)
        out["conditions"][condition] = result
    return out


def ablation_tables(result: dict[str, Any]) -> tuple[list[str], list[list[Any]], list[str], list[list[Any]]]:
    """(header, rows) for tables/ablation.csv and tables/ablation_deltas.csv."""
    score_keys = ["ece", "accuracy", "mean_confidence", "brier_top1", "auroc", "n_failed_calls"]
    rows, deltas = [], []
    for condition, res in result["conditions"].items():
        for variant, scores in res["variants"].items():
            rows.append([condition, variant, res["n_common"]] + [scores[k] for k in score_keys]
                        + [res.get("topk_answers_with_invalid_lower_guess") if variant == "llm_topk" else None])
        for name, d in res["comparisons"].items():
            deltas.append([condition, name, res["n_common"], round(d["delta_ece"], 4),
                           round(d["ci95_low"], 4), round(d["ci95_high"], 4)])
    return (["condition", "variant", "n_common"] + score_keys + ["answers_with_invented_lower_guess"], rows,
            ["condition", "comparison", "n_common", "delta_ece", "ci95_low", "ci95_high"], deltas)
