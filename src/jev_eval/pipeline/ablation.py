"""E1: LLM prompt ablation (top-k) against Jev and the v1 prompt, reusing the final run.

The ablation config runs only the LLM (systems: [llm], llm.prompt: topk) and points
`paths.baseline` at the final run's raw_predictions.jsonl. Its Jev rows join the analysis as
"jev" and its v1 LLM rows feed the comparison table, so Jev is never called again. Every reused
row must have been produced from exactly this config's message and options (input_hash).
"""
from collections import Counter
from types import SimpleNamespace
from typing import Any

from jev_eval.config import EvalConfig
from jev_eval.data.io import load_examples, read_jsonl
from jev_eval.eval.metrics import auroc, brier_top1, expected_calibration_error, paired_bootstrap_ece
from jev_eval.pipeline.run_inference import TEXT_FIELD, criteria_by_language, input_hash

VARIANTS = ("jev", "llm_v1", "llm_topk")
COMPARISONS = (("llm_topk", "llm_v1"), ("jev", "llm_topk"), ("jev", "llm_v1"))


def load_baseline(config: EvalConfig) -> list[dict[str, Any]]:
    """Baseline rows for this config's examples and languages; refuses rows made from other inputs."""
    examples = {ex.example_id: ex for ex in load_examples(config.paths["sample"], config.sample_size)}
    criteria = criteria_by_language(config)
    rows = [r for r in read_jsonl(config.paths["baseline"])
            if r["example_id"] in examples and r["language"] in config.languages]
    stale = [r for r in rows if r.get("input_hash") != input_hash(
        getattr(examples[r["example_id"]], TEXT_FIELD[r["language"]]), criteria[r["language"]],
        SimpleNamespace(model_id=r.get("model_id", ""), prompt_version=r.get("prompt_version", "")))]
    if stale:
        raise ValueError(f"{len(stale)} baseline rows in {config.paths['baseline']} were produced from other "
                         f"messages/options than this config (e.g. {[(r['example_id'], r['language']) for r in stale[:3]]})")
    have = {(r["example_id"], r["language"], r["system"]) for r in rows}
    missing = [(e, lang, s) for e in examples for lang in config.languages for s in ("jev", "llm")
               if (e, lang, s) not in have]
    if missing:
        raise ValueError(f"baseline is incomplete: {len(missing)} (example, language, system) rows missing "
                         f"(e.g. {missing[:3]}); finish the baseline run first")
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
             n_bootstrap: int = 5000, seed: int = 42) -> dict[str, Any]:
    """Per language: each variant's scores on the items all three answered validly, failed calls
    per variant, and paired-bootstrap ΔECE (a - b) for each comparison."""
    tagged = ([(r, "llm_topk") for r in records if r["system"] == "llm"]
              + [(r, "jev" if r["system"] == "jev" else "llm_v1") for r in baseline])
    valid: dict[str, dict[str, dict[str, dict]]] = {}
    failed: Counter = Counter()
    for row, variant in tagged:
        if row.get("error") is None:
            valid.setdefault(row["language"], {}).setdefault(variant, {})[row["example_id"]] = row
        else:
            failed[(row["language"], variant)] += 1
    out: dict[str, Any] = {
        "n_bins": n_bins,
        "baseline_models": dict(Counter(f"{r['system']}: {r.get('model_id')} / {r.get('prompt_version')}"
                                        for r in baseline)),
        "languages": {},
    }
    for language, by_variant in valid.items():
        ids = sorted(set.intersection(*(set(by_variant.get(v, {})) for v in VARIANTS)))
        if not ids:
            continue
        result = {"n_common": len(ids),
                  "variants": {v: {**_scores([by_variant[v][i] for i in ids], n_bins),
                                   "n_failed_calls": failed[(language, v)]} for v in VARIANTS},
                  "comparisons": {}}
        for a, b in COMPARISONS:
            pairs = [{"jev": by_variant[a][i]["confidence"], "llm": by_variant[b][i]["confidence"],
                      "correct_jev": by_variant[a][i]["correct"], "correct_llm": by_variant[b][i]["correct"]}
                     for i in ids]
            result["comparisons"][f"{a} - {b}"] = paired_bootstrap_ece(pairs, n_bins, n_bootstrap, seed)
        out["languages"][language] = result
    return out


def ablation_tables(result: dict[str, Any]) -> tuple[list[str], list[list[Any]], list[str], list[list[Any]]]:
    """(header, rows) for tables/ablation.csv and tables/ablation_deltas.csv."""
    score_keys = ["ece", "accuracy", "mean_confidence", "brier_top1", "auroc", "n_failed_calls"]
    rows, deltas = [], []
    for language, res in result["languages"].items():
        for variant, scores in res["variants"].items():
            rows.append([language, variant, res["n_common"]] + [scores[k] for k in score_keys])
        for name, d in res["comparisons"].items():
            deltas.append([language, name, res["n_common"], round(d["delta_ece"], 4),
                           round(d["ci95_low"], 4), round(d["ci95_high"], 4)])
    return (["language", "variant", "n_common"] + score_keys, rows,
            ["language", "comparison", "n_common", "delta_ece", "ci95_low", "ci95_high"], deltas)
