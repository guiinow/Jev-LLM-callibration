"""Build CLINC150 inputs (EN + PT-BR) for the final run, without manual review.

Sources (pinned):
- tech4humans/clinc_oos_pt: test split of CLINC150 "plus" with the English text and a machine
  translation (`text_pt`) on the same row; intents are integers without names. CC BY 3.0.
- clinc/clinc_oos ("plus", test): the original, used only for the intent names and to prove the
  rows are aligned (same English text and same intent at every one of the 5,500 positions).

Decisions: only the 150 in-scope intents (the out-of-scope class has no natural description as an
option); a stratified sample of PER_INTENT examples per intent (seed 42); pairs with an empty PT
text, PT identical to EN, or a duplicated EN text are excluded before sampling and listed in the
manifest. PT-BR option descriptions (C3) come from data/clinc150/intent_names_pt.json, translated by
an LLM and not human-reviewed.

Writes data/clinc150/{sample_clinc150.jsonl, intents_clinc150.json, manifest.json}.
"""
import json
import random
from collections import Counter
from pathlib import Path

PT_DATASET, PT_REVISION = "tech4humans/clinc_oos_pt", "e45fe5519bdeac1b7d8de0e335d5c8573d627520"
EN_DATASET, EN_REVISION = "clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec"
OUT_DIR = Path("data/clinc150")
PER_INTENT = 3
SEED = 42
OOS = "oos"


def check_aligned(en_texts: list[str], en_labels: list[int], pt_rows: list[dict]) -> None:
    if len(en_texts) != len(pt_rows):
        raise ValueError(f"different sizes: {EN_DATASET} {len(en_texts)} vs {PT_DATASET} {len(pt_rows)}")
    bad = [i for i, (text, label, row) in enumerate(zip(en_texts, en_labels, pt_rows))
           if text != row["text"] or label != row["intent"]]
    if bad:
        raise ValueError(f"{len(bad)} rows differ in English text or intent (first: {bad[:5]}); not aligned")


def exclusions(pt_rows: list[dict], names: list[str]) -> dict[int, str]:
    """Row index -> reason, for rows that cannot be used as a clean EN/PT pair."""
    counts = Counter(r["text"].strip().lower() for r in pt_rows if names[r["intent"]] != OOS)
    out = {}
    for i, row in enumerate(pt_rows):
        if names[row["intent"]] == OOS:
            continue
        pt, en = (row["text_pt"] or "").strip(), row["text"].strip()
        if not pt:
            out[i] = "empty_pt"
        elif pt.lower() == en.lower():
            out[i] = "pt_identical_to_en"
        elif counts[en.lower()] > 1:
            out[i] = "duplicate_en_text"
    return out


def stratified_sample(pt_rows: list[dict], names: list[str], excluded: dict[int, str],
                      per_intent: int = PER_INTENT, seed: int = SEED) -> list[int]:
    """Row indexes: `per_intent` per in-scope intent (label-id order), then shuffled, as for Banking77."""
    grouped: dict[int, list[int]] = {}
    for i, row in enumerate(pt_rows):
        if names[row["intent"]] != OOS and i not in excluded:
            grouped.setdefault(row["intent"], []).append(i)
    rng = random.Random(seed)
    selected = []
    for label_id in sorted(grouped):
        rows = grouped[label_id]
        if len(rows) < per_intent:
            raise ValueError(f"intent {names[label_id]!r} has only {len(rows)} usable rows")
        rng.shuffle(rows)
        selected.extend(rows[:per_intent])
    rng.shuffle(selected)
    return selected


def build_intents(names: list[str], pt_names: dict[str, str]) -> list[dict]:
    in_scope = [n for n in names if n != OOS]
    if set(pt_names) != set(in_scope):
        missing, extra = sorted(set(in_scope) - set(pt_names)), sorted(set(pt_names) - set(in_scope))
        raise ValueError(f"PT descriptions must cover exactly the in-scope intents (missing {missing[:5]}, "
                         f"extra {extra[:5]})")
    intents = [{"label_id": names.index(n), "label_name": n, "description_en": n.replace("_", " "),
                "description_pt_br": pt_names[n], "description_pt_br_status": "llm_translated_not_reviewed"}
               for n in in_scope]
    for field in ("description_en", "description_pt_br"):
        values = [r[field].strip().lower() for r in intents]
        if len(set(values)) != len(values):
            raise ValueError(f"{field} values are not distinct")
    return intents


def build_sample(indexes: list[int], pt_rows: list[dict], names: list[str]) -> list[dict]:
    return [{"example_id": f"clinc-{n:04d}", "split": "test", "label_id": pt_rows[i]["intent"],
             "label_name": names[pt_rows[i]["intent"]], "text_en": pt_rows[i]["text"].strip(),
             "text_pt_br": pt_rows[i]["text_pt"].strip(), "translation_status": "machine_translated",
             "test_index": i} for n, i in enumerate(indexes)]


def main() -> None:
    from datasets import load_dataset

    en = load_dataset(EN_DATASET, "plus", split="test", revision=EN_REVISION)
    pt_rows = list(load_dataset(PT_DATASET, split="test", revision=PT_REVISION))
    names = en.features["intent"].names
    check_aligned(list(en["text"]), list(en["intent"]), pt_rows)
    excluded = exclusions(pt_rows, names)
    indexes = stratified_sample(pt_rows, names, excluded)
    pt_names = {k: v for k, v in json.loads((OUT_DIR / "intent_names_pt.json").read_text(encoding="utf-8")).items()
                if not k.startswith("_")}
    intents = build_intents(names, pt_names)
    sample = build_sample(indexes, pt_rows, names)

    jsonl = "".join(json.dumps({k: v for k, v in r.items() if k != "test_index"}, ensure_ascii=False) + "\n"
                    for r in sample)
    (OUT_DIR / "sample_clinc150.jsonl").write_text(jsonl, encoding="utf-8")
    (OUT_DIR / "intents_clinc150.json").write_text(json.dumps(intents, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
    manifest = {
        "pt": {"dataset": PT_DATASET, "revision": PT_REVISION, "license": "CC BY 3.0",
               "translation": "machine translation (dataset card names gemini-1.5-flash-002 and gpt-4o-mini), "
                              "not human-reviewed"},
        "en": {"dataset": EN_DATASET, "config": "plus", "revision": EN_REVISION, "license": "CC BY 3.0"},
        "checks": {"rows": len(pt_rows), "aligned_text_and_intent": True,
                   "in_scope_intents": len(intents), "out_of_scope_rows_left_out": sum(
                       names[r["intent"]] == OOS for r in pt_rows)},
        "excluded_before_sampling": [{"test_index": i, "reason": reason, "text_en": pt_rows[i]["text"],
                                     "text_pt": pt_rows[i]["text_pt"]} for i, reason in sorted(excluded.items())],
        "excluded_counts": dict(Counter(excluded.values())),
        "sample": {"per_intent": PER_INTENT, "seed": SEED, "n": len(sample),
                   "test_index_by_example_id": {r["example_id"]: r["test_index"] for r in sample}},
        "option_descriptions_pt": "data/clinc150/intent_names_pt.json (LLM-translated, not human-reviewed)",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("checks", "excluded_counts")}, indent=2),
          f"\nWrote {len(sample)} examples and {len(intents)} intents to {OUT_DIR}/")


if __name__ == "__main__":
    main()
