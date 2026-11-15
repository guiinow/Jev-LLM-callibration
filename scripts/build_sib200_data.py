"""Build SIB-200 inputs (EN + PT) for the final run: the whole test split (204 sentences).

Source (pinned): Davlan/sib200, configs eng_Latn and por_Latn (CC BY-SA 4.0). The sentences come
from FLORES-200, translated by professional translators (the Portuguese is por_Latn; the variety
is not marked as Brazilian). EN and PT are paired by `index_id`; the pairing is accepted only if
both configs have the same ids and the same category for every id. Pairs with an empty PT text or
PT identical to EN are excluded and listed in the manifest. Order is shuffled with seed 42 so that
a short run (first N rows) covers several topics.

Writes data/sib200/{sample_sib200.jsonl, intents_sib200.json, manifest.json}.
"""
import json
import random
from pathlib import Path

DATASET, REVISION = "Davlan/sib200", "38977a667f6fc264d5c26ec57a01e16db040b358"
OUT_DIR = Path("data/sib200")
SEED = 42

# The 7 SIB-200 topics; label names are the category with "/" replaced, descriptions are the topic.
TOPICS = {
    "science/technology": "ciência/tecnologia",
    "travel": "viagem",
    "politics": "política",
    "sports": "esportes",
    "health": "saúde",
    "entertainment": "entretenimento",
    "geography": "geografia",
}


def label_name(category: str) -> str:
    return category.replace("/", "_")


def pair_by_id(en_rows: list[dict], pt_rows: list[dict]) -> tuple[list[tuple[dict, dict]], list[dict]]:
    """(pairs, excluded). Refuses different id sets or a category mismatch for any id."""
    en, pt = {r["index_id"]: r for r in en_rows}, {r["index_id"]: r for r in pt_rows}
    if len(en) != len(en_rows) or len(pt) != len(pt_rows):
        raise ValueError("duplicate index_id within a config")
    if set(en) != set(pt):
        raise ValueError(f"different index_id sets: only EN {sorted(set(en) - set(pt))[:5]}, "
                         f"only PT {sorted(set(pt) - set(en))[:5]}")
    mismatch = [i for i in en if en[i]["category"] != pt[i]["category"]]
    if mismatch:
        raise ValueError(f"category differs between EN and PT for index_id {mismatch[:5]}")
    unknown = sorted({r["category"] for r in en_rows} - set(TOPICS))
    if unknown:
        raise ValueError(f"unknown categories {unknown}")
    pairs, excluded = [], []
    for i in sorted(en):
        e, p = en[i]["text"].strip(), (pt[i]["text"] or "").strip()
        reason = "empty_pt" if not p else "pt_identical_to_en" if p.lower() == e.lower() else None
        if reason:
            excluded.append({"index_id": i, "reason": reason, "text_en": e, "text_pt": p})
        else:
            pairs.append((en[i], pt[i]))
    return pairs, excluded


def build_sample(pairs: list[tuple[dict, dict]], seed: int = SEED) -> list[dict]:
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    topics = list(TOPICS)
    return [{"example_id": f"sib-{pairs[k][0]['index_id']:04d}", "split": "test",
             "label_id": topics.index(pairs[k][0]["category"]), "label_name": label_name(pairs[k][0]["category"]),
             "text_en": pairs[k][0]["text"].strip(), "text_pt_br": pairs[k][1]["text"].strip(),
             "translation_status": "human_translated"} for k in order]


def build_intents() -> list[dict]:
    return [{"label_id": i, "label_name": label_name(c), "description_en": c, "description_pt_br": pt,
             "description_pt_br_status": "translated_by_us"} for i, (c, pt) in enumerate(TOPICS.items())]


def main() -> None:
    from datasets import load_dataset

    en = list(load_dataset(DATASET, "eng_Latn", split="test", revision=REVISION))
    pt = list(load_dataset(DATASET, "por_Latn", split="test", revision=REVISION))
    pairs, excluded = pair_by_id(en, pt)
    sample = build_sample(pairs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sample_sib200.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample), encoding="utf-8")
    (OUT_DIR / "intents_sib200.json").write_text(json.dumps(build_intents(), indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
    manifest = {
        "source": {"dataset": DATASET, "configs": ["eng_Latn", "por_Latn"], "split": "test",
                   "revision": REVISION, "license": "CC BY-SA 4.0",
                   "translation": "FLORES-200 professional human translation (por_Latn; variety not marked)"},
        "checks": {"rows_en": len(en), "rows_pt": len(pt), "same_index_ids": True, "same_categories": True},
        "excluded": excluded,
        "sample": {"n": len(sample), "seed": SEED, "per_topic": {label_name(c): sum(
            r["label_name"] == label_name(c) for r in sample) for c in TOPICS}},
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    print(json.dumps(manifest["checks"] | {"excluded": len(excluded)} | manifest["sample"], indent=2),
          f"\nWrote {len(sample)} examples to {OUT_DIR}/")


if __name__ == "__main__":
    main()
