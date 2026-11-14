"""Build the final-run inputs from the public, human-reviewed PT-BR Banking77 (USP).

Source: c2d-usp/banking77-pt-br (Dimant et al., 2025, CC BY 4.0), aligned row by row with
PolyAI/banking77 (CC BY 4.0). The USP release has no English text and no ids, so this
script first checks that its test split has the same order as PolyAI's (same intent at
every one of the 3,080 positions, mapping PT names to ids with the dataset card's table)
and stops otherwise. Each of our 462 examples is then located in the original test split
by its English text, and the PT-BR sentence at that index is used.

Writes (data/pilot/ and the old data files are not touched):
- data/usp/sample_usp.jsonl   same schema as data/sample.jsonl, text_pt_br from USP
- data/usp/intents_usp.json   PT-BR descriptions = USP intent names ("_" -> " ")
- data/usp/alignment_usp.jsonl per example: USP test index, `alterado`, `modelo_tradutor`
- data/usp/provenance.json    dataset revisions and the alignment checks
"""
import json
import re
import statistics
import sys
from pathlib import Path

from jev_eval.data.intent_labels import LABELS

EN_DATASET, EN_REVISION = "PolyAI/banking77", "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"
PT_DATASET, PT_REVISION = "c2d-usp/banking77-pt-br", "85f51c4a36c69482fae32ec31afec784f9722a7e"
OUT_DIR = Path("data/usp")


def card_label_names(card: str) -> dict[str, int]:
    """PT intent name -> label id, from the '| id | name |' table in the dataset card."""
    names = {name: int(i) for i, name in re.findall(r"^\|\s*(\d+)\s*\|\s*(\S+)\s*\|\s*$", card, re.M)}
    if len(names) != 77 or sorted(names.values()) != list(range(77)):
        raise ValueError(f"expected 77 distinct intents in the dataset card table, got {len(names)}")
    return names


def drop_empty_tail(rows: list[dict]) -> list[dict]:
    """The USP test split ends with all-null rows; drop them, refuse nulls anywhere else."""
    end = len(rows)
    while end and rows[end - 1]["frase"] is None and rows[end - 1]["rotulo"] is None:
        end -= 1
    body = rows[:end]
    bad = [i for i, r in enumerate(body) if not (r["frase"] or "").strip() or r["rotulo"] is None]
    if bad:
        raise ValueError(f"USP rows with empty sentence or label before the tail: {bad[:5]}")
    return body


def check_same_order(en_labels: list[int], pt_rows: list[dict], pt_ids: dict[str, int]) -> None:
    if len(en_labels) != len(pt_rows):
        raise ValueError(f"different sizes: {EN_DATASET} {len(en_labels)} vs {PT_DATASET} {len(pt_rows)}")
    unknown = sorted({r["rotulo"] for r in pt_rows} - set(pt_ids))
    if unknown:
        raise ValueError(f"USP labels missing from the dataset card table: {unknown[:5]}")
    diff = [i for i, (a, r) in enumerate(zip(en_labels, pt_rows)) if pt_ids[r["rotulo"]] != a]
    if diff:
        raise ValueError(f"label order differs at {len(diff)} positions (first: {diff[:5]}); "
                         "the USP split cannot be aligned by index")


def build(sample, intents, en_texts, en_labels, pt_rows, pt_ids, label_names):
    """Pure part: returns (sample rows, intents, alignment rows). Raises on any mismatch."""
    index = {text: i for i, text in enumerate(en_texts)}
    if len(index) != len(en_texts):
        raise ValueError("duplicate English texts in the test split; matching by text is ambiguous")
    pt_name = {i: name for name, i in pt_ids.items()}
    out_sample, alignment = [], []
    for row in sample:
        i = index.get(row["text_en"])
        if i is None:
            raise ValueError(f"{row['example_id']}: English text not found in {EN_DATASET} test")
        pt = pt_rows[i]
        if not (en_labels[i] == pt_ids[pt["rotulo"]] == row["label_id"]
                and label_names[row["label_id"]] == row["label_name"]):
            raise ValueError(f"{row['example_id']}: label mismatch at test index {i}")
        text_pt = pt["frase"].strip()
        if text_pt == row["text_en"].strip():
            raise ValueError(f"{row['example_id']}: USP sentence identical to the English one")
        out_sample.append({**row, "text_pt_br": text_pt, "translation_status": "human_reviewed"})
        alignment.append({"example_id": row["example_id"], "test_index": i, "rotulo": pt["rotulo"],
                          "alterado": pt["alterado"], "modelo_tradutor": pt["modelo_tradutor"]})
    out_intents = []
    for row in intents:
        clean = {k: v for k, v in row.items() if k != "description_mt_model"}
        clean["description_pt_br"] = pt_name[row["label_id"]].replace("_", " ")
        clean["description_pt_br_status"] = PT_DATASET
        out_intents.append(clean)
    descriptions = [r["description_pt_br"].strip().lower() for r in out_intents]
    if len(set(descriptions)) != len(descriptions):
        raise ValueError("USP intent names are not distinct after replacing '_' with ' '")
    return out_sample, out_intents, alignment


def main() -> None:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    en = load_dataset(EN_DATASET, split="test", revision=EN_REVISION)
    pt = load_dataset(PT_DATASET, split="test", revision=PT_REVISION)
    card = Path(hf_hub_download(PT_DATASET, "README.md", repo_type="dataset", revision=PT_REVISION)).read_text()
    pt_ids = card_label_names(card)
    pt_rows = drop_empty_tail(list(pt))
    en_texts, en_labels = list(en["text"]), list(en["label"])
    check_same_order(en_labels, pt_rows, pt_ids)
    # Beyond labels: sentence lengths must track each other item by item (an in-block shuffle would not).
    length_corr = statistics.correlation([len(t) for t in en_texts], [len(r["frase"]) for r in pt_rows])
    if length_corr < 0.9:
        sys.exit(f"EN/PT sentence lengths barely correlate ({length_corr:.3f}); alignment is doubtful")

    sample = [json.loads(line) for line in Path("data/sample.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    intents = json.loads(Path("data/intents.json").read_text(encoding="utf-8"))
    out_sample, out_intents, alignment = build(sample, intents, en_texts, en_labels, pt_rows, pt_ids, LABELS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = lambda rows: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)  # noqa: E731
    (OUT_DIR / "sample_usp.jsonl").write_text(jsonl(out_sample), encoding="utf-8")
    (OUT_DIR / "alignment_usp.jsonl").write_text(jsonl(alignment), encoding="utf-8")
    (OUT_DIR / "intents_usp.json").write_text(json.dumps(out_intents, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    provenance = {
        "en": {"dataset": EN_DATASET, "revision": EN_REVISION, "license": "CC BY 4.0", "test_rows": len(en)},
        "pt_br": {"dataset": PT_DATASET, "revision": PT_REVISION, "license": "CC BY 4.0",
                  "citation": "Dimant et al., 2025", "test_rows_raw": len(pt),
                  "test_rows_used": len(pt_rows), "dropped_empty_tail_rows": len(pt) - len(pt_rows)},
        "checks": {"label_order_identical": True, "length_correlation_en_pt": round(length_corr, 4),
                   "sample_matched": len(out_sample),
                   "sample_alterado": sum(a["alterado"] for a in alignment),
                   "descriptions_distinct": True},
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
