"""Import reviewed CSVs (from export_review_sheets.py) back into the data files.

- Messages -> data/translations_pt_br.jsonl (translation_status = human_reviewed)
- Descriptions -> data/intents.json (description_pt_br_status = human_reviewed)
Rows with an empty `status` are left as machine drafts and reported.
Afterwards, run scripts/merge_translations.py to copy the texts into data/sample.jsonl.

Usage:
    python scripts/import_review_sheets.py --review-dir data/review
"""
import argparse
import csv
import json
import sys
from pathlib import Path

VALID_STATUS = {"ok", "corrigido"}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        delimiter = ";" if sample.count(";") >= sample.count(",") else ","
        return list(csv.DictReader(file, delimiter=delimiter))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", default="data/review")
    parser.add_argument("--translations", default="data/translations_pt_br.jsonl")
    parser.add_argument("--intents", default="data/intents.json")
    args = parser.parse_args()
    review_dir = Path(args.review_dir)
    problems: list[str] = []

    translations_path = Path(args.translations)
    messages = [json.loads(line) for line in translations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["example_id"]: row for row in messages}
    reviewed = 0
    for path in sorted(review_dir.glob("mensagens_*.csv")):
        for row in read_csv(path):
            status = row.get("status", "").strip().lower()
            text = row.get("text_pt_br_final", "").strip()
            if status not in VALID_STATUS:
                problems.append(f"{path.name}: {row['example_id']} sem status válido ({status!r})")
                continue
            if not text:
                problems.append(f"{path.name}: {row['example_id']} com tradução final vazia")
                continue
            target = by_id[row["example_id"]]
            target.update({"text_pt_br": text, "translation_status": "human_reviewed",
                           "review_status": status, "reviewer": row.get("reviewer", ""),
                           "review_note": row.get("observacao", "").strip()})
            reviewed += 1
    translations_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in messages), encoding="utf-8")

    intents_path = Path(args.intents)
    intents = json.loads(intents_path.read_text(encoding="utf-8"))
    by_label = {row["label_name"]: row for row in intents}
    reviewed_desc = 0
    for path in sorted(review_dir.glob("descricoes_*.csv")):
        for row in read_csv(path):
            status = row.get("status", "").strip().lower()
            text = row.get("description_pt_br_final", "").strip()
            if status not in VALID_STATUS or not text:
                problems.append(f"{path.name}: {row['label_name']} sem status válido ou vazio")
                continue
            by_label[row["label_name"]].update({"description_pt_br": text, "description_pt_br_status": "human_reviewed",
                                                "description_reviewer": row.get("reviewer", "")})
            reviewed_desc += 1
    intents_path.write_text(json.dumps(intents, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Mensagens revisadas: {reviewed}/{len(messages)}; descrições revisadas: {reviewed_desc}/{len(intents)}")
    edited = sum(r.get("review_status") == "corrigido" for r in messages)
    print(f"Mensagens corrigidas pelos revisores: {edited}")
    if problems:
        print(f"{len(problems)} pendências:", file=sys.stderr)
        for problem in problems[:30]:
            print("  " + problem, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
