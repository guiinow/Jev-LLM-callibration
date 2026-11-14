import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="data/sample.jsonl")
    parser.add_argument("--translations", default="data/translations_pt_br.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    sample_path, translations_path = Path(args.sample), Path(args.translations)
    translations = {}
    with translations_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            translations[row["example_id"]] = row
    output = Path(args.output) if args.output else sample_path
    rows, missing = [], []
    with sample_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            translation = translations.get(row["example_id"])
            if translation:
                row["text_pt_br"] = translation["text_pt_br"]
                row["translation_status"] = translation.get("translation_status", "pending_human_review")
            if not str(row.get("text_pt_br", "")).strip():
                missing.append(row["example_id"])
            rows.append(row)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Merged translations into {output}: {len(rows) - len(missing)}/{len(rows)} translated")
    if missing:
        print(f"WARNING: {len(missing)} examples still have no PT-BR text (e.g. {missing[:5]})", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

