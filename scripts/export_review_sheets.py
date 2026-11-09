"""Split machine-drafted PT-BR translations into one CSV per reviewer.

Each reviewer edits the `*_final` column (pre-filled with the draft) and sets `status`
to `ok` (draft kept) or `corrigido` (edited). Then run scripts/import_review_sheets.py.

CSV uses ';' and UTF-8 with BOM so it opens correctly in Excel (pt-BR) and Google Sheets.

Usage:
    python scripts/export_review_sheets.py --reviewers "Nome Um" "Nome Dois"
"""
import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

MESSAGE_FIELDS = ["example_id", "label_name", "intencao_pt_br", "text_en", "mt_draft",
                  "text_pt_br_final", "status", "observacao", "reviewer"]
DESCRIPTION_FIELDS = ["label_name", "description_en", "mt_draft", "description_pt_br_final",
                      "status", "observacao", "reviewer"]


def slug(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewers", nargs="+", required=True)
    parser.add_argument("--translations", default="data/translations_pt_br.jsonl")
    parser.add_argument("--intents", default="data/intents.json")
    parser.add_argument("--out-dir", default="data/review")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intents = json.loads(Path(args.intents).read_text(encoding="utf-8"))
    pt_label = {row["label_name"]: row.get("description_pt_br", "") for row in intents}
    messages = [json.loads(line) for line in Path(args.translations).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    messages.sort(key=lambda row: (row["label_name"], row["example_id"]))

    n = len(args.reviewers)
    # Group by intent so each reviewer sees whole intents (keeps terminology consistent).
    labels = sorted({row["label_name"] for row in messages})
    owner = {label: args.reviewers[i % n] for i, label in enumerate(labels)}
    for reviewer in args.reviewers:
        rows = [{
            "example_id": row["example_id"], "label_name": row["label_name"],
            "intencao_pt_br": pt_label.get(row["label_name"], ""), "text_en": row["text_en"],
            "mt_draft": row.get("mt_draft", row["text_pt_br"]), "text_pt_br_final": row["text_pt_br"],
            "status": "", "observacao": "", "reviewer": reviewer,
        } for row in messages if owner[row["label_name"]] == reviewer]
        desc_rows = [{
            "label_name": row["label_name"], "description_en": row["description_en"],
            "mt_draft": row.get("description_pt_br", ""), "description_pt_br_final": row.get("description_pt_br", ""),
            "status": "", "observacao": "", "reviewer": reviewer,
        } for row in intents if owner.get(row["label_name"]) == reviewer]
        write_csv(out_dir / f"mensagens_{slug(reviewer)}.csv", MESSAGE_FIELDS, rows)
        write_csv(out_dir / f"descricoes_{slug(reviewer)}.csv", DESCRIPTION_FIELDS, desc_rows)
        print(f"{reviewer}: {len(rows)} mensagens, {len(desc_rows)} descrições")
    print(f"Planilhas em {out_dir}/")


if __name__ == "__main__":
    main()
