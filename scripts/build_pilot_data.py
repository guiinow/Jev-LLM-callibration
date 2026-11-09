"""Build pilot inputs from the machine-drafted (NOT yet human-reviewed) translations.

PILOT ONLY: checks response formats, confidence extraction, error rates and run time
before the prompts/config are frozen. Results are not evidence for H1.

- data/pilot/sample_pilot.jsonl: data/sample.jsonl + machine-draft PT-BR messages
- data/pilot/intents_pilot.json: data/intents.json with the known machine-translation
  errors patched (the duplicate "taxa de câmbio" would otherwise block the run)
"""
import json
from pathlib import Path

PILOT_DESCRIPTION_FIXES = {
    "exchange_charge": "tarifa de câmbio",
    "top_up_by_card_charge": "tarifa de recarga com cartão",
    "top_up_reverted": "recarga estornada",
    "transfer_not_received_by_recipient": "transferência não recebida pelo favorecido",
}


def main() -> None:
    out_dir = Path("data/pilot")
    out_dir.mkdir(parents=True, exist_ok=True)
    drafts = {}
    for line in Path("data/translations_pt_br.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            drafts[row["example_id"]] = row
    rows = []
    for line in Path("data/sample.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            draft = drafts[row["example_id"]]
            row["text_pt_br"] = draft["text_pt_br"]
            row["translation_status"] = draft["translation_status"]
            rows.append(row)
    (out_dir / "sample_pilot.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    intents = json.loads(Path("data/intents.json").read_text(encoding="utf-8"))
    for row in intents:
        if row["label_name"] in PILOT_DESCRIPTION_FIXES:
            row["description_pt_br"] = PILOT_DESCRIPTION_FIXES[row["label_name"]]
            row["description_pt_br_status"] = "pilot_patch"
    (out_dir / "intents_pilot.json").write_text(json.dumps(intents, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    print(f"Wrote {len(rows)} pilot examples and {len(intents)} intents to {out_dir}/")


if __name__ == "__main__":
    main()
