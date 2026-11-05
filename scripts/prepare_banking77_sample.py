import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from jev_eval.data.intent_labels import LABELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = load_dataset("PolyAI/banking77", split="test")
    grouped = {label_id: [] for label_id in range(len(LABELS))}
    for row in dataset:
        grouped[row["label"]].append(row)
    rng = random.Random(args.seed)
    per_label = max(1, args.sample_size // len(LABELS))
    selected = []
    for label_id, rows in grouped.items():
        rng.shuffle(rows)
        selected.extend(rows[:per_label])
    rng.shuffle(selected)
    selected = selected[:args.sample_size]
    output = Path("data/sample.jsonl")
    output.parent.mkdir(exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for index, row in enumerate(selected):
            file.write(json.dumps({
                "example_id": f"test-{index:04d}", "split": "test", "label_id": row["label"],
                "label_name": LABELS[row["label"]], "text_en": row["text"],
                "text_pt_br": row["text"], "translation_status": "pending_human_review",
            }, ensure_ascii=False) + "\n")
    Path("data/intents.json").write_text(json.dumps([
        {"label_id": i, "label_name": label, "description_en": label.replace("_", " "),
         "description_pt_br": ""}
        for i, label in enumerate(LABELS)
    ], indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

