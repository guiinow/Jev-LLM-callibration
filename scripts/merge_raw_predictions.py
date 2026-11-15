"""Merge raw_predictions.jsonl files written on different machines (e.g. Qwen on a rented GPU while
Gemma runs on the Mac, both starting from a copy of outputs/final/raw_predictions.jsonl).

Rows are keyed by (dataset, example_id, condition, system). A key present in several files is kept
once if the copies agree; a valid answer wins over a failed call; two valid answers produced from
different inputs (input_hash) or with different predictions stop the merge. Nothing is overwritten
unless --output is given explicitly.

    python scripts/merge_raw_predictions.py outputs/final/raw_predictions.jsonl qwen_copy.jsonl \
        --output outputs/final/raw_predictions.merged.jsonl
"""
import argparse
import json
from pathlib import Path

KEY = ("dataset", "example_id", "condition", "system")


def merge(files: list[list[dict]]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for rows in files:
        for row in rows:
            key = tuple(row[k] for k in KEY)
            old = merged.get(key)
            if old is None or (old.get("error") is not None and row.get("error") is None):
                merged[key] = row
            elif old.get("error") is None and row.get("error") is None and (
                    old.get("input_hash") != row.get("input_hash")
                    or old.get("predicted_label_name") != row.get("predicted_label_name")
                    or old.get("confidence") != row.get("confidence")):
                raise ValueError(f"conflicting valid answers for {key}: rerun one side instead of merging")
    return list(merged.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = [[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
             for path in args.inputs]
    rows = merge(files)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{sum(map(len, files))} rows in, {len(rows)} rows out -> {args.output}")


if __name__ == "__main__":
    main()
