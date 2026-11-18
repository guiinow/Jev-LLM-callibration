"""Junta à execução final as respostas de um sistema rodado à parte, com um novo nome de sistema.

Exemplo (Gemini 3.1 Flash-Lite, rodado como "llm" em outputs/gemini_flash_lite/):

    python scripts/import_system_rows.py outputs/gemini_flash_lite/raw_predictions.jsonl \
        --from-system llm --to-system gemini_verb

Antes de gravar, confere pelo `input_hash` que cada linha foi produzida com a mensagem, as opções e o
prompt principal de `--config` (padrão: configs/final.yaml), e que cobre todos os (base, exemplo,
condição) da config. Recusa se o destino já tiver linhas desse sistema. As linhas mantêm o
`model_id` e o `input_hash` originais; só o campo `system` muda.
"""
import argparse
import json
import sys
from pathlib import Path

from jev_eval.conditions import get_condition
from jev_eval.config import load_config
from jev_eval.data.datasets import build_adapter
from jev_eval.data.io import read_jsonl
from jev_eval.pipeline.run_inference import criteria_by_language, input_hash
from jev_eval.prompts.templates import build_task_prompt
from jev_eval.systems import system_parts


def expected_inputs(config) -> dict[tuple[str, str, str], tuple]:
    conditions = [get_condition(name) for name in config.conditions]
    expected = {}
    for spec in config.datasets:
        adapter = build_adapter(spec)
        criteria = criteria_by_language(adapter, conditions)
        for ex in adapter.load_examples(config.sample_size):
            for c in conditions:
                expected[(spec.name, ex.example_id, c.name)] = (
                    ex.text[c.content_language], criteria[c.instruction_language], build_task_prompt(spec.name, c))
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("--from-system", required=True)
    parser.add_argument("--to-system", required=True)
    parser.add_argument("--config", default="configs/final.yaml")
    args = parser.parse_args()

    system_parts(args.to_system)  # valid name ("<model>_verb" / "<model>_prob")
    config = load_config(args.config)
    target = config.paths["raw_predictions"]
    expected = expected_inputs(config)
    rows = [r for r in read_jsonl(args.source) if r["system"] == args.from_system]

    keys = [(r["dataset"], r["example_id"], r["condition"]) for r in rows]
    unknown = [k for k in keys if k not in expected]
    if unknown or len(set(keys)) != len(keys):
        raise SystemExit(f"{len(unknown)} rows outside {args.config}, {len(keys) - len(set(keys))} duplicates")
    missing = set(expected) - set(keys)
    if missing:
        raise SystemExit(f"{len(missing)} (dataset, example, condition) missing, e.g. {sorted(missing)[:3]}")
    stale = [k for r, k in zip(rows, keys)
             if input_hash(expected[k][0], expected[k][1], r["model_id"], expected[k][2]) != r["input_hash"]]
    if stale:
        raise SystemExit(f"{len(stale)} rows were produced from other messages/options/prompts, e.g. {stale[:3]}")
    if any(r["system"] == args.to_system for r in read_jsonl(target)):
        raise SystemExit(f"{target} already has rows of {args.to_system!r}; refusing to mix runs")

    with target.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps({**row, "system": args.to_system}, ensure_ascii=False) + "\n")
    errors = sum(1 for r in rows if r.get("error"))
    print(f"appended {len(rows)} rows as {args.to_system!r} to {target} ({errors} with error)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
