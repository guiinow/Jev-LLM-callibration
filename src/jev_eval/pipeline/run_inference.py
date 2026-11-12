import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jev_eval.clients import ClassifierClient, JevClient, LlmClient, MockJevClient, MockLlmClient
from jev_eval.conditions import Condition, get_condition
from jev_eval.config import EvalConfig
from jev_eval.data.datasets import DatasetAdapter, Example, TaskType, build_adapter
from jev_eval.data.io import read_jsonl, to_jsonable, write_jsonl
from jev_eval.prompts.templates import TaskPrompt, build_task_prompt
from jev_eval.types import PredictionRecord

MAX_ATTEMPTS = 3


def criteria_by_language(adapter: DatasetAdapter, conditions: list[Condition]) -> dict[str, dict[str, str]]:
    """Label -> description for each instruction language the conditions need."""
    criteria: dict[str, dict[str, str]] = {}
    for language in sorted({condition.instruction_language for condition in conditions}):
        options = adapter.options(language)
        missing = [label for label, description in options.items() if not str(description).strip()]
        if missing:
            raise ValueError(
                f"{len(missing)} options of '{adapter.name}' have an empty 'description_{language}' "
                f"(e.g. {missing[:3]}); translate them before running instruction language '{language}'."
            )
        seen: dict[str, str] = {}
        for label, description in options.items():
            text = description.strip().lower()
            if text in seen:
                raise ValueError(
                    f"Options '{seen[text]}' and '{label}' share the same 'description_{language}' "
                    f"({description!r}); descriptions must be distinct."
                )
            seen[text] = label
        criteria[language] = options
    return criteria


def validate_examples(examples: list[Example], conditions: list[Condition]) -> None:
    """Fail fast if any message is missing for a requested content language."""
    for language in sorted({condition.content_language for condition in conditions}):
        empty = [ex.example_id for ex in examples if not ex.text.get(language, "").strip()]
        if empty:
            raise ValueError(f"{len(empty)} examples have an empty 'text_{language}' (e.g. {empty[:3]}).")
        if language == "pt_br":
            untranslated = [ex.example_id for ex in examples if ex.text["pt_br"].strip() == ex.text["en"].strip()]
            if untranslated:
                raise ValueError(
                    f"{len(untranslated)} examples have text_pt_br identical to text_en "
                    f"(e.g. {untranslated[:3]}); they look untranslated."
                )
            unreviewed = [ex.example_id for ex in examples
                          if ex.translation_status not in (None, "human_reviewed")]
            if unreviewed:
                print(f"WARNING: {len(unreviewed)} PT-BR translations are not 'human_reviewed' "
                      f"(e.g. {unreviewed[:3]}).", file=sys.stderr)


def build_clients(mock: bool, config: EvalConfig | None = None) -> dict[str, ClassifierClient]:
    if mock:
        return {"jev": MockJevClient(), "llm": MockLlmClient()}
    llm = (config.llm if config else {}) or {}
    jev = (config.jev if config else {}) or {}
    return {
        "jev": JevClient(model=jev.get("model")),
        "llm": LlmClient(model=llm.get("model"), base_url=llm.get("base_url"),
                         api_key_env=llm.get("api_key_env", "OPENAI_API_KEY"), api_key=llm.get("api_key")),
    }


def _key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return row["dataset"], row["example_id"], row["condition"], row["system"]


def _classify_with_retry(client: ClassifierClient, message: str, criteria: dict[str, str], prompt: TaskPrompt):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return client.classify(message, criteria, prompt)
        except ValueError:
            raise  # invalid/unparseable answer: deterministic at temperature 0, retrying won't help
        except Exception:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(2 ** attempt)


def run_inference(config: EvalConfig, mock: bool = False, fresh: bool = False) -> list[dict[str, Any]]:
    """Run every (dataset, example, condition, system) once, appending each record to disk as it finishes.

    Re-running resumes: successful records already in the output are kept and skipped;
    failed ones are retried.
    """
    conditions = [get_condition(name) for name in config.conditions]
    loaded: list[tuple[DatasetAdapter, list[Example], dict[str, dict[str, str]]]] = []
    for spec in config.datasets:
        if spec.task is not TaskType.CHOICE:
            raise NotImplementedError(f"Task type '{spec.task.value}' ({spec.name}) is not supported yet.")
        adapter = build_adapter(spec)
        examples = adapter.load_examples(config.sample_size)
        validate_examples(examples, conditions)
        loaded.append((adapter, examples, criteria_by_language(adapter, conditions)))
    prompts = {(spec.name, condition.name): build_task_prompt(spec.name, condition)
               for spec in config.datasets for condition in conditions}
    clients = build_clients(mock, config)

    output: Path = config.paths["raw_predictions"]
    existing = [] if fresh else read_jsonl(output)
    if any("condition" not in row for row in existing):
        raise ValueError(f"{output} was written by an older version (no 'dataset'/'condition' fields); "
                         "rerun with --fresh.")
    done = [row for row in existing if row.get("error") is None]
    write_jsonl(output, done)  # drop stale error rows so they get retried
    done_keys = {_key(row) for row in done}
    records = list(done)

    with output.open("a", encoding="utf-8") as file:
        for adapter, examples, criteria in loaded:
            for example in examples:
                for condition in conditions:
                    message = example.text[condition.content_language]
                    options = criteria[condition.instruction_language]
                    prompt = prompts[(adapter.name, condition.name)]
                    for system in config.systems:
                        if (adapter.name, example.example_id, condition.name, system) in done_keys:
                            continue
                        client = clients[system]
                        try:
                            result = _classify_with_retry(client, message, options, prompt)
                            record = PredictionRecord(
                                example.example_id, adapter.name, condition.name, system, example.label,
                                result.predicted_label, result.confidence,
                                result.predicted_label == example.label, to_jsonable(result.raw_response),
                                result.model_id, result.prompt_version, None,
                            )
                        except Exception as error:
                            record = PredictionRecord(
                                example.example_id, adapter.name, condition.name, system, example.label,
                                "", 0.0, False, {}, getattr(client, "model_id", ""), "",
                                f"{type(error).__name__}: {error}",
                            )
                        row = asdict(record)
                        file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        file.flush()
                        records.append(row)
    return records
