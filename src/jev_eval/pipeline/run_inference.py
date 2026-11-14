import hashlib
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
from jev_eval.pipeline.metadata import append_run_metadata, collect_run_metadata
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
    """Only the systems the config asks for, so an LLM-only run needs no TypeSafe key."""
    systems = config.systems if config else ["jev", "llm"]
    llm = (config.llm if config else {}) or {}
    jev = (config.jev if config else {}) or {}
    factories = {
        "jev": (lambda: MockJevClient()) if mock else (lambda: JevClient(model=jev.get("model"))),
        "llm": (lambda: MockLlmClient()) if mock else (lambda: LlmClient(
            model=llm.get("model"), base_url=llm.get("base_url"),
            api_key_env=llm.get("api_key_env", "OPENAI_API_KEY"), api_key=llm.get("api_key"),
            num_ctx=llm.get("num_ctx"))),
    }
    return {system: factories[system]() for system in systems}


def prompt_for(dataset: str, condition: Condition, system: str, config: EvalConfig) -> TaskPrompt:
    """Jev always gets the main template; the LLM gets `llm.prompt` (an ablation variant) if set."""
    variant = (config.llm or {}).get("prompt") if system == "llm" else None
    return build_task_prompt(dataset, condition, variant)


def _key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return row["dataset"], row["example_id"], row["condition"], row["system"]


def input_hash(message: str, criteria: dict[str, str], model_id: str, prompt: TaskPrompt) -> str:
    """Fingerprint of everything that determines an answer; a change means the row must be redone."""
    payload = json.dumps({"message": message, "criteria": criteria, "model": model_id,
                          "prompt": [prompt.version, prompt.instructions, prompt.llm_template,
                                     prompt.content_language_name]},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
    failed ones are retried. If a kept record was produced from different inputs (edited
    translation, other options, model or prompt), the run stops instead of mixing them.
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
    prompts = {(spec.name, condition.name, system): prompt_for(spec.name, condition, system, config)
               for spec in config.datasets for condition in conditions for system in config.systems}
    clients = build_clients(mock, config)
    expected = {(adapter.name, ex.example_id, c.name, system): input_hash(
                    ex.text[c.content_language], criteria[c.instruction_language],
                    getattr(clients[system], "model_id", ""), prompts[(adapter.name, c.name, system)])
                for adapter, examples, criteria in loaded for ex in examples for c in conditions
                for system in config.systems}

    output: Path = config.paths["raw_predictions"]
    existing = [] if fresh else read_jsonl(output)
    if any("condition" not in row for row in existing):
        raise ValueError(f"{output} was written by an older version (no 'dataset'/'condition' fields); "
                         "rerun with --fresh.")
    done = [row for row in existing if row.get("error") is None]
    stale = [row for row in done if _key(row) in expected and row.get("input_hash") != expected[_key(row)]]
    if stale:
        raise ValueError(
            f"{len(stale)} existing records in {output} were produced from different inputs or by an older "
            f"pipeline without input_hash (e.g. {[_key(r) for r in stale[:3]]}). Point the config to a new "
            "output directory (or use --fresh) instead of mixing them."
        )
    write_jsonl(output, done)  # drop stale error rows so they get retried
    append_run_metadata(output.parent / "run_metadata.json", collect_run_metadata(config, mock))
    done_keys = {_key(row) for row in done}
    records = list(done)

    with output.open("a", encoding="utf-8") as file:
        for adapter, examples, criteria in loaded:
            for example in examples:
                for condition in conditions:
                    message = example.text[condition.content_language]
                    options = criteria[condition.instruction_language]
                    for system in config.systems:
                        key = (adapter.name, example.example_id, condition.name, system)
                        if key in done_keys:
                            continue
                        client, prompt = clients[system], prompts[(adapter.name, condition.name, system)]
                        started = time.perf_counter()
                        try:
                            result = _classify_with_retry(client, message, options, prompt)
                            record = PredictionRecord(
                                example.example_id, adapter.name, condition.name, system, example.label,
                                result.predicted_label, result.confidence,
                                result.predicted_label == example.label, to_jsonable(result.raw_response),
                                result.model_id, result.prompt_version, None,
                                result.confidence_sdk, result.served_model, result.usage, expected[key],
                                round(time.perf_counter() - started, 3),
                            )
                        except Exception as error:
                            # InvalidResponse carries the server's raw output; keep it for auditing.
                            record = PredictionRecord(
                                example.example_id, adapter.name, condition.name, system, example.label,
                                "", 0.0, False, to_jsonable(getattr(error, "raw_response", None) or {}),
                                getattr(client, "model_id", ""), prompt.version,
                                f"{type(error).__name__}: {error}",
                                None, getattr(error, "served_model", None), getattr(error, "usage", None),
                                expected[key], round(time.perf_counter() - started, 3),
                            )
                        row = asdict(record)
                        file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        file.flush()
                        records.append(row)
    return records
