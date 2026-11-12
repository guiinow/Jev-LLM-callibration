import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jev_eval.clients import ClassifierClient, JevClient, LlmClient, MockJevClient, MockLlmClient
from jev_eval.config import EvalConfig
from jev_eval.data.io import load_examples, read_jsonl, to_jsonable, write_jsonl
from jev_eval.pipeline.metadata import append_run_metadata, collect_run_metadata
from jev_eval.types import PredictionRecord, SampleExample

DESCRIPTION_FIELD = {"en": "description_en", "pt_br": "description_pt_br"}
TEXT_FIELD = {"en": "text_en", "pt_br": "text_pt_br"}
MAX_ATTEMPTS = 3


def criteria_by_language(config: EvalConfig) -> dict[str, dict[str, str]]:
    """Label -> description for each language; PT-BR runs must see PT-BR descriptions."""
    intents = json.loads(config.paths["intents"].read_text(encoding="utf-8"))
    criteria: dict[str, dict[str, str]] = {}
    for language in config.languages:
        field = DESCRIPTION_FIELD[language]
        missing = [row["label_name"] for row in intents if not str(row.get(field, "")).strip()]
        if missing:
            raise ValueError(
                f"{len(missing)} intents have an empty '{field}' in {config.paths['intents']} "
                f"(e.g. {missing[:3]}); translate them before running language '{language}'."
            )
        seen: dict[str, str] = {}
        for row in intents:
            text = row[field].strip().lower()
            if text in seen:
                raise ValueError(
                    f"Intents '{seen[text]}' and '{row['label_name']}' share the same '{field}' "
                    f"({row[field]!r}); descriptions must be distinct."
                )
            seen[text] = row["label_name"]
        criteria[language] = {row["label_name"]: row[field] for row in intents}
    return criteria


def validate_examples(examples: list[SampleExample], languages: list[str]) -> None:
    """Fail fast if any message is missing for a requested language."""
    for language in languages:
        field = TEXT_FIELD[language]
        empty = [ex.example_id for ex in examples if not getattr(ex, field).strip()]
        if empty:
            raise ValueError(f"{len(empty)} examples have an empty '{field}' (e.g. {empty[:3]}).")
        if language == "pt_br":
            untranslated = [ex.example_id for ex in examples if ex.text_pt_br.strip() == ex.text_en.strip()]
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
            model=llm.get("model"), base_url=llm.get("base_url"), api_key_env=llm.get("api_key_env", "OPENAI_API_KEY"),
            api_key=llm.get("api_key"), num_ctx=llm.get("num_ctx"))),
    }
    return {system: factories[system]() for system in systems}


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["example_id"], row["language"], row["system"]


def input_hash(message: str, criteria: dict[str, str], client: ClassifierClient) -> str:
    """Fingerprint of everything that determines an answer; a change means the row must be redone."""
    payload = json.dumps({"message": message, "criteria": criteria, "model": getattr(client, "model_id", ""),
                          "prompt_version": getattr(client, "prompt_version", "")},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _classify_with_retry(client: ClassifierClient, message: str, criteria: dict[str, str], language: str):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return client.classify(message, criteria, language)
        except ValueError:
            raise  # invalid/unparseable answer: deterministic at temperature 0, retrying won't help
        except Exception:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(2 ** attempt)


def run_inference(config: EvalConfig, mock: bool = False, fresh: bool = False) -> list[dict[str, Any]]:
    """Run every (example, language, system) once, appending each record to disk as it finishes.

    Re-running resumes: successful records already in the output are kept and skipped;
    failed ones are retried. If a kept record was produced from different inputs (edited
    translation, other options, model or prompt), the run stops instead of mixing them.
    """
    examples = load_examples(config.paths["sample"], config.sample_size)
    validate_examples(examples, config.languages)
    criteria = criteria_by_language(config)
    clients = build_clients(mock, config)

    output: Path = config.paths["raw_predictions"]
    expected = {(ex.example_id, language, system): input_hash(getattr(ex, TEXT_FIELD[language]),
                                                             criteria[language], clients[system])
                for ex in examples for language in config.languages for system in config.systems}
    done = [] if fresh else [row for row in read_jsonl(output) if row.get("error") is None]
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
        for example in examples:
            for language in config.languages:
                message = getattr(example, TEXT_FIELD[language])
                for system in config.systems:
                    if (example.example_id, language, system) in done_keys:
                        continue
                    client = clients[system]
                    fingerprint = expected[(example.example_id, language, system)]
                    try:
                        result = _classify_with_retry(client, message, criteria[language], language)
                        record = PredictionRecord(
                            example.example_id, language, system, example.label_name,
                            result.predicted_label, result.confidence,
                            result.predicted_label == example.label_name, to_jsonable(result.raw_response),
                            result.model_id, result.prompt_version, None,
                            result.confidence_sdk, result.served_model, result.usage, fingerprint,
                        )
                    except Exception as error:
                        # InvalidResponse carries the server's raw output; keep it for auditing.
                        record = PredictionRecord(
                            example.example_id, language, system, example.label_name,
                            "", 0.0, False, to_jsonable(getattr(error, "raw_response", None) or {}),
                            getattr(client, "model_id", ""), getattr(client, "prompt_version", ""),
                            f"{type(error).__name__}: {error}",
                            None, getattr(error, "served_model", None), getattr(error, "usage", None), fingerprint,
                        )
                    row = asdict(record)
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    file.flush()
                    records.append(row)
    return records
