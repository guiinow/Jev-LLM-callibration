import json
from pathlib import Path
from typing import Any

from jev_eval.clients import ClassifierClient, JevClient, LlmClient, MockJevClient, MockLlmClient
from jev_eval.config import EvalConfig
from jev_eval.data.io import load_examples, read_jsonl, write_jsonl
from jev_eval.types import PredictionRecord


def _criteria(config: EvalConfig) -> dict[str, str]:
    return {row["label_name"]: row["description_en"] for row in json.loads(
        config.paths["intents"].read_text(encoding="utf-8")
    )}


def build_clients(mock: bool) -> dict[str, ClassifierClient]:
    if mock:
        return {"jev": MockJevClient(), "llm": MockLlmClient()}
    return {"jev": JevClient(), "llm": LlmClient()}


def run_inference(config: EvalConfig, mock: bool = False) -> list[dict[str, Any]]:
    examples = load_examples(config.paths["sample"], config.sample_size)
    criteria = _criteria(config)
    clients = build_clients(mock)
    records: list[dict[str, Any]] = []
    for example in examples:
        for language in config.languages:
            message = example.text_en if language == "en" else example.text_pt_br
            for system in config.systems:
                try:
                    result = clients[system].classify(message, criteria, language)
                    record = PredictionRecord(
                        example.example_id, language, system, example.label_name,
                        result.predicted_label, result.confidence,
                        result.predicted_label == example.label_name, result.raw_response,
                        result.model_id, "v1", None,
                    )
                except Exception as error:
                    record = PredictionRecord(
                        example.example_id, language, system, example.label_name,
                        "", 0.0, False, {}, "", "v1", str(error),
                    )
                records.append(record)
    write_jsonl(config.paths["raw_predictions"], records)
    return [record.__dict__ for record in records]

