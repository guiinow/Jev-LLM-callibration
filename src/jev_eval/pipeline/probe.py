"""Pre-flight checks before a real run (a handful of calls, nothing written to outputs/).

LLM: sends the longest real prompt per language to Ollama's native API and reports the
exact prompt token count, so we know the prompt fits in num_ctx (Ollama truncates
silently). The model is unloaded before each call so a warm prompt cache cannot make the
reported count smaller than the real prompt.

Jev: one call, to confirm the served model version and that probabilities[choice] exists.
"""
from typing import Any

from jev_eval.config import EvalConfig
from jev_eval.data.io import load_examples
from jev_eval.pipeline.metadata import ollama_root
from jev_eval.pipeline.run_inference import TEXT_FIELD, criteria_by_language
from jev_eval.prompts.llm_prompt import build_prompt

# Smallest default num_ctx any Ollama release has used; below it no default could truncate.
SMALLEST_OLLAMA_DEFAULT_CTX = 2048


def probe_llm(config: EvalConfig) -> list[dict[str, Any]]:
    import httpx
    llm = config.llm or {}
    root = ollama_root(llm.get("base_url"))
    if root is None:
        raise ValueError("probe only supports an Ollama endpoint (base_url with port 11434)")
    model = llm["model"]
    examples = load_examples(config.paths["sample"], config.sample_size)
    criteria = criteria_by_language(config)
    rows = []
    with httpx.Client(timeout=600) as client:
        for language in config.languages:
            field = TEXT_FIELD[language]
            longest = max(examples, key=lambda ex: len(getattr(ex, field)))
            prompt = build_prompt(getattr(longest, field), criteria[language], language)
            client.post(f"{root}/api/generate", json={"model": model, "keep_alive": 0}).raise_for_status()
            reply = client.post(f"{root}/api/chat", json={
                "model": model, "stream": False, "keep_alive": 0, "options": {"temperature": 0},
                "messages": [{"role": "user", "content": prompt}],
            })
            reply.raise_for_status()
            body = reply.json()
            rows.append({"language": language, "example_id": longest.example_id, "prompt_chars": len(prompt),
                         "prompt_tokens": body.get("prompt_eval_count"), "output_tokens": body.get("eval_count"),
                         "answer": (body.get("message") or {}).get("content", "")[:120]})
    return rows


def probe_jev(config: EvalConfig) -> dict[str, Any]:
    from jev_eval.clients.jev_client import JevClient
    examples = load_examples(config.paths["sample"], config.sample_size)
    example = min(examples, key=lambda ex: len(ex.text_en))
    result = JevClient(model=(config.jev or {}).get("model")).classify(
        example.text_en, criteria_by_language(config)["en"], "en")
    return {"requested_model": result.model_id, "served_model": result.served_model,
            "choice": result.predicted_label, "probabilities[choice]": result.confidence,
            "sdk_confidence": result.confidence_sdk}


def run_probe(config: EvalConfig, skip_jev: bool = False) -> None:
    if "llm" in config.systems:
        num_ctx = (config.llm or {}).get("num_ctx")
        for row in probe_llm(config):
            print(row)
            total = (row["prompt_tokens"] or 0) + (row["output_tokens"] or 0)
            print(f"  {row['language']}: prompt+output = {total} tokens; "
                  f"< {SMALLEST_OLLAMA_DEFAULT_CTX} (smallest Ollama default): {total < SMALLEST_OLLAMA_DEFAULT_CTX}; "
                  f"num_ctx in config = {num_ctx}")
    if "jev" in config.systems and not skip_jev:
        print(probe_jev(config))
