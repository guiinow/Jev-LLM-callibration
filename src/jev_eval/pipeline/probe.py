"""Pre-flight checks before a real run (a handful of calls, nothing written to outputs/).

LLM: for each condition, sends the longest real prompt to Ollama's native API and reports the
exact prompt token count, so we know the prompt fits in num_ctx (Ollama truncates silently).
The model is unloaded before each call so a warm prompt cache cannot make the reported count
smaller than the real prompt.

Jev: one call (C1), to confirm the served model version and that probabilities[choice] exists.
"""
from typing import Any

from jev_eval.conditions import get_condition
from jev_eval.config import EvalConfig
from jev_eval.data.datasets import build_adapter
from jev_eval.pipeline.metadata import ollama_root
from jev_eval.pipeline.run_inference import criteria_by_language, prompt_for

# Smallest default num_ctx any Ollama release has used; below it no default could truncate.
SMALLEST_OLLAMA_DEFAULT_CTX = 2048


def probe_llm(config: EvalConfig) -> list[dict[str, Any]]:
    import httpx
    llm = config.llm or {}
    root = ollama_root(llm.get("base_url"))
    if root is None:
        raise ValueError("probe only supports an Ollama endpoint (base_url with port 11434)")
    model = llm["model"]
    conditions = [get_condition(name) for name in config.conditions]
    rows = []
    with httpx.Client(timeout=600) as client:
        for spec in config.datasets:
            adapter = build_adapter(spec)
            examples = adapter.load_examples(config.sample_size)
            criteria = criteria_by_language(adapter, conditions)
            for condition in conditions:
                longest = max(examples, key=lambda ex: len(ex.text[condition.content_language]))
                prompt = prompt_for(spec.name, condition, "llm", config).render_llm(
                    longest.text[condition.content_language], criteria[condition.instruction_language])
                client.post(f"{root}/api/generate", json={"model": model, "keep_alive": 0}).raise_for_status()
                reply = client.post(f"{root}/api/chat", json={
                    "model": model, "stream": False, "keep_alive": 0, "options": {"temperature": 0},
                    "messages": [{"role": "user", "content": prompt}],
                })
                reply.raise_for_status()
                body = reply.json()
                rows.append({"dataset": spec.name, "condition": condition.name, "example_id": longest.example_id,
                             "prompt_chars": len(prompt), "prompt_tokens": body.get("prompt_eval_count"),
                             "output_tokens": body.get("eval_count"),
                             "answer": (body.get("message") or {}).get("content", "")[:120]})
    return rows


def probe_jev(config: EvalConfig) -> dict[str, Any]:
    from jev_eval.clients.jev_client import JevClient
    spec = config.datasets[0]
    adapter = build_adapter(spec)
    c1 = get_condition("C1")
    example = min(adapter.load_examples(config.sample_size), key=lambda ex: len(ex.text["en"]))
    result = JevClient(model=(config.jev or {}).get("model")).classify(
        example.text["en"], criteria_by_language(adapter, [c1])["en"], prompt_for(spec.name, c1, "jev", config))
    return {"example_id": example.example_id, "message": example.text["en"], "label": example.label,
            "requested_model": result.model_id, "served_model": result.served_model,
            "choice": result.predicted_label, "probabilities[choice]": result.confidence,
            "sdk_confidence": result.confidence_sdk}


def run_probe(config: EvalConfig, skip_jev: bool = False) -> None:
    if "llm" in config.systems:
        num_ctx = (config.llm or {}).get("num_ctx")
        for row in probe_llm(config):
            print(row)
            total = (row["prompt_tokens"] or 0) + (row["output_tokens"] or 0)
            print(f"  {row['condition']}: prompt+output = {total} tokens; "
                  f"< {SMALLEST_OLLAMA_DEFAULT_CTX} (smallest Ollama default): {total < SMALLEST_OLLAMA_DEFAULT_CTX}; "
                  f"num_ctx in config = {num_ctx}")
    if "jev" in config.systems and not skip_jev:
        print(probe_jev(config))
