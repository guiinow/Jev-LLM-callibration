"""Draft EN -> PT-BR translations with a local model served by Ollama.

Drafts are marked `translation_status: "machine_draft"` and MUST be reviewed by the group
before the main run (see scripts/export_review_sheets.py). The model used to draft is a
different family from the evaluated LLM, to avoid biasing the comparison.

Usage:
    python scripts/translate_local.py                       # messages + intent descriptions
    python scripts/translate_local.py --only descriptions
    python scripts/translate_local.py --model gemma3:12b

Resumable: already-translated items are skipped; re-run after an interruption.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma4:12b-it-qat"
PROMPT_VERSION = "mt-ptbr-v1"

GLOSSARY = """top up -> recarga / recarregar
transfer -> transferência
card payment -> pagamento com cartão
cash withdrawal -> saque
pending -> pendente
declined -> recusado
reverted -> estornado
refund -> reembolso
exchange rate -> taxa de câmbio
disposable virtual card -> cartão virtual descartável
virtual card -> cartão virtual
physical card -> cartão físico
spare card -> cartão adicional
PIN -> PIN
passcode -> código de acesso
beneficiary -> favorecido
direct debit -> débito automático
statement -> extrato
ATM -> caixa eletrônico
contactless -> pagamento por aproximação
fee / charge (cost) -> taxa / tarifa
verify identity -> verificar identidade
source of funds -> origem dos recursos"""

MESSAGE_SYSTEM = f"""You translate customer-support messages from a UK bank's app from English into Brazilian Portuguese (pt-BR).
Rules:
- Preserve exactly the meaning and the customer's intent. Do not add, remove or explain information.
- Keep the informal register, capitalization style and punctuation of the original (do not add a question mark if there is none).
- Keep currencies, amounts, dates and brand/product names as in the original (e.g. £, €, Apple Pay, Google Pay, Visa, Mastercard).
- Use natural Brazilian Portuguese (not European Portuguese), but avoid heavy slang.
- Use this glossary consistently:
{GLOSSARY}
Answer with the translation only: no quotes, no notes, no alternatives."""

DESCRIPTION_SYSTEM = f"""You translate short intent labels of a banking customer-support classifier from English into Brazilian Portuguese (pt-BR).
Each label is a short phrase describing what the customer wants (e.g. "activate my card" -> "ativar meu cartão").
Keep it short, keep the same meaning and level of specificity, and keep brand names (Apple Pay, Google Pay, Visa, Mastercard).
Use this glossary consistently:
{GLOSSARY}
Answer with the translation only: no quotes, no notes."""


def clean(text: str) -> str:
    text = text.strip().splitlines()[0].strip() if text.strip() else ""
    text = re.sub(r"^(tradução|translation)\s*:\s*", "", text, flags=re.IGNORECASE)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    return text.strip("“”").strip()


def translate(client: httpx.Client, model: str, system: str, text: str) -> str:
    for attempt in range(3):
        try:
            response = client.post(OLLAMA_URL, json={
                "model": model, "stream": False, "think": False,
                "options": {"temperature": 0, "seed": 42},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
            })
            response.raise_for_status()
            result = clean(response.json()["message"]["content"])
            if result:
                return result
            raise ValueError("empty translation")
        except Exception as error:
            if attempt == 2:
                raise
            print(f"  retry after error: {error}", file=sys.stderr)
            time.sleep(3)
    raise RuntimeError("unreachable")


def translate_messages(client, model, sample_path: Path, output_path: Path) -> None:
    rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["example_id"]] = row
    todo = [row for row in rows if row["example_id"] not in done]
    print(f"Messages: {len(done)} already translated, {len(todo)} to go")
    start = time.time()
    with output_path.open("a", encoding="utf-8") as file:
        for index, row in enumerate(todo, 1):
            pt = translate(client, model, MESSAGE_SYSTEM, row["text_en"])
            file.write(json.dumps({
                "example_id": row["example_id"], "label_name": row["label_name"], "text_en": row["text_en"],
                "text_pt_br": pt, "mt_draft": pt, "translation_status": "machine_draft",
                "mt_model": model, "mt_prompt_version": PROMPT_VERSION,
            }, ensure_ascii=False) + "\n")
            file.flush()
            if index % 10 == 0 or index == len(todo):
                rate = (time.time() - start) / index
                print(f"  {index}/{len(todo)}  ~{rate:.1f}s/item  ETA {rate * (len(todo) - index) / 60:.0f} min")


def translate_descriptions(client, model, intents_path: Path) -> None:
    intents = json.loads(intents_path.read_text(encoding="utf-8"))
    todo = [row for row in intents if not str(row.get("description_pt_br", "")).strip()]
    print(f"Descriptions: {len(intents) - len(todo)} already translated, {len(todo)} to go")
    for index, row in enumerate(todo, 1):
        row["description_pt_br"] = translate(client, model, DESCRIPTION_SYSTEM, row["description_en"])
        row["description_pt_br_status"] = "machine_draft"
        row["description_mt_model"] = model
        intents_path.write_text(json.dumps(intents, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if index % 10 == 0 or index == len(todo):
            print(f"  {index}/{len(todo)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", default="data/sample.jsonl")
    parser.add_argument("--output", default="data/translations_pt_br.jsonl")
    parser.add_argument("--intents", default="data/intents.json")
    parser.add_argument("--only", choices=["messages", "descriptions"])
    args = parser.parse_args()
    with httpx.Client(timeout=600) as client:
        if args.only != "messages":
            translate_descriptions(client, args.model, Path(args.intents))
        if args.only != "descriptions":
            translate_messages(client, args.model, Path(args.sample), Path(args.output))


if __name__ == "__main__":
    main()
