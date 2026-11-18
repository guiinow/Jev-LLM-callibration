"""Números do relatório que não saem prontos das tabelas do `analyze` (Q1 e Q3 por sistema).

Lê `outputs/final/raw_predictions.jsonl` e, para cada base e sistema em C3, imprime: ECE (15 faixas),
acurácia, confiança média, AURC, erros com confiança >= 0,9 e a fração de mensagens automatizável
mantendo 90% e 95% de acerto. A fração é a maior cobertura obtida com um limiar de confiança
(aceitar as respostas com confiança >= t, sem desempate dentro de um mesmo valor) cujo acerto entre as
aceitas fica >= a meta; os limiares são escolhidos nos dados de teste, como no relatório.

Uso: python relatorio/deepseek_numbers.py [--condition C3] [--paired-with jev]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from jev_eval.eval.metrics import expected_calibration_error, selective_summary
from jev_eval.pipeline.analyze import with_chosen_probability

RAW = Path("outputs/final/raw_predictions.jsonl")
DATASETS = ("banking77", "clinc150", "sib200")
SYSTEMS = ("jev", "llm", "gemma_prob", "gemma_verb", "deepseek_verb", "deepseek_pro_verb", "gemini_verb")


def automatable(conf: np.ndarray, hit: np.ndarray, target: float) -> float:
    best = 0.0
    for t in np.unique(conf):
        accepted = conf >= t
        if hit[accepted].mean() >= target:
            best = max(best, accepted.mean())
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="C3")
    parser.add_argument("--paired-with", default=None, help="restringe aos itens válidos também neste sistema")
    args = parser.parse_args()
    rows = with_chosen_probability([json.loads(line) for line in RAW.open(encoding="utf-8")])
    valid = defaultdict(dict)
    for r in rows:
        if r["condition"] == args.condition and r["error"] is None:
            valid[(r["dataset"], r["system"])][r["example_id"]] = r
    print(f"condição {args.condition}" + (f", pareado com {args.paired_with}" if args.paired_with else ""))
    print("base;sistema;n;ece;acuracia;conf_media;aurc;erros;erros_conf>=0.9;auto90;auto95")
    for dataset in DATASETS:
        for system in SYSTEMS:
            items = valid.get((dataset, system))
            if not items:
                continue
            ids = sorted(items)
            if args.paired_with:
                ids = sorted(set(ids) & set(valid.get((dataset, args.paired_with), {})))
            conf = np.array([items[i]["confidence"] for i in ids], dtype=float)
            hit = np.array([bool(items[i]["correct"]) for i in ids])
            ece = expected_calibration_error(conf.tolist(), hit.tolist(), 15)
            aurc = selective_summary(conf.tolist(), hit.tolist())["aurc"]
            wrong = ~hit
            print(f"{dataset};{system};{len(ids)};{ece:.3f};{hit.mean():.3f};{conf.mean():.3f};{aurc:.3f};"
                  f"{wrong.sum()};{(wrong & (conf >= 0.9)).sum()};"
                  f"{automatable(conf, hit, 0.90):.3f};{automatable(conf, hit, 0.95):.3f}")


if __name__ == "__main__":
    main()
