# Calibração Jev x LLM no Banking77

Projeto de curso sobre a calibração de confiança na classificação de intenções bancárias em PT-BR. A pergunta principal compara o `confidence` estruturado do Jev com a confiança verbalizada por um LLM generativo nos mesmos exemplos do Banking77 traduzidos e revisados.

## Status

O pipeline, os testes, uma amostra smoke de seis pares EN/PT-BR, os gráficos e os PDFs de especificação estão prontos. A amostra completa ainda não foi baixada e nenhuma API real é chamada pelo smoke run.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

O `.env` deve ser preenchido apenas para as execuções reais; nunca commite chaves. A CLI carrega o `.env` da raiz do repositório automaticamente (variáveis já definidas no ambiente têm prioridade).

`run` grava cada resposta em `raw_predictions.jsonl` assim que ela chega e, se interrompido, retoma de onde parou ao ser executado de novo (respostas com erro são refeitas; cada chamada tem até 3 tentativas). Use `run --fresh` para descartar as respostas existentes. Antes de chamar as APIs, o `run` falha se faltar `description_pt_br` em alguma intenção ou se algum `text_pt_br` estiver vazio ou idêntico ao inglês. O `metrics.json` reporta, por idioma, os erros excluídos (`errors`), as métricas sobre os exemplos em que os dois sistemas responderam (`paired`, base do bootstrap) e sobre todas as respostas válidas (`all_valid`). O smoke usa `data/intents_smoke.json`, com descrições PT-BR marcadas `[SMOKE-PLACEHOLDER]` só para testar o pipeline.

## Comandos

```bash
jev-eval --config configs/smoke.yaml smoke
jev-eval --config configs/default.yaml run --mock
jev-eval --config configs/default.yaml analyze
jev-eval --config configs/default.yaml plot
```

O smoke gera `outputs/smoke/raw_predictions.jsonl`, `outputs/smoke/metrics.json` e `outputs/smoke/reliability_diagram.png`. Os mocks são determinísticos e servem somente para testar o pipeline, não para evidência científica.

## Tradução PT-BR

1. Rascunho automático com modelo local via [Ollama](https://ollama.com) (`gemma4:12b-it-qat`, de família diferente do LLM avaliado, para não enviesar a comparação), com glossário fixo:
   `python scripts/translate_local.py` → preenche `description_pt_br` em `data/intents.json` e gera `data/translations_pt_br.jsonl` com `translation_status: machine_draft` (retomável).
2. Planilhas de revisão humana (CSV `;`, UTF-8), divididas por intenção entre os revisores:
   `python scripts/export_review_sheets.py --reviewers Nome1 Nome2 Nome3 Nome4` → `data/review/`.
   Cada revisor corrige a coluna `*_final` e preenche `status` com `ok` ou `corrigido` (e `observacao` para casos difíceis).
3. Importar e aplicar: `python scripts/import_review_sheets.py` e depois `python scripts/merge_translations.py`.

O `run` recusa descrições PT-BR vazias ou duplicadas e avisa enquanto houver traduções não revisadas.

## Saídas de uma execução

`analyze` grava `metrics.json` (ECE, acurácia, IC 95% bootstrap por sistema, confiança média, excesso de confiança, erros com confiança ≥ 0,9 e ΔECE pareado). `plot` grava, ao lado de `raw_predictions.jsonl`, sempre sobre os exemplos em que os dois sistemas responderam:

| Arquivo | Para quê |
|---|---|
| `figures/reliability.png` | Diagrama de confiabilidade (área ∝ tamanho do bin) + histograma de confiança, por idioma |
| `figures/ece_accuracy_ci.png` | ECE e acurácia por sistema/idioma com IC 95% |
| `figures/delta_ece_bootstrap.png` | Distribuição bootstrap de ΔECE = Jev − LLM (teste de H1: IC abaixo de 0) |
| `figures/confidence_correct_vs_wrong.png` | Confiança em acertos vs erros, por sistema e idioma |
| `figures/risk_coverage.png` | Exploratório: acurácia das respostas aceitas vs cobertura, por limiar de confiança |
| `figures/top_confusions.png` | Pares (verdadeiro → previsto) mais confundidos |
| `tables/summary.csv` | Tabela principal do relatório |
| `tables/reliability_bins.csv` | Contagem, confiança e acurácia por bin |
| `tables/high_confidence_errors.csv` | Erros com confiança ≥ 0,9, com o texto, para análise qualitativa |
| `tables/confusion_pairs.csv`, `tables/language_flips.csv` | Análise de erros e efeito da tradução (EN vs PT-BR) |

Piloto (100 exemplos, PT-BR em rascunho, LLM local `ministral-3:14b` via Ollama): `python scripts/build_pilot_data.py`, depois `jev-eval --config configs/pilot.yaml run`, `analyze` e `plot`.

## Método

Usamos Banking77, split `test`, amostragem estratificada (seed 42), até 500 exemplos, e as mesmas mensagens e opções para Jev e LLM. A métrica primária é ECE com bins uniformes congelados no YAML (`smoke=6`, `default=15`). O intervalo de 95% da diferença `ECE_jev - ECE_llm` usa bootstrap pareado por `example_id`, 5000 reamostragens e seed 42. Acurácia é apenas contexto. O estudo avalia produtos/interfaces e não permite concluir superioridade de RLCD sobre RLHF.

## Próximos passos (não executados)

1. Preencher as chaves no `.env`.
2. Executar `python scripts/prepare_banking77_sample.py --sample-size 500 --seed 42`.
3. Traduzir e revisar humanamente as mensagens PT-BR e as descrições das intenções; aplicar `scripts/merge_translations.py`.
4. Fazer piloto com APIs reais e congelar prompts/modelos.
5. Executar a avaliação principal EN+PT-BR com Jev+LLM.
6. Escrever o relatório ACM de duas páginas e preparar a apresentação.

## AI disclosure (rascunho)

Ferramentas de IA generativa foram usadas para auxiliar a implementação do pipeline e a revisão de código. As decisões metodológicas, a revisão das traduções, a execução das avaliações e a interpretação dos resultados são responsabilidade do grupo.
