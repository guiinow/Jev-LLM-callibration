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

O `.env` deve ser preenchido apenas para as execuções reais; nunca commite chaves.

## Comandos

```bash
jev-eval --config configs/smoke.yaml smoke
jev-eval --config configs/default.yaml run --mock
jev-eval --config configs/default.yaml analyze
jev-eval --config configs/default.yaml plot
```

O smoke gera `outputs/smoke/raw_predictions.jsonl`, `outputs/smoke/metrics.json` e `outputs/smoke/reliability_diagram.png`. Os mocks são determinísticos e servem somente para testar o pipeline, não para evidência científica.

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
