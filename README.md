# Calibração de confiança: Jev e LLMs de EN para PT-BR

Projeto final da disciplina de LLMs (UFMG, 2026/2). Avaliamos se a confiança
do Jev é calibrada em português e como a mudança de inglês para português afeta
Jev e LLMs generativos.

## Perguntas e desenho

- **Q1 (H1):** em PT-BR, o ECE do Jev é menor que o do sistema comparado?
- **Q2:** o ECE e a acurácia mudam de EN para PT?
- **Q3:** a confiança permite selecionar respostas para automação?

A métrica principal é o ECE com 15 bins uniformes. A hipótese usa
`ΔECE = ECE_Jev - ECE_sistema` e é apoiada quando o IC 95% pareado fica abaixo
de zero. A confiança do Jev é a probabilidade da opção escolhida; nos LLMs
generativos, é o valor numérico retornado no JSON.

Cada exemplo é avaliado em três condições:

| Condição | Mensagem | Instruções e opções |
|---|---|---|
| C1 | Inglês | Inglês |
| C2 | Português | Inglês |
| C3 | Português | Português |

C3 é a condição principal de Q1. O contraste primário de Q2 é C2 − C1.

Os datasets são:

| Dataset | Amostra | Tradução |
|---|---:|---|
| Banking77 | 462 | PT-BR da USP, revisada por humanos |
| CLINC150 | 450 | Tradução automática, não revisada |
| SIB-200 | 204 | FLORES-200, tradução profissional |

O estudo compara Jev, Ministral, Gemma e modelos via API. Gemma é avaliado
por confiança verbalizada e por probabilidades das opções. Os resultados
versionados estão em [`outputs/final/`](outputs/final/).

## Requisitos e instalação

Python 3.12 ou superior:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
pytest
```

Para chamadas ao Jev, preencha `TYPESAFE_API_KEY` no ambiente ou no `.env`.
As configurações locais usam Ollama e não exigem chave de API:

```bash
brew install ollama
ollama serve
ollama pull ministral-3:14b
ollama create ministral-3-14b-ctx8k \
  -f configs/ollama/ministral-3-14b-ctx8k.Modelfile
```

## Reprodução

Os dados de entrada já estão versionados. Para reconstruí-los:

```bash
python scripts/build_usp_data.py
python scripts/build_clinc150_data.py
python scripts/build_sib200_data.py
```

Teste local sem APIs:

```bash
jev-eval --config configs/smoke.yaml smoke
```

Execução e análise principais:

```bash
jev-eval --config configs/final.yaml probe
jev-eval --config configs/final.yaml run
jev-eval --config configs/final.yaml analyze
jev-eval --config configs/final.yaml plot
jev-eval --config configs/final.yaml check-benchmark
jev-eval --config configs/final.yaml report --lang pt
```

O `run` é retomável e salva as respostas à medida que chegam. A configuração
final usa Jev `1.14.0`, Ministral com contexto 8192, temperatura zero e
amostragem estratificada com seed 42. A execução registrada foi feita em
Linux x86_64; os metadados completos estão em
`outputs/final/run_metadata.json`.

Modelos abertos opcionais:

```bash
pip install -e ".[open-models]"
jev-eval --config configs/open_models.yaml run
jev-eval --config configs/open_models.yaml analyze
```

## Resultados salvos

Em cada diretório `outputs/<execução>/<dataset>/`:

- `metrics.json`: ECE, acurácia, ICs, bootstrap pareado e mudança EN→PT;
- `tables/language_shift.csv`: Q2;
- `tables/selective.csv`: Q3, AURC e seleção por cobertura;
- `tables/bin_sensitivity.csv` e `tables/robustness.csv`: análises de robustez;
- `tables/summary.csv`: tabela principal;
- `figures/`: diagramas de confiabilidade, ECE, risco e confusões;
- `report/`: figuras vetoriais para o relatório.

Na raiz da execução:

- `raw_predictions.jsonl`: previsões, confianças, respostas brutas e hashes
  das entradas;
- `run_metadata.json`: versões, configuração e ambiente;
- `summary_datasets.json`: resumo por dataset, sem misturar previsões.

O benchmark em inglês é conferido com:

```bash
jev-eval --config configs/final.yaml check-benchmark
```

## Limitações e desvios

- As conclusões valem para dados traduzidos, não para mensagens originalmente
  escritas em português.
- As traduções do CLINC150 e os nomes de suas intenções não foram revisados
  pelo grupo.
- Os datasets têm tamanhos, número de classes e qualidade de tradução
  diferentes; por isso os resultados são apresentados por dataset.
- A amostra é menor que o teste completo do benchmark, produzindo ICs largos.
- O estudo compara sistemas completos e não permite concluir que uma
  arquitetura ou método de treinamento causa a diferença observada.
- As leituras do Gemma usam prompts diferentes, portanto o contraste mistura
  forma de leitura e prompt.

## Análise de erros

Para gerar a planilha:

```bash
jev-eval --config configs/final.yaml error-sheet
```

Depois de preencher `category`, gere o resumo:

```bash
jev-eval --config configs/final.yaml error-summary
```

## Declaração de uso de IA

Ferramentas de IA generativa foram usadas na implementação do pipeline, na
tradução dos nomes das intenções do CLINC150 e na redação do relatório. Todo o
material foi revisado pelo grupo, que é responsável pelo desenho experimental,
pela execução, pela validação dos resultados e pelas conclusões.

## Referências internas

- [Proposta refinada](docs/proposta-refinada.md)
- [Plano do artigo](docs/plano_do_artigo_e_prs.md)
- [Especificação do curso](docs/especificacao-do-curso.md)
