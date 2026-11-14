# Calibração Jev x LLM no Banking77 (PT-BR)

Projeto final da disciplina de LLMs (UFMG, 2026/2). Comparamos a calibração de confiança de dois sistemas na classificação de intenções bancárias: o **Jev** (TypeSafe), que devolve a escolha e as probabilidades das opções, e um **LLM generativo** (`ministral-3:14b`, local via Ollama), que devolve a intenção e uma confiança verbalizada. Os dois sistemas recebem as mesmas mensagens do Banking77 e as mesmas 77 opções. Na execução final, o PT-BR vem da tradução pública revisada por humanos [`c2d-usp/banking77-pt-br`](https://huggingface.co/datasets/c2d-usp/banking77-pt-br) (Dimant et al., 2025, CC BY 4.0).

- **H1 (única hipótese principal):** em PT-BR, ECE_Jev < ECE_LLM. A hipótese é apoiada se o IC 95% pareado de ΔECE = ECE_Jev − ECE_LLM ficar abaixo de zero. Se o intervalo incluir zero, o resultado é inconclusivo.
- **Métrica primária:** ECE com 15 bins uniformes, fixados antes do teste. A acurácia entra como contexto.
- **Confiança:** no Jev, `probabilities[escolha]` (proposta, seção 4.2). No LLM, o número verbalizado no JSON.

## Condições

Cada exemplo roda em três condições de idioma, sempre com os dois sistemas:

| Condição | Mensagem | Instruções e opções | Papel |
|---|---|---|---|
| C1 | EN | EN | Linha de base em inglês (exploratória) |
| C2 | PT-BR | EN | PT-BR com opções em inglês |
| C3 | PT-BR | PT-BR | PT-BR com opções em português, o desenho descrito na proposta (seção 4.1) |

Nas figuras, a C2 aparece como "principal". **A condição em que o H1 é testado ainda é decisão do grupo** e precisa constar no relatório. O `metrics.json` traz o ΔECE pareado com IC de todas as condições.

Os prompts são templates congelados e versionados em `src/jev_eval/prompts/templates.py`. Editar o texto muda o fingerprint e quebra `tests/test_templates.py` até que a `version` seja incrementada. O prompt principal é o `choice-v1`. A ablação usa a variante `choice-topk-v1`.

## Execuções

| Pasta | Config | Papel | PT-BR | Jev | LLM | Máquina |
|---|---|---|---|---|---|---|
| `outputs/full_ministral/` | `configs/full_local.yaml` | Preliminar (antes das condições) | Rascunho automático, não revisado | `jev-latest` (alias) | `ministral-3:14b`, `num_ctx` padrão do Ollama | Do Marco Túlio |
| `outputs/final/` | `configs/final.yaml` | **Evidência para H1** (C1, C2 e C3) | Tradução da USP, revisada por humanos (`data/usp/`) | `jev-1.14.0` | `ministral-3-14b-ctx8k` (`num_ctx` 8192) | MacBook Pro M3 Max, 36 GB, Ollama (Metal) |
| `outputs/ablation_topk/` | `configs/ablation_topk.yaml` | Ablação de prompt (E1) | Igual à final | Reaproveitado da final | Igual à final, com o prompt top-3 | Igual à final |

A versão exata de cada componente fica registrada em `outputs/<execução>/run_metadata.json`, uma entrada por invocação de `run`: commit, chip, RAM, versão do Ollama, digest e quantização do modelo, `num_ctx`, versões dos pacotes, condições e datasets. Cada linha de `raw_predictions.jsonl` também traz:
- `served_model`: a versão que a API diz ter usado;
- `usage`: os tokens do LLM;
- `latency_s`: o tempo da chamada;
- `input_hash`.

## Setup

Requer Python ≥ 3.12 (o numpy fixado exige isso).

```bash
python3.12 -m venv .venv        # ou: uv venv --python 3.12 .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # preencha TYPESAFE_API_KEY; nunca commite o .env
pytest                          # a suíte deve passar
```

A CLI `jev-eval` carrega o `.env` da raiz automaticamente. Variáveis já definidas no ambiente têm prioridade.

| Variável | Uso |
|---|---|
| `TYPESAFE_API_KEY` | Obrigatória para chamar o Jev |
| `TYPESAFE_DEFAULT_MODEL` | Fallback. As configs fixam `jev.model` |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL` | Só para configs sem seção `llm:`. As nossas usam o Ollama local, sem chave |

### Ollama (LLM local)

```bash
brew install ollama && ollama serve          # ou o app do ollama.com
ollama pull ministral-3:14b
ollama create ministral-3-14b-ctx8k -f configs/ollama/ministral-3-14b-ctx8k.Modelfile
ollama show ministral-3-14b-ctx8k --parameters   # deve mostrar num_ctx 8192
```

O `num_ctx` é fixado no Modelfile por três motivos: o Ollama **trunca prompts longos sem avisar**, o valor padrão mudou entre versões, e o endpoint compatível com OpenAI (`/v1`) não aceita `num_ctx` por requisição. Por segurança, uma resposta cujo prompt mais a saída use 90% ou mais do `num_ctx` é gravada como erro `ContextOverflow`, e não como resposta válida.

## Reproduzir os resultados principais

```bash
# 1. Entradas PT-BR a partir da tradução da USP (já versionadas em data/usp/; refazer é opcional)
python scripts/build_usp_data.py

# 2. Pré-voo: tokens do prompt mais longo de cada condição e uma chamada ao Jev (versão servida)
jev-eval --config configs/final.yaml probe

# 3. Teste curto opcional (10 exemplos -> outputs/final_timing/, não versionado): tempo por chamada
jev-eval --config configs/final_timing.yaml run

# 4. Execução, métricas, figuras e tabelas
jev-eval --config configs/final.yaml run       # retomável; grava cada resposta ao chegar
jev-eval --config configs/final.yaml analyze   # -> outputs/final/banking77/metrics.json
jev-eval --config configs/final.yaml plot      # -> outputs/final/banking77/{figures,tables}/
```

**Ablação de prompt (E1, análise secundária).** O LLM roda de novo com o prompt `choice-topk-v1`, que pede as 3 intenções mais prováveis, cada uma com a probabilidade de estar correta (o "Verb. 1S top-k" de Tian et al., 2023). A previsão é a intenção de maior probabilidade, e a confiança é essa probabilidade. Como em Tian et al., só esse palpite principal precisa ser uma intenção válida. Palpites inferiores com intenção fora das opções são ignorados e contados em `answers_with_invented_lower_guess`.
- Modelo, `num_ctx`, entradas e exemplos são os mesmos da execução final. A ablação roda as condições C2 e C3; dá para acrescentar a C1.
- O Jev **não é chamado de novo**. O `analyze` e o `plot` usam as respostas do Jev e do LLM com o prompt principal de `outputs/final/raw_predictions.jsonl` (`paths.baseline`). Antes, conferem pelo `input_hash` que cada uma foi produzida com a mesma mensagem, as mesmas opções e o mesmo prompt. Por isso a ablação exige a execução final completa.
- O H1 continua definido sobre o prompt principal.

```bash
jev-eval --config configs/ablation_topk.yaml probe     # tokens do prompt top-3
jev-eval --config configs/ablation_topk.yaml run       # só LLM
jev-eval --config configs/ablation_topk.yaml analyze   # -> outputs/ablation_topk/banking77/{metrics,ablation}.json
jev-eval --config configs/ablation_topk.yaml plot      # -> ... tables/ablation.csv, ablation_deltas.csv
```

Os números reportados podem ser recalculados sem chamar nenhuma API: basta rodar `analyze` e `plot` sobre os `raw_predictions.jsonl` versionados.

**Preliminar.** As entradas são montadas com `python scripts/build_pilot_data.py`. Os registros do preliminar são anteriores às condições (`language` = `en`/`pt_br`):
- o `analyze` e o `plot` os leem como `C1` e `PT-preliminar` (mensagem PT, opções PT, instruções EN, que não é nenhuma das três condições);
- eles também derivam `probabilities[escolha]` do `raw_response` salvo, porque o preliminar gravou o campo `confidence` do SDK. O `metrics.json` informa em `jev_confidence_source` quantos registros puderam ser derivados.

Para refazer a análise sem tocar nas saídas originais: `jev-eval --config configs/full_local.yaml analyze --output-dir outputs/full_ministral/rederived_c1` (e `plot` com o mesmo `--output-dir`).

**Garantias do `run`:**
- Respostas com erro são refeitas, com até 3 tentativas para erros de rede.
- O `run` **recusa retomar** sobre registros produzidos com outras entradas: texto, opções, modelo ou prompt diferentes (detectados pelo `input_hash`), ou registros antigos sem hash. Nesses casos, use uma pasta nova ou `--fresh`.
- Antes de chamar as APIs, o `run` falha se:
  - faltar alguma descrição no idioma de instrução de uma condição;
  - duas descrições forem iguais;
  - algum texto PT-BR estiver vazio ou idêntico ao inglês.

  Ele também avisa enquanto houver traduções não revisadas.
- Respostas inválidas guardam o texto bruto do LLM. O parser lê só o primeiro objeto JSON da resposta, porque o modelo às vezes acrescenta uma nota e um segundo bloco "corrigido".

**Teste sem APIs:** `jev-eval --config configs/smoke.yaml smoke` ou `jev-eval --config configs/default.yaml run --mock`. Os mocks são determinísticos e servem só para testar o pipeline. O smoke usa descrições PT-BR marcadas `[SMOKE-PLACEHOLDER]`.

## Datasets

Cada config lista seus `datasets`: nome, tipo de tarefa (`choice`, `binary` ou `multilabel`), caminhos e, se o dataset vier do Hugging Face, um `source` com `revision` obrigatória. Um dataset novo entra com um adapter em `src/jev_eval/data/`, registrado por nome, e um template de prompt. Hoje só o `banking77` (`choice`) está implementado. As métricas nunca juntam datasets: `analyze` e `plot` rodam um dataset por vez e gravam em `<saída>/<dataset>/`.

## Tradução PT-BR

**Execução final: tradução da USP.** Usamos [`c2d-usp/banking77-pt-br`](https://huggingface.co/datasets/c2d-usp/banking77-pt-br) (Dimant et al., 2025, CC BY 4.0). Segundo o cartão do dataset, a tradução foi feita com GPT (`gpt 4o-mini`, e `gpt 3.5-turbo` em 3 frases do teste) e depois revisada por humanos. A coluna `alterado` marca as frases corrigidas nessa revisão.

O dataset não traz o texto em inglês nem IDs. Por isso o `scripts/build_usp_data.py`:
1. baixa as duas fontes em revisões fixadas (`PolyAI/banking77` `1846984`, `c2d-usp/banking77-pt-br` `85f51c4`);
2. descarta as 2 linhas vazias no fim do teste da USP (3.082 → 3.080) e recusa linhas vazias em outra posição;
3. confere que a ordem é a mesma. Mapeando os nomes PT para IDs pela tabela do cartão do dataset, a intenção coincide nas 3.080 posições. Além disso, o comprimento das frases EN e PT tem correlação de 0,972 item a item, contra 0,118 com as frases embaralhadas dentro de cada intenção. Se qualquer checagem falhar, o script para;
4. casa cada um dos 462 exemplos da amostra com o seu índice no teste original pelo texto em inglês (não há duplicatas no teste), confere o rótulo e copia a frase PT-BR;
5. usa os nomes PT das intenções, com `_` trocado por espaço, como descrições PT-BR das opções (C3), e confere que as 77 são distintas.

Ele gera, em `data/usp/`:
- `sample_usp.jsonl`;
- `intents_usp.json`;
- `alignment_usp.jsonl`: índice no teste, `alterado` e `modelo_tradutor` por exemplo (28 dos 462 têm `alterado = 1`);
- `provenance.json`.

**Preliminar: rascunho próprio, não revisado.** Feito com modelo local via Ollama (`gemma4:12b-it-qat`) e glossário fixo (`scripts/translate_local.py`), e montado em `data/pilot/` por `scripts/build_pilot_data.py`. As planilhas de revisão (`scripts/export_review_sheets.py`, `import_review_sheets.py`, `merge_translations.py`, `data/review/`) eram o fluxo de revisão previsto e não são usadas na execução final.

## Saídas de uma execução

Por dataset, em `<saída>/<dataset>/`. As figuras e tabelas usam só os exemplos em que os dois sistemas responderam, para que ECE_Jev e ECE_LLM sejam calculados sobre os mesmos itens.

| Arquivo | Para quê |
|---|---|
| `../raw_predictions.jsonl` | Uma linha por (dataset, exemplo, condição, sistema): previsão, confiança, acerto, resposta bruta (também nas respostas inválidas), `served_model`, `usage`, `confidence_sdk` (Jev), `input_hash`, `latency_s` |
| `../run_metadata.json` | Proveniência de cada invocação de `run` |
| `metrics.json` | Por condição: ECE, acurácia, IC 95% por bootstrap, confiança média, excesso de confiança, erros com confiança ≥ 0,9, ΔECE pareado (5000 reamostragens, seed 42), chamadas que falharam e `jev_confidence_sdk_check` (ECE com o `confidence` do SDK, só como checagem) |
| `figures/reliability.png` | Diagrama de confiabilidade (área ∝ tamanho do bin) + histograma de confiança, por condição |
| `figures/ece_accuracy_ci.png` | ECE e acurácia por sistema e condição, com IC 95% |
| `figures/delta_ece_bootstrap.png` | Distribuição bootstrap de ΔECE = Jev − LLM |
| `figures/confidence_correct_vs_wrong.png` | Confiança em acertos e em erros |
| `figures/risk_coverage.png` | Exploratório: acurácia das respostas aceitas vs cobertura |
| `figures/top_confusions.png` | Pares (verdadeiro → previsto) mais confundidos |
| `tables/summary.csv` | Tabela principal |
| `tables/robustness.csv` | Robustez (E2), por condição e sistema: ECE com 10, 15 e 20 bins uniformes e com 15 bins de massa igual, Brier top-1, AUROC (acerto × erro) e número de valores distintos de confiança. O Jev aparece também com o `confidence` do SDK, só como checagem |
| `tables/ablation.csv`, `ablation_deltas.csv` (só na ablação) | Jev, LLM com o prompt principal e LLM top-3, sobre os itens em que os três responderam: métricas e ΔECE pareado com IC 95% (top-3 − principal, Jev − top-3, Jev − principal) |
| `tables/reliability_bins.csv`, `high_confidence_errors.csv`, `confusion_pairs.csv`, `language_flips.csv` | Bins, erros com confiança alta, confusões e efeito do idioma (C1 vs cada condição PT) |

## Método

- Banking77, split `test`, amostragem estratificada com seed 42: 6 exemplos por intenção × 77 = 462 (a proposta fala em "até 500").
- Mesmas mensagens e opções para os dois sistemas, em cada condição.
- O LLM roda com temperatura 0 e prompts congelados.
- O ECE usa bins uniformes congelados no YAML (`final` = 15).
- O IC 95% de ΔECE vem de bootstrap pareado por `example_id`, com 5000 reamostragens e seed 42.
- **Robustez (E2):** a métrica primária continua sendo o ECE com 15 bins uniformes. Os outros números são checagens e não viram hipótese.
  - Nos bins de massa igual, os itens empatados são divididos pela ordem original.
  - O Jev devolve as probabilidades com 2 casas decimais, então há muitos empates. `distinct_confidence_values` mostra quantos valores distintos existem.
- O estudo compara produtos e interfaces. Ele não permite concluir que RLCD seja superior a RLHF.

## Desvios do protocolo e limitações conhecidas

- **Condições C1, C2 e C3.** A proposta compara EN e PT-BR. As condições separam o idioma da mensagem do idioma das opções e instruções. A C3 é o PT-BR descrito na proposta. A C2 é uma condição nova.
- **Tradução substituída.** A proposta previa uma tradução própria revisada pelo grupo. Na execução final usamos a tradução pública da USP, também revisada por humanos, porque:
  - a revisão dela é completa, e não por amostra;
  - ela está documentada e é citável;
  - qualquer pessoa pode reproduzir os insumos.

  A amostra e as intenções não mudam. As descrições PT-BR das opções passam a ser os nomes PT das intenções definidos pela USP.
- **Risco de contaminação.** A tradução está pública desde 2025, e o Banking77 em inglês desde 2020. O `ministral-3:14b` e o modelo por trás do Jev podem ter visto essas frases, ou os pares frase-intenção, no treino.
  - Isso pode inflar a acurácia, e não temos como medir.
  - Não há garantia de que o efeito seja igual nos dois sistemas.
  - A comparação de ECE continua pareada sobre os mesmos itens, mas os números absolutos devem ser lidos com isso em mente.
- **Preliminar:**
  - PT-BR não revisado;
  - Jev pelo alias `jev-latest`, sem registro da versão servida;
  - confiança do Jev gravada a partir do `confidence` do SDK (corrigido na análise);
  - `num_ctx` padrão do Ollama (o probe mostrou que nenhum prompt foi truncado);
  - outra máquina;
  - anterior às condições.
- O estudo avalia o Banking77 **traduzido**, e não a linguagem bancária brasileira nativa.

## AI disclosure (rascunho)

Ferramentas de IA generativa foram usadas para auxiliar a implementação do pipeline e a revisão de código. As decisões metodológicas, a revisão das traduções, a execução das avaliações e a interpretação dos resultados são responsabilidade do grupo.
