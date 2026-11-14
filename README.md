# Calibração Jev x LLM no Banking77 (PT-BR)

Projeto final da disciplina de LLMs (UFMG, 2026/2). Comparamos a calibração de confiança de dois sistemas na classificação de intenções bancárias: o **Jev** (TypeSafe), que devolve a escolha e as probabilidades das opções, e um **LLM generativo** (`ministral-3:14b`, local via Ollama), que devolve a intenção e uma confiança verbalizada. Os dois sistemas recebem as mesmas mensagens do Banking77 em PT-BR e as mesmas 77 opções. Na execução final, o PT-BR vem da tradução pública revisada por humanos [`c2d-usp/banking77-pt-br`](https://huggingface.co/datasets/c2d-usp/banking77-pt-br) (Dimant et al., 2025, CC BY 4.0).

- **H1 (única hipótese principal):** em PT-BR, ECE_Jev < ECE_LLM. A hipótese é apoiada se o IC 95% pareado de ΔECE = ECE_Jev − ECE_LLM ficar abaixo de zero. Se o intervalo incluir zero, o resultado é inconclusivo.
- **Inglês:** os mesmos exemplos, como comparação exploratória.
- **Métrica primária:** ECE com 15 bins uniformes, fixados antes do teste. A acurácia entra como contexto.
- **Confiança:** no Jev, `probabilities[escolha]` (proposta, seção 4.2). No LLM, o número verbalizado no JSON.

## Execuções

| Pasta | Config | Papel | PT-BR | Jev | LLM | Máquina |
|---|---|---|---|---|---|---|
| `outputs/full_ministral/` | `configs/full_local.yaml` | Preliminar | Rascunho automático (não revisado) | `jev-latest` (alias) | `ministral-3:14b`, `num_ctx` padrão do Ollama | Do Marco Túlio |
| `outputs/final/` | `configs/final.yaml` | **Evidência para H1** | Tradução pública revisada por humanos (USP, `data/usp/`) | Versão fixada (`jev.model`) | `ministral-3-14b-ctx8k` (`num_ctx` 8192) | MacBook Pro M3 Max, 36 GB, Ollama (Metal) |

A versão exata de cada componente (commit, chip, RAM, versão do Ollama, digest e quantização do modelo, `num_ctx`, versões dos pacotes) fica registrada em `outputs/<execução>/run_metadata.json`, uma entrada por invocação de `run`. Cada linha de `raw_predictions.jsonl` também traz o `served_model` (a versão que a API diz ter usado), o `usage` (tokens do LLM) e o `input_hash`. Todas as execuções com LLM a partir da final rodam na mesma máquina.

| Componente da execução final | Valor |
|---|---|
| Hardware | MacBook Pro, Apple M3 Max, 36 GB (detalhes em `run_metadata.json` → `machine`) |
| Ollama | Versão em `run_metadata.json` → `ollama.version` |
| LLM | `ministral-3:14b` (digest e quantização em `run_metadata.json` → `ollama`), com `num_ctx` 8192 via `configs/ollama/ministral-3-14b-ctx8k.Modelfile` |
| Jev | `jev.model` em `configs/final.yaml`. A versão servida fica em `served_model` em cada linha |

## Setup

Requer Python ≥ 3.12 (o numpy fixado exige isso).

```bash
python3.12 -m venv .venv        # ou: uv venv --python 3.12 .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # preencha TYPESAFE_API_KEY; nunca commite o .env
pytest                          # a suíte deve passar
```

A CLI `jev-eval` carrega o `.env` da raiz automaticamente (variáveis já definidas no ambiente têm prioridade). Variáveis:

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

O `num_ctx` é fixado no Modelfile porque o Ollama **trunca prompts longos sem avisar**, o valor padrão mudou entre versões e o endpoint compatível com OpenAI (`/v1`) não aceita `num_ctx` por requisição. Por segurança, uma resposta cujo prompt mais a saída use 90% ou mais do `num_ctx` configurado é gravada como erro `ContextOverflow`, e não como resposta válida.

## Reproduzir os resultados principais

```bash
# 1. Entradas PT-BR a partir da tradução da USP (já versionadas em data/usp/; refazer é opcional)
python scripts/build_usp_data.py           # -> data/usp/{sample_usp.jsonl, intents_usp.json, alignment_usp.jsonl, provenance.json}

# 2. Pré-voo: tokens do prompt mais longo (EN e PT-BR) e uma chamada ao Jev (versão servida)
jev-eval --config configs/final.yaml probe

# 3. Teste curto opcional (10 exemplos -> outputs/final_timing/, não versionado): tempo por chamada em latency_s
jev-eval --config configs/final_timing.yaml run

# 4. Execução, métricas, figuras e tabelas
jev-eval --config configs/final.yaml run       # retomável; grava cada resposta ao chegar
jev-eval --config configs/final.yaml analyze   # -> outputs/final/metrics.json
jev-eval --config configs/final.yaml plot      # -> outputs/final/figures/, outputs/final/tables/
```

**Ablação de prompt (E1, análise secundária).** O LLM é rodado de novo com o prompt `llm-intent-topk-v1`: as 3 intenções mais prováveis, cada uma com a probabilidade de estar correta (o "Verb. 1S top-k" de Tian et al., 2023). A previsão é a intenção de maior probabilidade, e a confiança é essa probabilidade. O modelo, o `num_ctx`, as entradas e os exemplos são os mesmos da execução final. O Jev **não é chamado de novo**: o `analyze` e o `plot` usam as respostas do Jev e do LLM v1 de `outputs/final/raw_predictions.jsonl` (`paths.baseline`), depois de conferir pelo `input_hash` que cada uma foi produzida a partir da mesma mensagem e das mesmas opções. A ablação só pode rodar com a execução final completa. O H1 continua definido sobre o prompt v1.

```bash
jev-eval --config configs/ablation_topk.yaml probe     # tokens do prompt top-3
jev-eval --config configs/ablation_topk.yaml run       # só LLM, PT-BR (incluir en em languages se der tempo)
jev-eval --config configs/ablation_topk.yaml analyze   # -> outputs/ablation_topk/{metrics.json, ablation.json}
jev-eval --config configs/ablation_topk.yaml plot      # -> ... tables/ablation.csv, tables/ablation_deltas.csv
```

Sem chamar nenhuma API, os números reportados podem ser recalculados a partir de `outputs/final/raw_predictions.jsonl`, que está versionado, com `analyze` e `plot`.

**Preliminar.** As entradas são montadas com `python scripts/build_pilot_data.py`. Para refazer a análise sem tocar nas saídas originais:
`jev-eval --config configs/full_local.yaml analyze --output-dir outputs/full_ministral/rederived_c1` (e `plot` com o mesmo `--output-dir`). Os registros do preliminar foram gravados com o campo `confidence` do SDK. O `analyze`/`plot` derivam `probabilities[escolha]` do `raw_response` salvo e informam em `metrics.json` (`jev_confidence_source`) quantos registros puderam ser derivados.

**Garantias do `run`:**
- Respostas com erro são refeitas, com até 3 tentativas para erros de rede.
- O `run` **recusa retomar** sobre registros produzidos a partir de outras entradas (texto, opções, modelo ou prompt diferentes, detectados pelo `input_hash`) ou sobre registros antigos sem hash. Nesse caso, use uma pasta nova ou `--fresh`.
- O `run` falha antes de chamar as APIs se faltar uma descrição PT-BR, se duas descrições forem iguais ou se algum `text_pt_br` estiver vazio ou idêntico ao inglês. Ele avisa enquanto houver traduções não revisadas.

**Teste sem APIs:** `jev-eval --config configs/smoke.yaml smoke` ou `jev-eval --config configs/default.yaml run --mock`. Os mocks são determinísticos e servem só para testar o pipeline.

## Tradução PT-BR

**Execução final: tradução da USP.** Usamos [`c2d-usp/banking77-pt-br`](https://huggingface.co/datasets/c2d-usp/banking77-pt-br) (Dimant et al., 2025, CC BY 4.0). Segundo o cartão do dataset, a tradução foi feita com GPT (`gpt 4o-mini`, e `gpt 3.5-turbo` em 3 frases do teste) e depois revisada por humanos, e a coluna `alterado` marca as frases corrigidas nessa revisão. O dataset não traz o texto em inglês nem IDs, então o `scripts/build_usp_data.py`:
1. baixa as duas fontes em revisões fixadas (`PolyAI/banking77` `1846984`, `c2d-usp/banking77-pt-br` `85f51c4`);
2. descarta as 2 linhas vazias no fim do teste da USP (3.082 → 3.080) e recusa linhas vazias em outra posição;
3. confere que a ordem é a mesma: mapeando os nomes PT para IDs pela tabela do cartão do dataset, a intenção coincide nas 3.080 posições. Além disso, o comprimento das frases EN e PT tem correlação de 0,972 item a item (0,118 com as frases embaralhadas dentro de cada intenção). Se qualquer checagem falhar, o script para;
4. casa cada um dos 462 exemplos da amostra com o seu índice no teste original pelo texto em inglês (sem duplicatas no teste), confere o rótulo e copia a frase PT-BR;
5. usa os nomes PT das intenções, com `_` trocado por espaço, como descrições PT-BR das opções, e confere que as 77 são distintas.

Saídas: `data/usp/sample_usp.jsonl` (mesmo esquema de `data/sample.jsonl`), `data/usp/intents_usp.json`, `data/usp/alignment_usp.jsonl` (índice no teste, `alterado` e `modelo_tradutor` por exemplo; 28 dos 462 têm `alterado = 1`) e `data/usp/provenance.json`. O `data/pilot/` e os arquivos antigos de `data/` não são alterados.

**Preliminar: rascunho próprio (não revisado).** O preliminar usou um rascunho feito com modelo local via Ollama (`gemma4:12b-it-qat`) e glossário fixo (`scripts/translate_local.py`), montado em `data/pilot/` por `scripts/build_pilot_data.py`. As planilhas de revisão (`scripts/export_review_sheets.py`, `scripts/import_review_sheets.py`, `scripts/merge_translations.py`, `data/review/`) eram o fluxo de revisão previsto e não são mais usadas na execução final.

## Saídas de uma execução

| Arquivo | Para quê |
|---|---|
| `raw_predictions.jsonl` | Uma linha por (exemplo, idioma, sistema): previsão, confiança, acerto, resposta bruta (também nas respostas inválidas), `served_model`, `usage`, `confidence_sdk` (Jev), `input_hash`, `latency_s` (tempo da chamada, com as tentativas) |
| `run_metadata.json` | Proveniência de cada invocação de `run` |
| `metrics.json` | ECE, acurácia, IC 95% por bootstrap, confiança média, excesso de confiança, erros com confiança ≥ 0,9, ΔECE pareado (5000 reamostragens, seed 42), chamadas que falharam por sistema, e `jev_confidence_sdk_check` (ECE usando o `confidence` do SDK, só como checagem) |
| `figures/reliability.png` | Diagrama de confiabilidade (área ∝ tamanho do bin) + histograma de confiança, por idioma |
| `figures/ece_accuracy_ci.png` | ECE e acurácia por sistema e idioma, com IC 95% |
| `figures/delta_ece_bootstrap.png` | Distribuição bootstrap de ΔECE (teste de H1) |
| `figures/confidence_correct_vs_wrong.png` | Confiança em acertos e em erros |
| `figures/risk_coverage.png` | Exploratório: acurácia das respostas aceitas vs cobertura |
| `figures/top_confusions.png` | Pares (verdadeiro → previsto) mais confundidos |
| `tables/summary.csv` | Tabela principal (inclui `n_paired` e `n_failed_calls`) |
| `tables/robustness.csv` | Robustez (E2), sobre os itens pareados: ECE com 10, 15 e 20 bins uniformes e com 15 bins de massa igual, Brier top-1, AUROC (acerto × erro) e número de valores distintos de confiança. O Jev aparece também com o `confidence` do SDK, só como checagem |
| `tables/ablation.csv`, `ablation_deltas.csv` (só na ablação) | Jev, LLM v1 e LLM top-3 sobre os itens em que os três responderam: métricas e ΔECE pareado com IC 95% (top-3 − v1, Jev − top-3, Jev − v1) |
| `tables/reliability_bins.csv`, `high_confidence_errors.csv`, `confusion_pairs.csv`, `language_flips.csv` | Bins, erros com confiança alta, confusões e efeito da tradução (EN vs PT-BR) |

As figuras e tabelas usam só os exemplos em que os dois sistemas responderam, para que ECE_Jev e ECE_LLM sejam calculados sobre os mesmos itens.

## Método

- Banking77, split `test`, amostragem estratificada com seed 42: 6 exemplos por intenção × 77 = 462 (a proposta fala em "até 500").
- Mesmas mensagens e opções para os dois sistemas. As opções em PT-BR usam as descrições traduzidas.
- O LLM roda com temperatura 0 e prompt congelado (`llm-intent-json-v1`, em `src/jev_eval/prompts/llm_prompt.py`).
- O ECE usa bins uniformes congelados no YAML (`final` = 15).
- O IC 95% de ΔECE vem de bootstrap pareado por `example_id`, com 5000 reamostragens e seed 42.
- **Robustez (E2):** a métrica primária continua sendo o ECE com 15 bins uniformes. Os outros números são checagens e não viram hipótese. Nos bins de massa igual, os itens empatados são divididos pela ordem original. O Jev devolve as probabilidades com 2 casas decimais, então há muitos empates, e `distinct_confidence_values` mostra quantos valores distintos existem.
- O estudo compara produtos e interfaces. Ele não permite concluir que RLCD seja superior a RLHF.

## Desvios do protocolo e limitações conhecidas

- **Preliminar:** PT-BR não revisado; Jev pelo alias `jev-latest`, sem registro da versão servida; confiança do Jev gravada a partir do campo `confidence` do SDK (corrigido na análise, ver acima); `num_ctx` padrão do Ollama numa máquina diferente.
- A execução final roda numa máquina diferente da do preliminar. As diferenças entre as duas não se devem só à revisão da tradução.
- **Tradução substituída.** A proposta previa uma tradução própria revisada pelo grupo. Na execução final usamos a tradução pública da USP, também revisada por humanos, porque ela é revisada de forma completa (e não por amostra), está documentada e citável, e qualquer pessoa pode reproduzir os insumos. A amostra, as intenções e o protocolo não mudam. As descrições PT-BR das opções passam a ser os nomes PT das intenções definidos pela USP.
- **Risco de contaminação.** A tradução está pública desde 2025, e o Banking77 em inglês desde 2020. O `ministral-3:14b` e o modelo por trás do Jev podem ter visto essas frases, ou os pares frase-intenção, no treino. Isso pode inflar a acurácia nos dois idiomas, e não temos como medir isso. Também não há garantia de que o efeito seja igual nos dois sistemas. A comparação de ECE continua pareada sobre os mesmos itens, mas os números absolutos devem ser lidos com isso em mente.
- A tradução da USP foi feita com modelos GPT. Isso é de uma família diferente do LLM avaliado e não deveria favorecê-lo.
- O estudo avalia o Banking77 **traduzido**, e não a linguagem bancária brasileira nativa.

## AI disclosure (rascunho)

Ferramentas de IA generativa foram usadas para auxiliar a implementação do pipeline e a revisão de código. As decisões metodológicas, a revisão das traduções, a execução das avaliações e a interpretação dos resultados são responsabilidade do grupo.
