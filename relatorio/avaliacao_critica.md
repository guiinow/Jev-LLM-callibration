# Avaliação crítica do trabalho e sugestões

Base: enunciado (`docs/especificacao-do-curso.md`), propostas, código em `src/jev_eval/` e a execução final no branch `resultados/final-l40s` (commit `eebd720`: `outputs/final/` e `outputs/ablation_topk/`). Todo número citado aqui vem desses arquivos.

## 1. Inconsistências e lacunas encontradas (corrigir antes de entregar)

| # | Problema | Onde | Impacto | Ação sugerida |
|---|---|---|---|---|
| 1 | **Os resultados finais não estão na `main`.** `outputs/final/` só existe no branch `resultados/final-l40s`. O enunciado pede saídas que sustentem os números no commit entregue. | git | Alto (reprodutibilidade/avaliação) | Fazer merge do branch na `main` e informar esse commit na entrega. |
| 2 | **README diz que a execução final rodou num MacBook M3 Max (36 GB, Metal)**, mas o `run_metadata.json` registra Linux x86_64 com NVIDIA L40S (CUDA 12, Ollama recompilado). | `README.md`, tabela "Execuções" | Médio | Atualizar o README com a máquina real e os desvios de ambiente (torch 2.10 em vez do 2.15 fixado; Ollama 0.40.16 recompilado). |
| 3 | **A análise de erros categorizada (E3) não foi feita.** `error_annotation_C3.csv` tem 221 linhas e a coluna `category` está vazia em todas. O próprio enunciado cita "categorized error analysis" como o que se espera de um trabalho desse tipo. | `outputs/final/banking77/error_analysis/` | **Alto** (critério "Analysis", 20%) | Preencher a planilha (dá para dividir entre os 4 integrantes) e rodar `error-summary`. É o ganho mais barato. |
| 4 | **Qwen3.8-27B não foi executado**, embora a proposta refinada e o README o tratem como parte do desenho (Jev × Gemma × Qwen). | configs, README | Médio | Rodar `open_models_qwen.yaml` numa GPU de ≥ 80 GB (a L40S tem 48 GB, insuficiente em bf16), ou declarar a ausência como desvio. |
| 5 | **A condição do H1 está marcada como "a confirmar antes de ver os resultados"** no README e na proposta, e não há registro de que foi confirmada antes. | README, proposta refinada | Médio (credibilidade da pré-definição) | Registrar no README em que commit/data a C3 foi fixada. Felizmente o H1 vale em C1, C2 e C3 no Banking77 e no CLINC150, então a conclusão não depende da escolha (no SIB-200, C1 é inconclusivo). |
| 6 | **Rótulos desatualizados na figura de confiabilidade**: o painel C2 aparece como "PT (principal)" e C3 como "PT localizado", contradizendo a decisão (H1 em C3, C2 como contraste de Q2). | `src/jev_eval/eval/figures.py` → `figures/reliability.png` | Baixo | Corrigir os rótulos se a figura for usada na apresentação. |
| 7 | **Números diferentes para a "mesma" célula** conforme a tabela: ECE do Jev no Banking77/C1 = 0,0933 (`summary.csv`, pareado com o Ministral, n = 458), 0,0972 (`readout_cells.csv`, todas as válidas, n = 462). O `benchmark_check.md` usa 0,097. | tabelas | Baixo, mas confunde a banca | Dizer sempre qual conjunto de itens é usado. O relatório usa `readout_cells` na figura e contrastes pareados nas tabelas. |
| 8 | **O IC bootstrap percentil às vezes não contém a estimativa pontual** (Jev SIB-200/C2: 0,0221 com IC [0,0227; 0,0705]; Gemma verb. SIB-200/C3: 0,0184 [0,0143; 0,0671]). Isso acontece porque o ECE é enviesado para cima em amostras pequenas. | `metrics.py` | Médio (rigor estatístico) | Usar IC BCa ou bootstrap com correção de viés, ou reportar também um ECE "debiased" (p.ex. o estimador de Kumar et al., 2019). |
| 9 | **Ministral foi quantizado (Q4_K_M)**, mas a quantização não aparece como limitação no README. Ela é justamente o argumento usado para não quantizar o Qwen. | `run_metadata.json` | Baixo | Declarar. |
| 10 | **Jev 1.14.0 vs. 1.13.0 no benchmark.** A validação "bate" dentro dos ICs, mas a versão é outra. | benchmark_check | Baixo | Declarar (já está no relatório). |
| 11 | **O Modelfile herdou `temperature 0.15`** do modelo base (aparece no `run_metadata`). O cliente passa `temperature=0` em cada requisição, o que deve prevalecer, mas isso não está documentado. | Ollama | Baixo | Fixar `PARAMETER temperature 0` no Modelfile. |
| 12 | **Rascunho de declaração de IA vago** ("auxiliar a implementação"). O enunciado pede ferramenta, componente e extensão. A tradução dos nomes das intenções do CLINC150 por Claude e a redação deste relatório também precisam ser declaradas. | README | Médio (exigência formal) | Usar a declaração do `main.tex`, completada pelo grupo. |
| 13 | **Declaração de contribuições ausente.** | — | Exigência formal | Preencher no `main.tex` (deixei como placeholder; não dá para inferir dos commits). |
| 14 | Enunciado em inglês e relatório em português. | — | A confirmar | Confirmar com os professores se o relatório pode ser em PT. |

## 2. Fragilidades metodológicas

1. **O contraste "forma de leitura" no Gemma tem confusão.** `gemma_prob` usa o prompt de códigos (`*-codes-v1`, "responda só o código") e `gemma_verb` usa o prompt JSON. Mudam a leitura **e** o prompt ao mesmo tempo, e a acurácia também muda (Banking77/C1: 0,673 vs. 0,697). O resultado "verbalizado mais bem calibrado que as probabilidades" contraria a literatura usual e merece um controle: ler as probabilidades no **mesmo** prompt JSON, isto é, a probabilidade do token da intenção, ou da sequência inteira, logo após `"intent": "`.
2. **Comparações múltiplas em Q2.** São 24 contrastes EN→PT (4 sistemas × 3 datasets × 2 condições) sem correção. Os 3 "significativos" (~12%) estão perto do esperado por acaso (5% de 24 ≈ 1,2). Reportar Holm/BH ou tratar Q2 como estimação, não como teste.
3. **Ausência de equivalência.** A conclusão "a tradução não piora a calibração" exige um teste de equivalência (TOST) com uma margem definida antes, por exemplo |ΔECE| < 0,03. Sem isso, só dá para dizer "não detectado".
4. **Q3 sem contraste pareado.** A AURC do Jev e a do Ministral são comparadas por ICs separados. O código já faz bootstrap pareado do ΔECE e pode fazer o mesmo para ΔAURC. No Banking77/C3 os ICs quase se tocam (Jev [0,054; 0,106] vs. Ministral [0,105; 0,178]).
5. **ECE do Ministral é degenerado.** Com 5 a 8 valores distintos, todos ≥ 0,85, o ECE fica idêntico com 10/15/20/30 faixas e igual ao excesso de confiança médio. A análise de sensibilidade às faixas é trivialmente "estável" para ele. Vale dizer isso explicitamente e complementar com Brier, AUROC e NLL (já calculados em parte).
6. **Gemma fora de Q3 e da robustez.** `selective.csv` e `robustness.csv` só têm Jev e Ministral. Estender ao Gemma é só rodar a análise (as respostas já existem).
7. **Amostras pequenas.** 204 a 462 itens. O benchmark usa os testes completos (Banking77: 3.076). Para o Jev o custo é baixo (o benchmark fez 346 mil chamadas por menos de US$ 10); rodar o teste completo do Banking77 com Jev e Gemma eliminaria boa parte da incerteza.
8. **Qualidade da tradução como variável não controlada.** O CLINC150 é tradução automática não revisada (incluindo os nomes das intenções, que entram em C3). Diferenças entre C2 e C3 no CLINC podem vir dos nomes traduzidos por LLM.

## 3. Experimentos e análises adicionais (por custo/benefício)

**Baratos (sem novas chamadas, só análise):**
- Preencher e resumir a planilha de erros (E3) e cruzar as categorias com a confiança: por exemplo, quantos erros com confiança ≥ 0,9 são `intencoes_proximas` ou `traducao`.
- Q3 e robustez para o Gemma; ΔAURC pareado; correção de Holm em Q2; TOST de equivalência para o ΔECE EN→PT.
- ECE com correção de viés e IC BCa.
- Mostrar o "language flip" com IC: o Jev muda a previsão em 2–7% dos itens, contra 8–13% do Ministral e 6–23% do Gemma. É um resultado interessante de **estabilidade** que hoje não tem incerteza associada.
- Calibração pós-hoc (temperature scaling ou Platt) ajustada em C1 e aplicada em C2/C3. Dá para fazer nas probabilidades do Jev e do Gemma prob. Responde se a calibração ajustada em inglês transfere para o português, o que estende Q2 diretamente.

**Moderados (novas chamadas):**
- Controle do item 2.1: probabilidades do Gemma no prompt JSON.
- Teste completo do Banking77 com Jev (e Gemma) em C1/C3.
- Incluir a classe fora de escopo do CLINC150 (151 opções, como no benchmark). Isso permite comparar com o publicado e testar se a confiança sinaliza fora de escopo.
- Ablação top-k também em C1 e para o Gemma verb.

**Maiores:**
- Qwen3.8-27B (GPU ≥ 80 GB).
- Um conjunto **nativo** em PT-BR, por exemplo mensagens reais de atendimento anonimizadas ou um dataset de intenções originalmente em português, para separar "tradução" de "idioma". Hoje todas as conclusões valem para português traduzido.
- Revisar uma amostra das traduções do CLINC150 (por exemplo, 100 itens) para estimar a taxa de erro de tradução.

## 4. Pontos fortes (para destacar na apresentação)

- Desenho 2 × 2 que separa **modelo** de **forma de leitura** e **conteúdo** de **instruções** (C1/C2/C3): é a contribuição mais original em relação ao benchmark.
- Hipótese pré-definida, métrica primária e número de faixas fixados antes, bootstrap pareado, resumo entre datasets sem juntar previsões.
- Validação contra o benchmark publicado antes de olhar o PT (todos os valores publicados caem nos ICs).
- Engenharia reprodutível: prompts congelados com *fingerprint* testado, `input_hash` por resposta, retomada segura, metadados por execução, testes automatizados, manifestos de pareamento EN↔PT.
- Resultado negativo/inconclusivo bem tratado em Q2 ("não detectado" ≠ "equivalente").

## 5. Sobre o relatório gerado (`relatorio/main.tex`)

- Formato ACM sigconf, em português, seguindo a estrutura sugerida no enunciado. **Não foi compilado aqui** (não há LaTeX instalado nesta máquina). O texto tem cerca de 1.450 palavras, mais uma figura e duas tabelas, no limite de 2 páginas. Se passar, os primeiros cortes sugeridos são o parágrafo da ablação e os detalhes de validação.
- A figura `figs/ece_datasets.pdf` é gerada por `relatorio/make_figure.py` a partir dos `readout_cells.csv`. Nenhum número novo foi calculado: todos os valores do relatório vêm das tabelas da execução final.
- Placeholders a preencher: contribuições individuais e declaração de uso de IA (marcadas com `[A PREENCHER]` / `[REVISAR]`).
