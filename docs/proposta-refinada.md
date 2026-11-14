# Proposta refinada

Atualiza a [proposta inicial](proposta-inicial.md) do projeto ("Confiança em classificação de intenções bancárias em PT-BR: uma avaliação de JEV e LLMs generativos"). Tudo o que não aparece aqui continua como na proposta inicial.

## Em uma frase

Quando a mensagem passa do inglês para o português, a confiança de um modelo continua confiável? E isso depende do modelo ou de *como* a confiança é obtida?

## Contexto mínimo

- **JEV** é um modelo da TypeSafe AI que não gera texto: escolhe entre opções fixas e devolve a probabilidade de cada uma.
- **Confiança confiável** (calibração) significa que, quando o modelo diz 90%, ele acerta cerca de 90% das vezes. Medimos com o **ECE**: quanto menor, melhor.
- **O benchmark** (Deußer et al., 2026, "Evaluating and Benchmarking the System One Model Jev") já avaliou o JEV em 37 datasets e o comparou com Qwen3.8-27B e Gemma-4-E4B, lidos pelas probabilidades exatas das opções. Ele não inclui uma avaliação específica de exemplos traduzidos para PT-BR.

## O que mudou em relação à proposta inicial

| Tema | Antes | Agora | Por quê |
|---|---|---|---|
| Pergunta | Jev vs LLM em PT-BR | A mesma (H1), mais o efeito EN→PT e a separação entre modelo e forma de medir a confiança | Só comparar Jev com um LLM não diz *por que* um ganha |
| Datasets | Só Banking77 | Banking77, CLINC150 e SIB-200 | Um dataset não permite generalizar. São três tarefas de classificação com opções fixas, todas do benchmark e as mais simples de usar |
| LLMs | Um LLM, com confiança dita por escrito | Gemma e Qwen (os do benchmark), cada um com **duas** confianças: probabilidades das opções e confiança escrita. O Ministral já executado fica como extra | O H1 mistura "qual modelo" com "como a confiança é lida" |
| Inglês | Exploratório | Resultado de primeira classe: mudança de ECE de EN para PT em cada sistema | É a contribuição nova sobre o benchmark |
| Seleção de respostas | Fora do escopo | Incluída: acurácia ao aceitar só as respostas mais confiantes | Mostra se a confiança é *útil*, não só calibrada |
| Idiomas | EN e PT-BR | Três condições (abaixo) | Separa o efeito do conteúdo do efeito das instruções |
| Traduções | Todas traduzidas e revisadas pelo grupo | Banking77: tradução pública da USP, revisada por humanos. Demais datasets: versões públicas, **sem revisão nossa** | Escala; o risco fica declarado |

## O que não mudou

H1 e ECE com 15 faixas como métrica principal, bootstrap pareado por exemplo com IC de 95%, uso zero-shot, e o escopo: comparamos **sistemas completos**, não arquiteturas, então não concluímos causa.

## Perguntas

| # | Pergunta | Papel |
|---|---|---|
| **Q1 (H1)** | Em PT-BR, o ECE do Jev é menor que o do LLM? | Principal, pré-definida |
| **Q2** | O ECE muda de EN para PT? Em quais sistemas e datasets? | Central |
| **Q3** | A confiança ainda serve para decidir quando responder? | Central |

Q1 e Q2 são respondidas separando modelo de forma de medir:

| | Jev | Gemma | Qwen |
|---|---|---|---|
| Probabilidades das opções | ✅ | ✅ | ✅ |
| Confiança escrita (0 a 1) | — | ✅ | ✅ |

## Condições de idioma

| | Conteúdo | Instruções e opções | Uso |
|---|---|---|---|
| **C1** | EN | EN | Linha de base |
| **C2** | PT | EN | Isola a mudança do conteúdo. Medida do efeito EN→PT (C2 menos C1) |
| **C3** | PT | PT | Desenho da proposta inicial. Teste do H1 |

## Dados

| Dataset | Tarefa | Amostra | Versão em português |
|---|---|---|---|
| Banking77 | 77 intenções bancárias | 462 exemplos (6 por intenção) | [c2d-usp/banking77-pt-br](https://huggingface.co/datasets/c2d-usp/banking77-pt-br) |
| CLINC150 | 150 intenções | cerca de 500, estratificado | [tech4humans/clinc_oos_pt](https://huggingface.co/datasets/tech4humans/clinc_oos_pt), que traz o inglês na mesma linha |
| SIB-200 | 7 tópicos | o teste inteiro em português (204 frases) | [Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200), configurações `eng_Latn` e `por_Latn` |

**Tamanho da amostra.** Seguimos o benchmark no desenho (split de teste, ECE com 15 faixas, opções fixas), mas não no tamanho: ele usa o teste completo, e rodar isso nos modelos locais é inviável. Usamos cerca de 500 exemplos por dataset, sorteados com seed 42 e estratificados por classe. Os intervalos de confiança ficam mais largos, e isso é declarado como limitação.

**Fora por agora:** Belebele (possível quarto dataset, se sobrar tempo), MMLU, IMDB, BoolQ e GoEmotions. Exigem alternativas diferentes a cada item, pareamento difícil de provar ou métricas de outro tipo (BoolQ e GoEmotions).

**Pareamento EN↔PT sem revisão manual:** alinhamos por ID ou texto original (ordem só se documentada), checamos contagens, IDs duplicados ou ausentes, campos vazios, rótulos divergentes e opções, e mantemos só pares comprovadamente alinhados. O que for excluído e o motivo ficam num manifesto. Se o alinhamento de um dataset não puder ser comprovado, ele sai da análise pareada principal.

## Métricas e estatística

- **Principal:** ECE com 15 faixas, por dataset, condição e sistema. Mudança EN→PT: `ECE(PT) − ECE(EN)`.
- **Secundárias:** acurácia; diagramas de confiabilidade; ECE com 10, 20 e 30 faixas; acurácia e risco ao aceitar só 50% e 80% das respostas mais confiantes, e a área sob a curva risco × cobertura; Brier e NLL onde há distribuição completa; quantas respostas mudam de EN para PT.
- **Intervalos:** bootstrap pareado por exemplo (EN e PT do mesmo item ficam juntos). Resultados **por dataset**; o resumo entre datasets é a média com pesos iguais, sem juntar as previsões.
- Um intervalo que inclui zero **não prova** que os idiomas são equivalentes.

## Reprodução e validação

Antes de olhar o PT, conferimos que nossa execução em inglês reproduz o benchmark ([harness](https://github.com/AppliedMachineLearning-Lab/jev-benchmarking) e [respostas publicadas](https://doi.org/10.5281/zenodo.23039006)), por exemplo Banking77 com acurácia de 79,7% e ECE de 0,087 no Jev. Para o SIB-200 o número publicado é a média de todos os idiomas, então não serve de referência para o inglês isolado. Prompts congelados antes da execução principal; versões de modelos fixadas.

## Limitações

- O CLINC150 foi traduzido por LLM e não foi revisado por nós; o SIB-200 vem do FLORES-200, traduzido por tradutores profissionais; só o Banking77 foi revisado por humanos para este uso. Diferenças podem vir de erros de tradução, não só do modelo.
- Dados traduzidos não representam mensagens escritas originalmente em português.
- Modelos, formas de medir confiança e números de classes diferem. Conclusões não são causais sobre arquiteturas.
- O ECE depende das faixas e do tamanho da amostra; melhor calibração não implica melhor seleção.
- Possível contaminação dos benchmarks nos modelos.

## Decisões em aberto

1. **Condição do H1:** C3, como na proposta inicial. Confirmar antes de ver os resultados.
2. **Qwen:** o Qwen3.8-27B em precisão completa não cabe no Mac de 36 GB usado até agora; quantizar quebraria a comparação com o benchmark. Gemma é obrigatório, Qwen é desejável.
3. **CLINC150:** incluir a classe fora de escopo (151 opções, como no benchmark) ou só as 150 intenções? Para a C3, as descrições das 150 intenções em PT ainda precisam ser criadas.
4. **Belebele como quarto dataset**, se sobrar tempo. Exige alternativas diferentes a cada item; propomos tratá-las como conteúdo (PT em C2 e C3, EN em C1).

O plano de entregas em PRs está em [plano_do_artigo_e_prs.md](plano_do_artigo_e_prs.md).

## Referências

- Casanueva et al. (2020). *Efficient Intent Detection with Dual Sentence Encoders.* (Banking77)
- Deußer et al. (2026). *Evaluating and Benchmarking the System One Model Jev.* arXiv:2609.37647.
- Dimant et al. (2025). Tradução PT-BR do Banking77 (c2d-usp).
- Tian et al. (2023). *Just Ask for Calibration.* EMNLP.
- Xiong et al. (2024). *Can LLMs Express Their Uncertainty?* ICLR.
- Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
