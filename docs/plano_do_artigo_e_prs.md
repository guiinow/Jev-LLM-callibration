# O artigo e o plano de PRs

Este documento junta as duas ideias do projeto (a da `main` e o protocolo EN/PT) em **um único artigo** e diz, em linguagem simples, o que cada PR muda. Ele substitui o mapeamento anterior de PRs.

## 1. A ideia em uma frase

> **Quando a mensagem passa do inglês para o português, a confiança de um modelo continua confiável? E isso depende do modelo ou de *como* a confiança é obtida?**

"Confiança confiável" quer dizer: quando o modelo diz 90%, ele acerta cerca de 90% das vezes. Medimos isso com o **ECE** (quanto menor, melhor alinhada está a confiança com o acerto).

## 2. Três perguntas, uma história

| # | Pergunta | De onde vem | Em português claro |
|---|---|---|---|
| **Q1** | Em PT-BR, o Jev tem ECE menor que o LLM? | H1 da `main` (proposta) | Qual sistema tem a confiança mais honesta em português? |
| **Q2** | O ECE muda de EN para PT? Em quais sistemas? | RQ1 do protocolo | Traduzir a mensagem piora a confiança? De quanto? |
| **Q3** | A confiança ainda serve para decidir quando responder? | RQ3 do protocolo | Dá para aceitar só as respostas confiantes e errar menos? |

A **Q1 é a pergunta principal** (pré-definida, uma hipótese). Q2 e Q3 são o que torna o trabalho mais forte do que a comparação simples.

## 3. O que torna a contribuição clara e nova

Hoje o H1 compara o Jev (que dá probabilidades das opções) com um LLM que **diz a confiança por escrito**. Se o Jev ganhar, não sabemos o motivo: foi o modelo ou foi a forma de medir a confiança?

A solução vem do protocolo: **cada LLM é medido das duas formas**, pelas probabilidades das opções e pela confiança dita por escrito. Assim separamos "qual modelo" de "como a confiança é lida".

| | Jev | Gemma | Qwen |
|---|---|---|---|
| Probabilidades das opções | ✅ | ✅ | ✅ |
| Confiança dita por escrito | — | ✅ | ✅ |

Cada célula é medida em **inglês (C1)** e em **português (C2 e C3)**.

Contribuição em uma linha: *uma avaliação reproduzível de como a mudança EN→PT afeta a confiança, e de quanto isso depende do modelo ou da forma de ler a confiança.*

## 4. O que já está pronto

- Banking77 em EN e PT-BR (tradução da USP, revisada por humanos), 462 exemplos.
- Jev e um LLM local (Ministral 14B com confiança dita por escrito), nas condições C1, C2 e C3.
- ECE, acurácia, intervalo de confiança do Jev−LLM, gráficos, registro de versões de cada execução, ablação "3 palpites".

Falta, em resumo: comparar EN com PT dentro de cada sistema, medir a seleção de respostas, ler a confiança de LLMs pelas probabilidades, e sair do Banking77.

## 5. Plano de PRs

Cada PR faz **uma coisa** e tem um critério de "pronto" que qualquer pessoa do grupo consegue conferir.

### Fase A: fechar o artigo com o que já existe

**A1 · Registrar o desenho do artigo**
- **O que muda:** só texto. O README passa a dizer: a pergunta, as hipóteses, qual condição testa cada uma e o que é principal ou secundário.
- **Para quê:** todos escrevem e falam a mesma coisa. Decisões ficam fixadas *antes* de ver resultados.
- **Pronto quando:** o README tem a seção "Perguntas e hipóteses" com as decisões da seção 7 deste documento.

**A2 · Comparar EN e PT em cada sistema** (responde Q2)
- **O que muda:** a análise passa a calcular, para cada sistema, quanto o ECE e a acurácia mudam de EN para PT, com intervalo de confiança. Também conta quantas respostas mudam de uma língua para outra.
- **Para quê:** é o resultado central da Q2, usando as execuções que o grupo já fez. Nenhuma chamada nova à API.
- **Pronto quando:** `metrics.json` traz, por sistema, a mudança de ECE (PT menos EN) com intervalo de 95%.
- **Depende de:** A1.

**A3 · Medir se a confiança ajuda a escolher quando responder** (responde Q3)
- **O que muda:** novas métricas: acurácia ao aceitar só as respostas mais confiantes (50% e 80% do total), e a área sob a curva risco × cobertura. Também mostra o ECE com 10, 20 e 30 faixas, para provar que o resultado não depende da escolha de 15.
- **Para quê:** mostra se a confiança é *útil*, não só bem calibrada.
- **Pronto quando:** tabela por sistema e condição com essas métricas, e uma linha dizendo se a conclusão muda com o número de faixas.
- **Depende de:** A1.

**A4 · Conferir nosso resultado em inglês contra o benchmark publicado**
- **O que muda:** um script compara nossa execução em EN do Jev no Banking77 com o publicado (acurácia 79,7%, ECE 0,087).
- **Para quê:** mostra que nossa configuração funciona antes de confiar nos resultados em PT.
- **Pronto quando:** relatório curto: "bate" ou "difere em X, por causa de Y".
- **Depende de:** nada.

### Fase B: separar "modelo" de "forma de medir a confiança"

**B1 · Gemma e Qwen, com as duas formas de confiança**
- **O que muda:** novos modelos que rodam localmente e devolvem (a) a probabilidade de cada opção e (b) uma confiança dita por escrito. Os dois usam o mesmo modelo e a mesma versão.
- **Para quê:** é o que permite separar o efeito do modelo do efeito da forma de medir.
- **Pronto quando:** uma execução de teste produz as duas confianças para os mesmos exemplos, e o resultado em EN fica coerente com o benchmark.
- **Depende de:** A1.

**B2 · A tabela central: modelo × forma de medir × idioma**
- **O que muda:** análise que, dentro de cada LLM, compara a mudança de ECE pelas probabilidades com a mudança pela confiança escrita, e compara o Jev com o LLM *na mesma forma de medir*.
- **Para quê:** responde se o resultado do H1 vem do modelo ou da forma de medir.
- **Pronto quando:** uma tabela com Jev, Gemma e Qwen nas duas formas, em EN e PT, com intervalos.
- **Depende de:** A2, B1.

### Fase C: mostrar que não é só o Banking77

**C1 · Pareamento automático EN↔PT e três datasets de classificação**
- **O que muda:** uma ferramenta que pareia EN com PT por ID ou texto, descarta o que não bate e **lista o que foi descartado e por quê**. Entram CLINC150, SIB-200 e IMDB.
- **Para quê:** garante que comparamos o mesmo exemplo nas duas línguas, sem inferir nada que não dá para provar.
- **Pronto quando:** cada dataset tem um relatório de pareamento (quantos entraram, quantos saíram, motivos) e roda no mesmo comando.
- **Depende de:** A1.

**C2 · Datasets de múltipla escolha e resumo entre datasets**
- **O que muda:** entram Belebele e MMLU. Os resultados passam a ser mostrados **por dataset** e resumidos com média de pesos iguais, sem juntar todas as previsões.
- **Para quê:** mostra se o padrão se repete em outras tarefas.
- **Pronto quando:** tabela por dataset e a média dos datasets, cada um com intervalo.
- **Depende de:** C1, B2.

**D1 · (opcional) BoolQ e GoEmotions**
- **O que muda:** duas tarefas de tipo diferente (sim/não e várias emoções por frase), cada uma com suas métricas.
- **Para quê:** generalização extra. Se faltar tempo, vira "trabalho futuro" sem prejudicar o artigo.
- **Pronto quando:** resultados separados dos datasets de escolha, nunca misturados com eles.
- **Depende de:** C1.

### Fase E: fechar o artigo

**E1 · Figuras e tabelas do artigo**
- **O que muda:** uma figura por pergunta (Q1, Q2, Q3) e as tabelas principais, no formato que cabe em 2 páginas.
- **Para quê:** o leitor entende cada resultado só olhando a figura.
- **Pronto quando:** três figuras e duas tabelas geradas por um comando.
- **Depende de:** A2, A3 e B2.

**E2 · Reprodutibilidade, limitações e declarações**
- **O que muda:** README com os comandos exatos, lista das limitações (traduções, contaminação, comparação entre sistemas completos), declaração de uso de IA e contribuição de cada integrante.
- **Para quê:** a disciplina exige reprodução, declaração de IA e contribuição individual.
- **Pronto quando:** outra pessoa reproduz a tabela central seguindo só o README.
- **Depende de:** tudo.

## 6. Ordem e o que o artigo vira em cada fase

```
A1 ──► A2 ──┐
  ├──► A3 ──┼──► E1 ──► E2
  └──► B1 ──┴► B2 ──► C2
A4 (independente)   C1 (depois de A1) ──► C2, D1
```

| Ao terminar | O artigo já tem |
|---|---|
| **A** | Q1, Q2 e Q3 no Banking77, com Jev × Ministral. Já é um artigo completo. |
| **A + B** | Separa modelo de forma de medir confiança. É o ponto forte. |
| **A + B + C** | Mostra que vale para outras tarefas (6 datasets). |

Dá para parar depois de qualquer fase com um trabalho coerente.

## 7. Decisões do grupo (com recomendação)

| Decisão | Recomendação |
|---|---|
| Em qual condição o H1 (Q1) é testado? | **C3**, o desenho descrito na proposta (seção 4.1). Decidir antes de ver os resultados. |
| Qual contraste mede o efeito do idioma (Q2)? | **C2 menos C1**, porque só o conteúdo muda. C3 menos C1 fica como secundário. |
| Quais LLMs? | **Gemma-4-E4B** como obrigatório (cabe num Mac de 36 GB). **Qwen3.8-27B** como desejável: em precisão completa não cabe nesse Mac, e quantizar quebraria a comparação com o benchmark. Manter o Ministral como extra, já que foi executado. |
| Como medir as duas confianças do mesmo LLM? | Mesmo programa e mesmos pesos para as duas, senão a diferença pode vir da quantização e não da forma de medir. |
| Quantos datasets? | **6 de escolha** (Banking77, CLINC150, SIB-200, Belebele, MMLU, IMDB). BoolQ e GoEmotions só se sobrar tempo. |

## 8. Riscos para dizer no artigo

- A tradução do Banking77 foi revisada por humanos; as dos outros datasets, não. Conclusões entre datasets precisam desse aviso.
- Com 462 exemplos, os intervalos de confiança serão largos. Um intervalo que inclui zero **não prova** que os idiomas são equivalentes.
- Comparamos sistemas completos, não arquiteturas. Não dá para concluir causa.
- Prazos: a especificação da disciplina marca o relatório para 13/11 e as apresentações para 16 a 25/11. A fase A é a que cabe antes da apresentação; B e C são a versão estendida.
