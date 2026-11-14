# PROPOSTA DE PESQUISA ACADÊMICA

**Título provisório:** Confiança em classificação de intenções bancárias em português brasileiro: uma avaliação de JEV e LLMs generativos

## 1. Resumo

Modelos de linguagem são cada vez mais usados para tomar decisões simples, como classificar mensagens de clientes e encaminhá-las a diferentes serviços. Para automatizar essas decisões com segurança, não basta acertar: é importante que a confiança reportada pelo modelo corresponda à sua frequência real de acerto.

Este projeto propõe comparar a calibração de confiança do JEV, um modelo de decisão que retorna probabilidades para opções predefinidas, com a de um LLM generativo solicitado a escolher uma intenção e informar sua confiança. A avaliação será feita em exemplos do conjunto Banking77 traduzidos e revisados em português brasileiro. O estudo terá uma hipótese principal, uma métrica primária — Expected Calibration Error (ECE) — e uma comparação exploratória com os mesmos exemplos em inglês.

A contribuição é uma avaliação controlada de confiança em classificação de intenções bancárias em PT-BR. Os resultados compararão os sistemas testados; não serão interpretados como evidência de que uma técnica de treinamento ou arquitetura, isoladamente, causa melhor calibração.

## 2. Problema e motivação

Em uma aplicação automatizada, um modelo pode escolher uma categoria incorreta e ainda assim reportar alta confiança. Se decisões confiantes forem aceitas automaticamente, esse tipo de erro pode ser encaminhado sem revisão humana.

O JEV oferece probabilidades associadas às opções de uma decisão. Um LLM generativo, em contraste, pode ser solicitado a responder com uma categoria e uma estimativa verbal de confiança. Pesquisas anteriores estudaram a elicitação de confiança em LLMs e encontraram resultados que variam conforme a tarefa e a estratégia de prompting.

Um benchmark recente avaliou JEV em 37 conjuntos de dados, incluindo Banking77, e reportou para esse conjunto acurácia de 79,7% e ECE de 0,087. O mesmo estudo observa limitações em tarefas com rótulos próximos ou ruidosos e não inclui uma avaliação específica de exemplos traduzidos para PT-BR.

O Banking77 é adequado para um estudo delimitado porque contém exemplos de atendimento bancário distribuídos por categorias de intenção. O trabalho que o introduziu estudou sua aplicação à detecção de intenções.

Banking77 / Efficient Intent Detection with Dual Sentence Encoders: <https://www.alphaxiv.org/abs/2003.04807>

## 3. Pergunta e hipótese

**Pergunta de pesquisa:**

Em exemplos do Banking77 traduzidos para português brasileiro, como se compara a calibração das probabilidades do JEV com a confiança verbalizada por um LLM generativo ao classificar intenções bancárias?

**Hipótese principal — H1:**

No conjunto de exemplos em PT-BR, o JEV apresentará ECE menor que o LLM generativo solicitado a informar sua confiança na intenção escolhida.

A avaliação em inglês dos mesmos exemplos será exploratória: servirá para observar se o padrão se mantém no texto original, mas não constituirá uma segunda hipótese principal.

## 4. Metodologia

### 4.1 Dados

Será usado somente o Banking77, sem CLINC150 ou HWU64. A avaliação principal será feita em uma amostra fixa e estratificada de até 500 exemplos do conjunto de teste, buscando representar suas classes de intenção.

Os exemplos selecionados serão traduzidos para PT-BR e revisados por integrantes do grupo para preservar o significado e a intenção original. Os rótulos e suas descrições também serão traduzidos e revisados. As versões em inglês e português de cada exemplo permanecerão pareadas. Serão registrados os critérios de seleção, as traduções e quaisquer correções.

Importante: como os exemplos são traduzidos, o estudo avaliará a classificação de exemplos do Banking77 em PT-BR, não a linguagem financeira brasileira nativa.

### 4.2 Sistemas comparados

- JEV: receberá a mensagem e as opções de intenção, usando sua saída nativa de escolha e probabilidades.
- LLM generativo: receberá a mesma mensagem e as mesmas opções; deverá retornar a intenção escolhida e uma estimativa numérica de confiança.

Os dois sistemas serão avaliados sobre os mesmos itens, com os mesmos rótulos e descrições. Os prompts, parâmetros e identificadores exatos das versões dos modelos serão registrados. A confiança do LLM será sua estimativa verbalizada; a do JEV será obtida de sua saída probabilística para a opção escolhida. Essa diferença entre as formas de obter confiança será explicitamente considerada na interpretação dos resultados.

### 4.3 Métrica principal

A métrica primária será o Expected Calibration Error (ECE): quanto menor o ECE, mais alinhada está a confiança reportada com a taxa observada de acerto.

A acurácia será reportada como contexto para interpretar o ECE. Um diagrama de confiabilidade poderá ser usado para visualizar a relação entre confiança e acerto. A hipótese será considerada apoiada se a diferença pareada de ECE favorecer JEV e seu intervalo de confiança de 95% não incluir zero. Se o intervalo incluir zero, o resultado será tratado como inconclusivo para essa hipótese.

A comparação em inglês será apresentada separadamente e de forma exploratória. Não serão incluídos, como objetivos centrais, testes de ruído, detecção fora de distribuição, latência, vazão ou múltiplas métricas de seletividade.

## 5. Plano de execução

1. Confirmar acesso à API do JEV e selecionar o LLM generativo; registrar versões e condições de uso.
2. Fixar e documentar a amostra do Banking77 e a forma de amostragem.
3. Traduzir e revisar as mensagens e descrições das intenções.
4. Definir os prompts e executar um piloto pequeno para verificar o formato das respostas, extração das confianças e custo.
5. Congelar os prompts e a configuração antes da execução principal.
6. Rodar os dois sistemas nos mesmos exemplos, em inglês e PT-BR; guardar respostas brutas.
7. Calcular acurácia e ECE, estimar intervalos pareados de confiança e gerar o diagrama de confiabilidade.
8. Discutir resultados, erros, limitações e quaisquer desvios do protocolo.

## 6. Escopo, contribuição e limitações

O projeto testa uma pergunta específica: qual dos dois sistemas apresenta melhor calibração de confiança nessa tarefa, com essa amostra, essas versões e essas configurações? A contribuição é estender a avaliação de JEV no Banking77 para exemplos em PT-BR traduzidos e compará-lo com a confiança verbalizada por um LLM generativo.

O estudo não pretende provar que RLCD é superior a RLHF, pois compara produtos disponíveis e suas interfaces, não isola os métodos de treinamento. Também não generaliza automaticamente para outros idiomas, modelos, tarefas bancárias ou mensagens originalmente escritas no Brasil. O ECE depende da forma de agrupar as previsões; por isso, a configuração de bins será definida antes de avaliar o conjunto de teste e reportada junto aos resultados.

## 7. Referências principais

- Casanueva, I. et al. Efficient Intent Detection with Dual Sentence Encoders. <https://www.alphaxiv.org/abs/2003.04807>. 2020. Trabalho associado ao conjunto Banking77.
- Deußer, T. et al. Evaluating and Benchmarking the System One Model Jev. <https://www.alphaxiv.org/abs/2609.37647>. 2026. Avalia JEV em conjuntos públicos, incluindo Banking77, e reporta métricas de calibração.
- Tian, K. et al. Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback. <https://www.alphaxiv.org/abs/2305.14975>. EMNLP, 2023. Estuda estratégias para obter confiança verbalizada de LLMs.
- Xiong, M. et al. Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. <https://www.alphaxiv.org/abs/2306.13063>. ICLR, 2024. Avalia elicitação de confiança, calibração e previsão de falhas em diferentes tarefas.
- Guo, C. et al. On Calibration of Modern Neural Networks. ICML, 2017. Referência metodológica para calibração e ECE.