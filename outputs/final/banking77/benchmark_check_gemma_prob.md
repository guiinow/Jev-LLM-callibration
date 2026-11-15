# Conferência com o benchmark publicado (gemma_prob, Banking77, inglês)

Nosso gemma_prob em C1 (EN + instruções EN), 462 exemplos, contra Deußer et al. (2026), Tabelas 2, 10 e 11 (Banking77, teste completo, n = 3.076).

| Métrica | Nosso | IC 95% | Publicado | Publicado dentro do IC? |
|---|---|---|---|---|
| Acurácia | 0.673 | [0.630, 0.714] | 0.670 | sim |
| ECE (15 faixas) | 0.247 | [0.211, 0.288] | 0.267 | sim |

**Bate:** todos os valores publicados caem dentro dos nossos intervalos.

Limites: com poucos exemplos os intervalos são largos, então este teste só detecta diferenças grandes. O benchmark usa o teste completo (`mteb/banking77`, 3.076 exemplos) e a nossa amostra é um subconjunto estratificado do teste do PolyAI (3.080).
