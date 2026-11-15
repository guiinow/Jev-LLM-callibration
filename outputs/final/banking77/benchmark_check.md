# Conferência com o benchmark publicado (jev, Banking77, inglês)

Nosso jev em C1 (EN + instruções EN), 462 exemplos, contra Deußer et al. (2026), Tabelas 2, 10 e 11 (Banking77, teste completo, n = 3.076).

| Métrica | Nosso | IC 95% | Publicado | Publicado dentro do IC? |
|---|---|---|---|---|
| Acurácia | 0.784 | [0.745, 0.823] | 0.797 | sim |
| ECE (15 faixas) | 0.097 | [0.073, 0.132] | 0.087 | sim |
| Acurácia nos 80% mais confiantes | 0.886 | [0.846, 0.923] | 0.888 | sim |
| Acurácia nos 50% mais confiantes | 0.954 | [0.923, 0.978] | 0.963 | sim |
| AURC | 0.075 | [0.052, 0.103] | 0.061 | sim |

**Bate:** todos os valores publicados caem dentro dos nossos intervalos.

Limites: com poucos exemplos os intervalos são largos, então este teste só detecta diferenças grandes. O benchmark usa o teste completo (`mteb/banking77`, 3.076 exemplos) e a nossa amostra é um subconjunto estratificado do teste do PolyAI (3.080).
