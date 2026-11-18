# Resultados

Tudo nesta pasta foi produzido pelo código do repositório, rodando de verdade. Nenhum número foi digitado à mão.

> **Atenção ao interpretar:** os experimentos usam o dataset **sintético** de motores (`sample_data/motor_industrial_sintetico.csv`), gerado por simulação com semente fixa. Eles mostram que o pipeline funciona e se comporta como esperado. **Não** representam desempenho em equipamentos reais; a validação com um dataset público está nos próximos passos.

## Conteúdo

| Arquivo | O que é |
|---|---|
| `classificacao_falha_24h.csv` | Métricas dos 4 classificadores: treino, validação cruzada (média e desvio), teste, baseline e tempos |
| `regressao_temperatura_rolamento.csv` | O mesmo para os 4 regressores |
| `resumo_execucao.json` | Configuração exata, vazamento detectado, distribuição das classes, testes estatísticos pareados e versões das bibliotecas |
| `experimentos_completos.json` | Registro completo de cada experimento (hiperparâmetros, matriz de confusão, importância de atributos, coeficientes etc.) |
| `testes_backend.txt` | Saída dos 22 testes automatizados (`pytest -v`) |
| `capturas/` | 15 capturas de tela da interface, tiradas num navegador automatizado durante o teste de ponta a ponta |

## Como reproduzir

```bash
cd backend
pip install -r requirements.txt
python -m scripts.reproduzir_resultados ../../resultados
```

O script chama a mesma API usada pela interface. Os números de classificação reproduzem **exatamente** os que aparecem nas capturas `12_comparacao_4_modelos.png` e `14_avaliacao_xgboost_tema_claro.png` (conferido: XGBoost com F1 0,26 e ROC AUC 0,753; AUC de 0,802, 0,586 e 0,794 para os outros três modelos).

## 1. Classificação: prever `falha_24h` (falha nas próximas 24 h)

**Configuração:** 6.000 linhas, separação estratificada 80/20 (4.800 treino / 1.200 teste), semente 42, validação cruzada estratificada de 5 folds, mediana para imputação, padronização, one-hot, hiperparâmetros padrão. A classe positiva (falha) representa **8,5%** das linhas.

**Vazamento detectado:** sozinha, a coluna `modo_falha` atingiu balanced accuracy = **1,000** no treino. Ela descreve o próprio evento de falha e foi removida antes do treino (armadilha colocada de propósito no dataset sintético).

| Modelo | F1 teste | F1 validação cruzada | Recall | Precision | ROC AUC | Accuracy | Accuracy do baseline |
|---|---|---|---|---|---|---|---|
| Regressão Logística | 0,202 | 0,166 ± 0,030 | 0,118 | 0,706 | 0,802 | 0,921 | 0,915 |
| Decision Tree | 0,239 | 0,205 ± 0,024 | 0,255 | 0,224 | 0,586 | 0,862 | 0,915 |
| Random Forest | 0,279 | 0,157 ± 0,023 | 0,167 | 0,850 | 0,794 | 0,927 | 0,915 |
| XGBoost | 0,260 | 0,189 ± 0,057 | 0,167 | 0,586 | 0,753 | 0,919 | 0,915 |

**Leitura:**
- A accuracy de todos fica perto da do baseline (0,915), que sempre prevê "sem falha". Accuracy não serve para escolher modelo aqui.
- O recall é baixo em todos (no máximo 0,255): com o limiar padrão de 0,5, a maior parte das falhas passa despercebida.
- Os valores de F1 no teste e na validação cruzada são diferentes. Por exemplo, o Random Forest tem o melhor F1 no teste (0,279), mas o pior na validação cruzada (0,157). Com 102 falhas no teste, o ranking entre modelos é instável.
- **Overfitting:** com hiperparâmetros padrão (árvores sem limite de profundidade), Decision Tree e Random Forest têm F1 = 1,000 no treino; o XGBoost tem 0,983. Na validação cruzada, todos ficam entre 0,16 e 0,21.
- **Matriz de confusão do Random Forest no teste:** 17 das 102 falhas detectadas, 85 não detectadas e 3 falsos alarmes (1.095 normais corretos).
- **Atributo que não generaliza:** entre os mais importantes do XGBoost aparecem `motor_id_MTR-09` e `motor_id_MTR-05`. O modelo aprendeu quais motores específicos falham mais, o que não vale para um motor novo. A próxima rodada de experimentos vai excluir `motor_id` para medir o efeito.

**Teste de significância pareado (F1, 5 folds, t-test com correção de Nadeau-Bengio):** nenhum dos 6 pares tem diferença significativa a 5%. Exemplo: Regressão Logística × Decision Tree tem p = **0,048** no t-test comum, mas p = **0,135** com a correção. Sem a correção, a conclusão seria (indevidamente) que a árvore é melhor.

Testes que **não** foram executados nesta configuração: limiar otimizado, `class_weight="balanced"` e `scale_pos_weight` automático, curva de aprendizado, importância por permutação. O app oferece todos eles; medir o impacto de cada um neste dataset é a próxima rodada de experimentos.

## 2. Regressão: prever `temperatura_rolamento_c`

**Configuração:** separação **temporal** pela coluna `timestamp` (os 20% mais recentes vão para o teste), validação cruzada com `TimeSeriesSplit` de 5 folds e hiperparâmetros padrão. As colunas `falha_24h` e `modo_falha` foram excluídas porque descrevem um evento futuro.

| Modelo | RMSE teste (°C) | RMSE validação cruzada | MAE teste | R² teste | R² treino | Tempo de ajuste |
|---|---|---|---|---|---|---|
| Baseline (média do treino) | 13,02 | — | 11,05 | — | — | — |
| Regressão Linear | 3,26 | 3,09 ± 0,15 | 2,54 | 0,912 | 0,929 | 0,03 s |
| Decision Tree | 4,37 | 4,40 ± 0,46 | 3,27 | 0,843 | 1,000 | 0,06 s |
| Random Forest | 3,11 | 3,30 ± 0,52 | 2,34 | 0,920 | 0,992 | 2,69 s |
| XGBoost | 2,93 | 3,01 ± 0,40 | 2,25 | 0,929 | 0,992 | 0,45 s |

**Leitura:**
- **O R² alto é esperado, não é mérito:** no gerador sintético, a temperatura do rolamento é calculada por uma fórmula a partir das próprias entradas (temperatura ambiente, corrente, vibração, rotação, porte) mais ruído.
- A Decision Tree tem R² = 1,000 no treino e 0,843 no teste, um caso claro de overfitting.
- Teste pareado (RMSE): XGBoost × Regressão Linear **não** é significativo (p corrigido = 0,775). Random Forest × XGBoost é (p = 0,029, XGBoost melhor). A Decision Tree perde de forma significativa para todos. Ressalva: no `TimeSeriesSplit` os folds têm tamanhos diferentes, então a correção usada é uma aproximação.

Os tempos de execução foram medidos em um ambiente Linux e podem variar em outras máquinas.

## Próximos experimentos

- Validar o pipeline num **dataset público real**, como o AI4I 2020 (UCI) ou o C-MAPSS da NASA.
- Medir o ganho de recall com limiar escolhido na validação, `class_weight` e `scale_pos_weight`.
- Repetir a classificação sem `motor_id` para checar a generalização para motores novos.
- Medir tempo de resposta e uso de memória com CSVs grandes (o limite de upload é 100 MB).
